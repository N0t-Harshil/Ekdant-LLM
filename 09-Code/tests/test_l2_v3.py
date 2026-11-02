"""L2 v3 (T50): L2 reads the earlier tokens, weights its trajectories by a
verifier, and reports a per-token confidence that drives a per-token gate."""

import copy
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ekdant.backbone import tiny_llama  # noqa: E402
from ekdant.gen_eval import describe_l2, generate_greedy  # noqa: E402
from ekdant.model import EkdantLM, V2Config  # noqa: E402
from ekdant.sandbox import Sandbox  # noqa: E402

DIM, B, T = 64, 2, 10
V3 = dict(sandbox_context=True, sandbox_confidence=True)
SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _boxes():
    torch.manual_seed(0)
    old = Sandbox(DIM, rounds=2, norm_inputs=True).eval()
    new = Sandbox(DIM, rounds=2, norm_inputs=True, context=True, heads=4,
                  confidence=True).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and missing
    return old, new


def _wake(box):
    """Give the zero-initialised v3 parts non-zero weights (as training would)."""
    torch.manual_seed(1)
    with torch.no_grad():
        for layer in box.debate:
            layer.context.out.weight.normal_(0, 0.2)
        box.verifier[1].weight.normal_(0, 0.5)
    return box


def test_untrained_v3_parts_change_nothing():
    old, new = _boxes()
    h = torch.randn(B, T, DIM)
    a, b = old(h), new(h)
    assert torch.allclose(a.consensus, b.consensus, atol=1e-5)


def test_context_read_is_causal_and_sees_earlier_tokens():
    _, box = _boxes()
    box = _wake(box)
    h = torch.randn(1, T, DIM)
    h2 = h.clone()
    h2[0, 6] += 3.0                                  # change token 6 only
    a, b = box(h).consensus, box(h2).consensus
    assert torch.allclose(a[0, :6], b[0, :6], atol=1e-5)      # earlier: unchanged
    assert not torch.allclose(a[0, 7:], b[0, 7:], atol=1e-4)  # later: they read it
    # the v2 L2 is per token: later positions would not move at all
    old = Sandbox(DIM, rounds=2, norm_inputs=True).eval()
    assert torch.allclose(old(h).consensus[0, 7:], old(h2).consensus[0, 7:])


def test_left_padding_does_not_change_real_tokens():
    _, box = _boxes()
    box = _wake(box)
    h = torch.randn(1, 6, DIM)
    padded = torch.cat([torch.randn(1, 4, DIM), h], dim=1)
    mask = torch.tensor([[0] * 4 + [1] * 6])
    a = box(h, attn_mask=torch.ones(1, 6, dtype=torch.long)).consensus
    b = box(padded, attn_mask=mask).consensus[:, 4:]
    assert torch.allclose(a, b, atol=1e-5)


def test_confidence_is_one_when_trajectories_agree_and_lower_otherwise():
    c = torch.randn(B, T, DIM)
    same = c.unsqueeze(0).expand(4, B, T, DIM)
    assert torch.allclose(Sandbox._confidence(same, c), torch.ones(B, T))
    spread = same + torch.randn(4, B, T, DIM)
    conf = Sandbox._confidence(spread, spread.mean(0))
    assert (conf < 1).all() and (conf > 0).all() and not conf.requires_grad
    _, box = _boxes()
    out = _wake(box)(torch.randn(B, T, DIM))
    assert out.confidence.shape == (B, T)
    assert {"confidence", "verifier_top_weight"} <= set(out.aux)
    assert 0.25 - 1e-6 <= float(out.aux["verifier_top_weight"]) <= 1.0


def _lm(v3: bool, **kw):
    torch.manual_seed(0)
    lm = tiny_llama()
    cfg = V2Config(addon="sandbox", **SMALL, **(V3 if v3 else {}), **kw)
    return EkdantLM(cfg, lm).eval()


@pytest.mark.parametrize("kw", [dict(), dict(addon_layer=1),
                                dict(addon_layers=[1, 2], addon_final=True)])
def test_v3_model_starts_as_the_v2_model_and_its_gate_trains(kw):
    v2, v3 = _lm(False, **kw), _lm(True, **kw)
    v3.load_state_dict(v2.state_dict(), strict=False)
    for g in v2.addon.gates() + v3.addon.gates():
        torch.nn.init.constant_(g, 0.3)
    ids = torch.randint(3, 64, (B, T))
    assert torch.allclose(v2(ids).logits, v3(ids).logits, atol=1e-5)
    out = v3(ids, labels=ids)
    assert "token_gate" in out.aux and "confidence" in out.aux
    out.loss.backward()
    sites = [v3.addon] if not kw.get("addon_layers") else v3.addon._all()
    assert all(s.token_gate.weight.grad is not None for s in sites)


def test_forced_floor_holds_for_every_token():
    m = _lm(True, gate_force=True)
    m.addon.set_gate_floor(0.5)
    with torch.no_grad():
        m.addon.token_gate.bias.fill_(-20.0)        # gate pushed towards 0
    seen = {}
    real = m.addon.forward

    def spy(h, mask=None, attn_mask=None):
        out, aux = real(h, mask, attn_mask)
        seen["shift"] = (out - h).abs().mean()
        return out, aux
    m.addon.forward = spy
    m(torch.randint(3, 64, (1, T)))
    assert float(seen["shift"]) > 0                 # L2 still contributes


def test_generation_recomputes_the_sequence_and_is_batch_independent():
    m = _lm(True)
    for layer in m.addon.sandbox.debate:
        torch.nn.init.normal_(layer.context.out.weight, 0, 0.2)
    torch.nn.init.constant_(m.addon.gate, 0.5)
    assert m.needs_full_context
    with pytest.raises(ValueError):
        m(torch.randint(3, 64, (1, 1)), past_key_values=object())
    p1, p2 = [5, 9, 11, 4], [7, 8, 3, 12, 13, 6, 10]
    alone = generate_greedy(m, [p1], 5, torch.device("cpu"), eos_id=2, pad_id=0)
    both = generate_greedy(m, [p1, p2], 5, torch.device("cpu"), eos_id=2, pad_id=0)
    assert alone[0] == both[0]
    # the same tokens as a plain argmax over full forwards
    ids = list(p1)
    for _ in range(len(alone[0])):
        nxt = int(m(torch.tensor([ids])).logits[0, -1].argmax())
        ids.append(nxt)
    assert ids[len(p1):] == alone[0]


def test_old_configs_load_and_labels():
    cfg = V2Config(addon="sandbox").to_dict()
    for k in ("sandbox_context", "sandbox_heads", "sandbox_confidence"):
        cfg.pop(k)
    old = V2Config.from_dict(cfg)
    assert not old.sandbox_context and describe_l2(old) == "end, free"
    assert describe_l2(V2Config(addon="sandbox", **V3)) == "end, free, v3 context+confidence"
    assert not _lm(False).needs_full_context


def test_v3_run_end_to_end(tiny, tmp_path):
    import json
    import ekdant.train_gen as G
    from test_train_gen import _args
    cfg = json.loads(pathlib.Path(tiny).read_text())
    cfg.update(V3)
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(cfg))
    G.main(_args(str(path), tmp_path / "run"))
    r = json.loads((tmp_path / "run" / "results.json").read_text())
    assert r["config"]["sandbox_context"] and r["config"]["sandbox_confidence"]
    t = r["test"]
    assert t["medquad_bench"]["ppl"] > 0 and "rouge_l" in t["medquad_bench"]
    assert t["l2_off"]["medquad_bench"]["ppl"] > 0


def test_shipped_v3_config():
    import json
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "v3_sandbox_end.json"
    cfg = V2Config.from_dict(json.loads(path.read_text()))
    assert cfg.sandbox_context and cfg.sandbox_confidence and cfg.addon_layer == -1
    ref = json.loads((path.parent / "v2_sandbox_final.json").read_text())
    same = {k: v for k, v in ref.items() if k not in V3}
    assert all(getattr(cfg, k) == v for k, v in same.items())

from test_train_gen import tiny  # noqa: E402,F401  (shared fixture)
