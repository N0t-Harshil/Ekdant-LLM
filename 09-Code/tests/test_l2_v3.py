"""L2 v3 (T50, T51): L2 reads the text through the model's own states,
starts and stays in the model's meaning, votes between an odd number of
trajectories, scores them against the history so far, and learns a
confidence that predicts whether the next token is right."""

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ekdant.backbone import tiny_llama  # noqa: E402
from ekdant.gen_eval import describe_l2, generate_greedy  # noqa: E402
from ekdant.model import EkdantLM, L2Cache, V2Config  # noqa: E402
from ekdant.sandbox import Sandbox, Verifier  # noqa: E402

DIM, B, T = 64, 2, 10
V3 = dict(n_sandbox=5, sandbox_context=True, sandbox_confidence=True,
          sandbox_aligned=True)
SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _box(**kw):
    torch.manual_seed(0)
    args = dict(K=5, rounds=2, norm_inputs=True, context=True, heads=4,
                confidence=True, aligned=True)
    args.update(kw)
    return Sandbox(DIM, **args).eval()


def _wake(box):
    """Non-zero weights for the zero-initialised parts (as training gives)."""
    torch.manual_seed(1)
    with torch.no_grad():
        for layer in box.debate:
            if layer.context is not None:
                layer.context.alpha.fill_(0.5)
        if box.verifier is not None:
            box.verifier.own.weight.normal_(0, 0.5)
            box.verifier.fit.weight.normal_(0, 0.5)
    return box


def test_aligned_l2_starts_in_the_models_meaning():
    # each trajectory = the model's state + a small nudge; the anchor is the
    # model's state; the old init was a random projection of it
    h = torch.randn(B, T, DIM)
    new, old = _box(), _box(aligned=False, context=False, confidence=False, K=4)
    cos = torch.nn.functional.cosine_similarity
    t_new = new.init_proj(h)
    t_old = old.init_proj(h)
    assert cos(t_new, h.unsqueeze(0), dim=-1).mean() > 0.8
    assert cos(t_old, h.unsqueeze(0), dim=-1).abs().mean() < 0.3
    assert torch.allclose(new.descent.anchor_of(h), h)
    # the trajectories still differ (the spread floor needs them apart)
    rho = new(h).aux["spread_ratio"]
    assert float(rho) > 0.01
    assert cos(new(h).consensus, h, dim=-1).mean() > 0.8
    # stored as a delta: weight decay (towards 0) pulls towards the model's state
    with torch.no_grad():
        new.init_proj.weight.zero_()
    assert torch.allclose(new.init_proj(h), h.unsqueeze(0).expand(5, *h.shape))


def test_l2_reads_the_text_causally_through_the_models_states():
    box = _wake(_box())
    h = torch.randn(1, T, DIM)
    h2 = h.clone()
    h2[0, 6] += 3.0                                  # change token 6 only
    a, b = box(h).consensus, box(h2).consensus
    assert torch.allclose(a[0, :6], b[0, :6], atol=1e-5)      # earlier: unchanged
    assert not torch.allclose(a[0, 7:], b[0, 7:], atol=1e-4)  # later: they read it
    old = _box(context=False, confidence=False, aligned=False, K=4)
    assert torch.allclose(old(h).consensus[0, 7:], old(h2).consensus[0, 7:])
    # cheap: one key projection per forward, values are the model's states
    assert box.mem_key is not None and not hasattr(box.debate[0].context, "qkv")


def test_untrained_text_read_changes_nothing():
    a = _box(context=False)
    b = _box(context=True)
    b.load_state_dict(a.state_dict(), strict=False)
    h = torch.randn(B, T, DIM)
    assert torch.allclose(a(h).consensus, b(h).consensus, atol=1e-5)


def test_left_padding_does_not_change_real_tokens():
    box = _wake(_box())
    h = torch.randn(1, 6, DIM)
    padded = torch.cat([torch.randn(1, 4, DIM), h], dim=1)
    mask = torch.tensor([[0] * 4 + [1] * 6])
    a = box(h, attn_mask=torch.ones(1, 6, dtype=torch.long)).consensus
    b = box(padded, attn_mask=mask).consensus[:, 4:]
    assert torch.allclose(a, b, atol=1e-5)


def test_reading_a_cached_prefix_equals_reading_everything():
    box = _wake(_box())
    h = torch.randn(1, T, DIM)
    full = box(h).consensus
    step = box(h[:, -1:], ctx=h).consensus           # last token, text from the cache
    assert torch.allclose(full[:, -1:], step, atol=1e-5)


def test_vote_and_history_shape_the_readout():
    # odd K: the trajectory most others agree with gets the most support
    c = torch.zeros(1, 1, DIM)
    c[..., 0] = 1.0
    traj = torch.stack([c, c, c, -c, 3 * c])          # 3 agree, 2 outliers
    sup = Verifier.support(traj)[:, 0, 0]
    assert sup[:3].min() > sup[3:].max()
    box = _box()
    w, _ = box.verifier(traj, None)
    assert float(w[:3, 0, 0].sum()) > 0.6             # the majority wins
    assert torch.allclose(w.sum(0), torch.ones(1, 1))
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", sandbox_confidence=True, n_sandbox=4).validate()
    # the history term: a trajectory that fits the text so far scores higher
    v = Verifier(DIM)
    with torch.no_grad():
        v.fit.weight.copy_(torch.eye(DIM))
        v.vote.zero_()
    hist = torch.randn(1, 1, DIM)
    fits = torch.stack([hist, -hist, torch.zeros_like(hist) + 0.01])
    w, _ = v(fits, hist)
    assert int(w[:, 0, 0].argmax()) == 0


def test_vote_support_matches_the_direct_formula():
    traj = torch.randn(5, B, T, DIM)
    t = traj.float()
    size = t.mean(0).pow(2).sum(-1)
    d = (t.unsqueeze(0) - t.unsqueeze(1)).pow(2).sum(-1) / size
    want = (torch.exp(-d).sum(1) - 1.0) / 4
    assert torch.allclose(Verifier.support(traj), want, atol=1e-4)


def test_per_token_gate_never_exceeds_the_cap():
    m = _lm(gate_cap=1.0)
    torch.nn.init.constant_(m.addon.gate, 5.0)      # capped gate ~1.0
    with torch.no_grad():
        m.addon.token_gate.bias.fill_(20.0)         # factor -> 2
    a = m.addon
    h = torch.randn(1, T, 32)
    with torch.no_grad():
        out, _ = a(h)
        s = a.sandbox(h, ctx=h)
        y = a.out_norm(s.consensus) * h.pow(2).mean(-1, keepdim=True).sqrt()
    # out = h + g_tok * (y - h): g_tok is the ratio, and it stays <= the cap
    d = y - h
    big = d.abs() > 1e-2
    ratio = (out - h)[big] / d[big]
    assert float(ratio.max()) <= 1.0 + 1e-4 and float(ratio.min()) > 0.9


def _lm(v3: bool = True, **kw):
    torch.manual_seed(0)
    cfg = V2Config(addon="sandbox", **SMALL, **(V3 if v3 else {}), **kw)
    return EkdantLM(cfg, tiny_llama()).eval()


@pytest.mark.parametrize("kw", [dict(), dict(addon_layer=1),
                                dict(addon_layers=[1, 2], addon_final=True)])
def test_v3_model_starts_as_the_backbone_and_its_parts_train(kw):
    base = EkdantLM(V2Config(addon="none", **SMALL), tiny_llama()).eval()
    m = _lm(**kw)
    base.lm.load_state_dict(m.lm.state_dict())
    ids = torch.randint(3, 64, (B, T))
    assert torch.allclose(base(ids).logits, m(ids).logits, atol=1e-5)
    for g in m.addon.gates():
        torch.nn.init.constant_(g, 0.3)
    out = m(ids, labels=ids)
    assert {"token_gate", "confidence", "learned_confidence", "vote_support"} <= set(out.aux)
    out.loss.backward()
    for s in m._all_sites():
        assert s.token_gate.weight.grad is not None and s.conf_head.weight.grad is not None


def test_confidence_loss_trains_only_the_confidence_head():
    ids = torch.randint(3, 64, (B, T))
    grads = []
    for lam in (0.0, 1.0):
        m = _lm(lambda_conf=lam)
        torch.nn.init.constant_(m.addon.gate, 0.3)
        m(ids, labels=ids).loss.backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()
                      if p.grad is not None})
    assert set(grads[1]) - set(grads[0]) == {"addon.conf_head.weight",
                                             "addon.conf_head.bias"}
    for n, g in grads[0].items():
        assert torch.allclose(g, grads[1][n], atol=1e-6), n


def test_forced_floor_holds_for_every_token():
    m = _lm(gate_force=True)
    m.addon.set_gate_floor(0.5)
    with torch.no_grad():
        m.addon.token_gate.bias.fill_(-20.0)        # gate pushed towards 0
    ids = torch.randint(3, 64, (1, T))
    with torch.no_grad():
        on = m(ids).logits
        with m.l2_off():
            off = m(ids).logits
    assert (on - off).abs().max() > 1e-3            # L2 still contributes


@pytest.mark.parametrize("kw", [dict(), dict(addon_layers=[1, 2], addon_final=True)])
def test_cached_generation_is_exact_and_batch_independent(kw):
    m = _lm(**kw)
    for s in m._all_sites():
        torch.nn.init.constant_(s.gate, 0.5)
    for layer in m._all_sites()[0].sandbox.debate:
        layer.context.alpha.data.fill_(0.5)
    o = m(torch.tensor([[5, 9, 11]]), use_cache=True)
    assert isinstance(o.aux["past_key_values"], L2Cache)
    p1, p2 = [5, 9, 11, 4], [7, 8, 3, 12, 13, 6, 10]
    alone = generate_greedy(m, [p1], 5, torch.device("cpu"), eos_id=2, pad_id=0)
    both = generate_greedy(m, [p1, p2], 5, torch.device("cpu"), eos_id=2, pad_id=0)
    assert alone[0] == both[0]
    ids = list(p1)                                   # full forwards, no cache
    for _ in range(len(alone[0])):
        ids.append(int(m(torch.tensor([ids])).logits[0, -1].argmax()))
    assert ids[len(p1):] == alone[0]


def test_old_configs_load_and_labels():
    cfg = V2Config(addon="sandbox").to_dict()
    for k in ("sandbox_context", "sandbox_heads", "sandbox_confidence",
              "sandbox_aligned", "lambda_conf"):
        cfg.pop(k)
    old = V2Config.from_dict(cfg)
    assert not old.sandbox_aligned and describe_l2(old) == "end, free"
    assert describe_l2(V2Config(addon="sandbox", **V3)) == \
        "end, free, v3 aligned+context+confidence"


def test_v3_run_end_to_end(tiny, tmp_path):
    import ekdant.train_gen as G
    from test_train_gen import _args
    cfg = json.loads(pathlib.Path(tiny).read_text())
    cfg.update(V3)
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(cfg))
    G.main(_args(str(path), tmp_path / "run"))
    r = json.loads((tmp_path / "run" / "results.json").read_text())
    assert r["config"]["sandbox_aligned"] and r["config"]["n_sandbox"] == 5
    t = r["test"]
    assert t["medquad_bench"]["ppl"] > 0 and "rouge_l" in t["medquad_bench"]
    assert t["l2_off"]["medquad_bench"]["ppl"] > 0
    diag = t["medquad_bench"]["addon"]
    assert {"confidence", "learned_confidence", "vote_support", "token_gate"} <= set(diag)


def test_shipped_v3_config():
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "v3_sandbox_end.json"
    cfg = V2Config.from_dict(json.loads(path.read_text()))
    assert cfg.sandbox_context and cfg.sandbox_confidence and cfg.sandbox_aligned
    assert cfg.n_sandbox == 5 and cfg.addon_layer == -1
    ref = json.loads((path.parent / "v2_sandbox_final.json").read_text())
    same = {k: v for k, v in ref.items() if k not in V3}
    assert all(getattr(cfg, k) == v for k, v in same.items())


from test_train_gen import tiny  # noqa: E402,F401  (shared fixture)
