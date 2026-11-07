"""T56: L2 without its limits - inside the network and at the end, sized to
the stream, reading what the model knew at earlier depths - and the second
benchmark: the model writes the explanation itself."""

import json
import math
import pathlib

import pytest
import torch

import ekdant.gen_eval as M
from ekdant.backbone import tiny_llama
from ekdant.gen_data import GenDataset, GenExample
from ekdant.model import EkdantLM, V2Config

V3D = dict(n_sandbox=5, sandbox_context=True, sandbox_confidence=True,
           sandbox_aligned=True, sandbox_diverse_read=True, token_gate_min=0.5,
           sandbox_free_views=True, debate_rounds=1, sandbox_mlp_ratio=2.0, gate_cap=1.0)
V4 = dict(V3D, addon_layers=[2], addon_final=True, sandbox_stream_scale=True,
          sandbox_read_layers=[1, 3])


def _pair(layers=4, **kw):
    torch.manual_seed(0)
    bb = tiny_llama(layers=layers)
    ref = {k: v.clone() for k, v in bb.state_dict().items()}
    m = EkdantLM(V2Config(addon="sandbox", **kw), bb)
    plain = tiny_llama(layers=layers)
    plain.load_state_dict(ref)
    return m, plain


def _open(m):
    """Open every gate and read weight, and L2's text read (alpha)."""
    with torch.no_grad():
        for s in m._all_sites():
            s.gate.fill_(0.3)
            s.read_w.normal_(0, 0.5)
        for layer in m.sandboxes()[0].debate:
            layer.context.alpha.fill_(0.5)


def test_v4_sites_and_what_each_reads():
    m, _ = _pair(**V4)
    assert m._site_reads == [[1], [1, 3]]         # the inner site reads below it
    sites = m._all_sites()
    assert [tuple(s.read_w.shape) for s in sites] == [(1, 32), (2, 32)]
    assert sites[0].mode == "mid" and sites[0].stream_scale
    assert sites[1].mode == "final"
    assert sites[0].sandbox is sites[1].sandbox    # one L2, two places


def test_v4_step_zero_equals_the_backbone():
    m, plain = _pair(**V4)
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        assert torch.allclose(m.eval()(ids).logits, plain(ids).logits, atol=1e-5)


def test_v4_cached_generation_is_exact():
    m, _ = _pair(**V4)
    _open(m)
    m.eval()
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        full = m(ids).logits
        o = m(ids[:, :4], use_cache=True)
        past, outs = o.aux["past_key_values"], [o.logits]
        for t in range(4, 7):
            o = m(ids[:, t:t + 1], past_key_values=past, use_cache=True,
                  position_ids=torch.full((2, 1), t))
            past = o.aux["past_key_values"]
            outs.append(o.logits)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)


def test_deep_read_changes_the_answer_and_learns():
    m, _ = _pair(**V4)
    _open(m)
    ids = torch.randint(3, 64, (2, 7))
    m.eval()
    with torch.no_grad():
        a = m(ids).logits
        for s in m._all_sites():
            s.read_w.zero_()
        b = m(ids).logits
    assert not torch.allclose(a, b)                # the earlier depths are used
    _open(m)
    m.train()
    out = m(ids, labels=ids.clone())
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(s.read_w.grad.abs().sum() > 0 for s in m._all_sites())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_stream_scale_sizes_the_inner_site_to_the_stream():
    sizes = {}
    for scale in (False, True):
        kw = dict(V3D, addon_layer=2, sandbox_stream_scale=scale)
        m, _ = _pair(**kw)
        h = 50.0 * torch.randn(1, 5, 32)           # a stream of rms ~50
        with torch.no_grad():
            m.addon.gate.fill_(1.0)
            new, _ = m.addon(h)
        sizes[scale] = float((new - h).pow(2).mean().sqrt())
    assert sizes[True] > 10 * sizes[False]


def test_v4_validation():
    with pytest.raises(ValueError):                # a read layer cannot be a site
        V2Config(addon="sandbox", **dict(V4, sandbox_read_layers=[2])).validate()
    with pytest.raises(ValueError):                # the read needs the text read
        V2Config(addon="sandbox", sandbox_read_layers=[1]).validate()
    with pytest.raises(ValueError):                # stream scale needs an inner site
        V2Config(addon="sandbox", **dict(V3D, sandbox_stream_scale=True)).validate()
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", **dict(V4, sandbox_read_layers=[3, 1])).validate()
    with pytest.raises(ValueError):                # beyond the backbone
        _pair(**dict(V4, sandbox_read_layers=[1, 4]))


def test_old_configs_unchanged():
    m, _ = _pair(**dict(V3D))
    assert m.addon.read_w is None and not m.addon.stream_scale
    assert m._site_reads == [[]]


def test_shipped_v4_config():
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "v4_sandbox_mix.json"
    new = json.loads(path.read_text())
    cfg = V2Config.from_dict(new)
    v3d = json.loads((path.parent / "v3d_sandbox_end.json").read_text())
    assert {k for k in new if new[k] != v3d.get(k)} == {
        "addon_layers", "addon_final", "sandbox_stream_scale", "sandbox_read_layers"}
    assert cfg.addon_layers == [15] and cfg.addon_final
    assert cfg.sandbox_read_layers == [8, 22]


# --- the explanation benchmark ---------------------------------------------

def test_names_answer():
    opts = ["Vitamin A", "Vitamin B12", "Iron", "Folic acid"]
    assert M.names_answer("Vitamin B12. It is needed for ...", opts, 1) == 1
    assert M.names_answer("Iron, not vitamin B12", opts, 1) == 0     # Iron first
    assert M.names_answer("it is unclear", opts, 1) == 0
    # "Vitamin A" is no match inside "vitamin a1" words? exact words only
    assert M.names_answer("vitamin a deficiency causes night blindness", opts, 0) == 1


class _Tok:
    eos_token_id, pad_token_id = 1, 0

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 60) for c in s][:40]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_explain_evaluate_scores_every_set():
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="sandbox", **V3D), tiny_llama(layers=2))
    tok = _Tok()
    ex = [GenExample(f"q{i}?\nOptions: aa; bb; cc; dd", "bb. because") for i in range(6)]
    metas = [(["aa", "bb", "cc", "dd"], 1)] * 6
    sets = {"medmcqa_explain": (GenDataset(ex, tok, 128, 32), metas),
            "pubmedqa_explain": (GenDataset(ex, tok, 128, 32), [None] * 6)}
    r = M.explain_evaluate(m, tok, sets, torch.device("cpu"), 4, True, n_gen=3, max_new=4,
                           log=lambda *_: None)
    assert set(r) == set(M.EXPLAIN_SETS)
    a = r["medmcqa_explain"]
    assert a["on"]["ppl"] > 0 and a["off"]["ppl"] > 0
    assert a["gen"]["n"] == 3 and "named_right" in a["gen"]
    assert "named_right" not in r["pubmedqa_explain"]["gen"]
    assert len(a["gen"]["examples"]) == 3


def _explain_file(d: pathlib.Path, seed: int, named: list, nll: list):
    d.mkdir()
    rows = {"row_idx": list(range(len(nll))), "row_nll": nll,
            "row_tokens": [1] * len(nll), "ppl": math.exp(sum(nll) / len(nll))}
    gen = {"rows": [{"i": i, "rouge_l": 0.2, "token_f1": 0.3, "named": x}
                    for i, x in enumerate(named)],
           "rouge_l": 0.2, "token_f1": 0.3, "named_right": sum(named) / len(named),
           "n": len(named)}
    m = {"version": M.MCQ_VERSION, "seed": seed, "l2": "end",
         "pubmedqa": {"accuracy": 0.5, "n": 2, "rows": [{"q": 0, "right": 1}, {"q": 1, "right": 0}]},
         "explain": {"medmcqa_explain": {"on": rows, "gen": gen}}}
    (d / "mcq.json").write_text(json.dumps(m))
    return str(d)


def test_explain_table_and_gain_rows(tmp_path):
    b17 = _explain_file(tmp_path / "v_baseline", 17, [1] * 20 + [0] * 80, [1.0] * 100)
    m17 = _explain_file(tmp_path / "v4_mix", 17, [1] * 40 + [0] * 60, [0.9] * 100)
    t = M.explain_table([b17, m17])
    assert "| **v_baseline** (baseline) | MedMCQA explanation | 2.718 | - | 20.0 | 30.0 | 20.0 |" in t
    assert "| v4_mix | MedMCQA explanation | 2.460 (-9.5%*) | - | 20.0 (+0.0) | 30.0 (+0.0) | 40.0 (+20.0*) |" in t
    b23 = _explain_file(tmp_path / "b23", 23, [1] * 20 + [0] * 80, [1.0] * 100)
    m23 = _explain_file(tmp_path / "m23", 23, [1] * 32 + [0] * 68, [1.0] * 100)
    g = M.gain_report([b17, b23], [("v4", [m17, m23])])
    # gain 16 points: +80 % relative, 20 % of the baseline's 80 % misses fixed
    # (numbers chosen away from a rounding boundary: x.x5 rounds differently
    # on different machines)
    assert "| v4 | MedMCQA explanation: names the right answer | 20.0 | 36.0 | +16.00 [" in g
    assert "| +80.0 % | +20.0 % | +20.0* / +12.0* |" in g
    assert "| v4 | MedMCQA explanation ppl (lower = better) | 2.718 | 2.589 |" in g
    assert "Can the model explain?" in M.full_report([b17, m17])


def test_old_files_are_scored_again_for_the_explanations(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "mcq.json").write_text(json.dumps({"version": 5, "l2": "end", "pubmedqa": {}}))
    assert "explanation benchmark (T56)" in M._score_state(str(d), ["pubmedqa"])
