"""T55: L2 v3d - v3 with v22's free views (independent starting views and a
learned anchor), and the gain report: how much better than the baseline,
all seeds together."""

import json
import math
import pathlib

import pytest
import torch

import ekdant.gen_eval as M
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

V3B = dict(n_sandbox=5, sandbox_context=True, sandbox_confidence=True,
           sandbox_aligned=True, sandbox_diverse_read=True, token_gate_min=0.5)
SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _model(**kw):
    return EkdantLM(V2Config(addon="sandbox", **SMALL, **V3B, **kw), tiny_llama(layers=2))


def test_free_views_use_v22_init_and_anchor_but_keep_the_rest_of_v3():
    torch.manual_seed(0)
    m = _model(sandbox_free_views=True)
    box = m.addon.sandbox
    assert not box.init_proj.aligned and not box.descent.aligned      # v22's views
    assert m.addon.aligned                                            # output at the model's scale
    assert box.verifier is not None and box.mem_key is not None       # v3's read + verifier
    assert m.addon.token_gate is not None
    v3b = _model().addon.sandbox
    assert v3b.init_proj.aligned and v3b.descent.aligned              # default unchanged


def test_free_views_need_the_v3_l2():
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", sandbox_free_views=True).validate()


def test_free_views_spread_more_than_aligned_views_at_init():
    torch.manual_seed(0)
    h = torch.randn(2, 6, _model().addon.sandbox.dim)
    rho = {}
    for free in (False, True):
        torch.manual_seed(1)
        m = _model(sandbox_free_views=free)
        with torch.no_grad():
            out = m.addon.sandbox(h)
        rho[free] = float(out.aux["spread_ratio"])
    assert rho[True] > rho[False]


def test_free_views_step_zero_equals_the_backbone_and_trains():
    torch.manual_seed(0)
    bb = tiny_llama(layers=2)
    ref = {k: v.clone() for k, v in bb.state_dict().items()}
    m = EkdantLM(V2Config(addon="sandbox", **SMALL, **V3B, sandbox_free_views=True), bb)
    plain = tiny_llama(layers=2)
    plain.load_state_dict(ref)
    ids = torch.randint(3, 64, (2, 7))
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, plain(ids).logits, atol=1e-5)
    m.train()
    out = m(ids, labels=ids.clone())
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_label_and_shipped_v3d_config():
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "v3d_sandbox_end.json"
    new = json.loads(path.read_text())
    cfg = V2Config.from_dict(new)
    assert cfg.sandbox_free_views and cfg.sandbox_train_passes == 1
    v3b = json.loads((path.parent / "v3b_sandbox_end.json").read_text())
    assert {k for k in new if new[k] != v3b.get(k)} == {"sandbox_free_views"}
    assert "free views" in M.describe_l2(cfg)


def _write(d: pathlib.Path, seed: int, right: list, nll: list, groups=None):
    d.mkdir()
    rows = [{"q": i, "right": r} for i, r in enumerate(right)]
    acc = sum(right) / len(right)
    m = {"version": M.MCQ_VERSION, "seed": seed, "l2": "end",
         "pubmedqa": {"accuracy": acc, "rows": rows},
         "medquad_ppl": {"on": {"ppl": math.exp(sum(nll) / len(nll)),
                                "row_idx": list(range(len(nll))),
                                "row_nll": nll, "row_tokens": [1] * len(nll)},
                         "row_group": groups or {str(i): f"g{i}" for i in range(len(nll))}}}
    (d / "mcq.json").write_text(json.dumps(m))
    return str(d)


def test_gain_report_pools_the_seeds(tmp_path):
    n = 200
    b17 = _write(tmp_path / "b17", 17, [1] * 100 + [0] * 100, [1.0] * n)   # 50 %
    b23 = _write(tmp_path / "b23", 23, [1] * 90 + [0] * 110, [1.0] * n)    # 45 %
    m17 = _write(tmp_path / "m17", 17, [1] * 120 + [0] * 80, [0.9] * n)    # 60 %
    m23 = _write(tmp_path / "m23", 23, [1] * 100 + [0] * 100, [0.9] * n)   # 50 %
    t = M.gain_report([b17, b23], [("v3d", [m17, m23])])
    # baseline 47.5 %, model 55.0 %: +7.5 points, +15.8 % relative,
    # 14.3 % of the baseline's 52.5 % wrong answers fixed; real on both seeds
    assert "| v3d | PubMedQA | 47.5 | 55.0 | +7.50 [" in t
    assert "| +15.8 % | +14.3 % | +10.0* / +5.0* |" in t
    assert "| v3d | Average | 47.5 | 55.0 | +7.50 [" in t
    ppl = f"{100 * (math.exp(-0.1) - 1):+.2f} %"
    assert f"| v3d | MedQuAD ppl (lower = better) | 2.718 | 2.460 | {ppl} [" in t
    assert "seeds 17, 23 together" in t


def test_gain_report_no_gain_is_not_starred(tmp_path):
    right = [1, 0] * 50
    b = _write(tmp_path / "b", 17, right, [1.0] * 100)
    m = _write(tmp_path / "m", 17, right[1:] + right[:1], [1.0] * 100)
    t = M.gain_report([b], [("same", [m])])
    line = next(x for x in t.splitlines() if x.startswith("| same | PubMedQA"))
    assert "+0.00 [" in line and "]*" not in line


def test_gain_report_checks_the_seed_order(tmp_path):
    b17 = _write(tmp_path / "b17", 17, [1, 0], [1.0, 1.0])
    b23 = _write(tmp_path / "b23", 23, [1, 0], [1.0, 1.0])
    m17 = _write(tmp_path / "m17", 17, [1, 1], [1.0, 1.0])
    m23 = _write(tmp_path / "m23", 23, [1, 1], [1.0, 1.0])
    with pytest.raises(ValueError, match="seed order"):
        M.gain_report([b17, b23], [("x", [m23, m17])])
    with pytest.raises(ValueError):
        M.gain_report([b17, b23], [("x", [m17])])


def test_gain_cli_writes_markdown(tmp_path):
    b = _write(tmp_path / "b", 17, [1, 0, 0, 0], [1.0] * 4)
    m = _write(tmp_path / "m", 17, [1, 1, 0, 0], [1.0] * 4)
    out = tmp_path / "gain.md"
    M.gain_main(["--baseline", b, "--model", "v3d", m, "--markdown", str(out)])
    assert "| v3d | PubMedQA | 25.0 | 50.0 | +25.00 [" in out.read_text()
