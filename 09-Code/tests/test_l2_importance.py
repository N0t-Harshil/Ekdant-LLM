"""T44: does the model depend on L2?  (1) The L2-off test: after training,
score the benchmark again with every L2 site switched off; the NLL rise is
L2's importance, with a paired-bootstrap CI.  (2) L2 warm-up: SmolLM2 is
frozen for the first N steps of phase 1, so L2 learns to be useful before
it has to compete with the pretrained backbone."""

import json

import pytest
import torch

import ekdant.gen_eval as E
import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _pair(**kw):
    torch.manual_seed(0)
    lm = tiny_llama(layers=3)
    state = {k: v.clone() for k, v in lm.state_dict().items()}
    base_lm = tiny_llama(layers=3)
    base_lm.load_state_dict(state)
    base = EkdantLM(V2Config(addon="none", **SMALL), base_lm).eval()
    m = EkdantLM(V2Config(addon="sandbox", **SMALL, **kw), lm).eval()
    with torch.no_grad():
        for g in (m.addon.gates()):
            g.fill_(0.5)
    return base, m


@pytest.mark.parametrize("kw", [{}, {"addon_layer": 2},
                                {"addon_layers": [1, 3], "addon_final": True}])
def test_l2_off_gives_exactly_the_backbone_then_restores(kw):
    base, m = _pair(**kw)
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        on = m(ids).logits
        with m.l2_off():
            off = m(ids).logits
        again = m(ids).logits
    assert torch.equal(off, base(ids).logits)
    assert not torch.allclose(on, off)
    assert torch.equal(again, on)


def test_l2_off_is_a_no_op_for_the_baseline():
    base, _ = _pair()
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        ref = base(ids).logits
        with base.l2_off():
            assert torch.equal(base(ids).logits, ref)


def test_run_reports_l2_importance(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    r = json.loads((out / "results.json").read_text())
    off = r["test"]["l2_off"]
    assert off["medquad_bench"]["ppl"] > 0 and off["pubmedqa_held"]["ppl"] > 0
    imp = off["importance"]
    assert imp["lo"] <= imp["diff"] <= imp["hi"]
    assert imp["diff"] == pytest.approx(off["medquad_bench"]["nll"]
                                        - r["test"]["medquad_bench"]["nll"], abs=1e-9)
    base = tmp_path / "base"
    G.main(_args(tiny, base, variant="baseline"))
    assert "l2_off" not in json.loads((base / "results.json").read_text())["test"]


def test_compare_shows_l2_off_and_handles_old_results(tiny, tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    G.main(_args(tiny, a, variant="baseline"))
    G.main(_args(tiny, b))
    capsys.readouterr()
    E.compare([str(a), str(b)])
    text = capsys.readouterr().out
    assert "L2 off" in text and "b: L2 off" in text


def test_warmup_freezes_the_backbone_only(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--l2-warmup", "1000", "--ft-epochs", "0"]))
    torch.manual_seed(0)
    init = tiny_llama().state_dict()
    last = torch.load(out / "last.pt", weights_only=False)["model"]
    for k, v in init.items():
        assert torch.equal(last["lm." + k], v), k          # SmolLM2 untouched
    assert any(not torch.equal(last[k], torch.zeros_like(last[k]))
               for k in last if k.endswith("gate"))         # L2 learned


def test_warmup_then_joint_training_moves_the_backbone(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--l2-warmup", "2", "--ft-epochs", "0"]))
    torch.manual_seed(0)
    init = tiny_llama().state_dict()
    last = torch.load(out / "last.pt", weights_only=False)["model"]
    assert any(not torch.equal(last["lm." + k], v) for k, v in init.items())


def test_warmup_is_off_by_default_and_refused_for_the_baseline(tiny, tmp_path):
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).l2_warmup == 0
    with pytest.raises(SystemExit):
        G.main(_args(tiny, tmp_path / "b", variant="baseline", extra=["--l2-warmup", "5"]))


def test_runs_started_before_t42_t44_still_resume(tiny, tmp_path):
    # a run launched with older code has none of the new flags in last.pt
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    ck = torch.load(out / "last.pt", weights_only=False)
    for k in ("gate_force", "gate_rollback", "snapshot_every", "floor_slowdown",
              "floor_step_down", "gate_floor_min", "l2_warmup"):
        ck["run_args"].pop(k)
    ck["cfg"].pop("gate_force")
    ck["gate_ramp"] = {"pos": ck["gate_ramp"]["pos"]}
    torch.save(ck, out / "last.pt")
    G.main(_args(tiny, out, extra=["--resume"]))
    assert "l2_off" in json.loads((out / "results.json").read_text())["test"]


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
