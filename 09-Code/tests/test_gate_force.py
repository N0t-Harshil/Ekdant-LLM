"""T42: forced L2 - the model must use L2, it is no longer an option.
With ``gate_force`` the gate floor is a hard minimum (the gate can open
further, never close below it), L2's output scale cannot be shrunk to zero,
and inside the network L2's output is sized to the residual stream, so a
gate of 0.5 means L2 adds half as much as the stream already holds."""

import copy
import json

import pytest
import torch

import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, GatedAddOn, V2Config

KW = dict(K=4, rounds=1, steps=8, eta=0.0625, rho_min=0.05, readout="mean",
          mlp_ratio=2.0, norm_inputs=True)
SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def test_forced_floor_is_a_hard_minimum():
    a = GatedAddOn(32, KW, mode="final", gate_cap=1.0, force=True)
    a.set_gate_floor(0.5)
    with torch.no_grad():
        a.gate.fill_(-5.0)                       # the model tries to close it
    assert float(a.effective_gate().min()) >= 0.5 - 1e-6
    with torch.no_grad():
        a.gate.fill_(5.0)                        # opening further is allowed
    assert float(a.effective_gate().min()) > 0.99


def test_unforced_floor_can_still_be_undone():
    # the T39 soft floor, unchanged: why forcing needs its own switch
    a = GatedAddOn(32, KW, mode="final", gate_cap=1.0)
    a.set_gate_floor(0.5)
    with torch.no_grad():
        a.gate.fill_(-5.0)
    assert float(a.effective_gate().max()) < 0.01


def test_forced_gate_still_learns_below_the_floor():
    # straight-through: the raw gate keeps its gradient, so it can come back
    torch.manual_seed(0)
    a = GatedAddOn(32, KW, mode="final", gate_cap=1.0, force=True)
    a.set_gate_floor(0.5)
    with torch.no_grad():
        a.gate.fill_(-0.5)
    out, _ = a(torch.randn(2, 5, 32), None)
    out.pow(2).sum().backward()
    assert a.gate.grad is not None and a.gate.grad.abs().sum() > 0


def test_forced_output_scale_is_frozen():
    forced = GatedAddOn(32, KW, mode="mid", force=True)
    free = GatedAddOn(32, KW, mode="mid")
    assert not forced.out_norm.weight.requires_grad
    assert free.out_norm.weight.requires_grad
    assert forced.in_norm.weight.requires_grad


def test_forced_mid_output_is_sized_to_the_stream():
    torch.manual_seed(0)
    a = GatedAddOn(32, KW, mode="mid", force=True).eval()
    a.set_gate_floor(0.5)
    h = torch.randn(1, 6, 32)
    with torch.no_grad():
        small, _ = a(h, None)
        big, _ = a(100 * h, None)
    # L2's addition grows with the stream (in_norm makes L2 itself scale-free)
    assert torch.allclose(big - 100 * h, 100 * (small - h), rtol=1e-3, atol=1e-3)
    # and is half the stream's size per token at g = 0.5
    rms = lambda x: x.pow(2).mean(-1).sqrt()  # noqa: E731
    assert torch.allclose(rms(small - h), 0.5 * rms(h), rtol=1e-3)


def test_forced_model_is_identity_until_the_floor_rises():
    torch.manual_seed(0)
    lm = tiny_llama(layers=3)
    base = EkdantLM(V2Config(addon="none", **SMALL), copy.deepcopy(lm)).eval()
    mix = EkdantLM(V2Config(addon="sandbox", addon_layers=[1, 2], addon_final=True,
                            gate_force=True, **SMALL), copy.deepcopy(lm)).eval()
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        assert torch.equal(mix(ids).logits, base(ids).logits)
        mix.addon.set_gate_floor(0.5)
        assert not torch.allclose(mix(ids).logits, base(ids).logits)
    assert all(s.force for s in mix.addon._all())
    assert all(g >= 0.5 - 1e-6 for g in mix.addon.site_gate_magnitudes())


def test_gate_force_is_off_by_default():
    assert V2Config().gate_force is False
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).gate_force is False


def test_gate_force_needs_a_floor(tiny, tmp_path):
    with pytest.raises(SystemExit):
        G.main(_args(tiny, tmp_path / "run", extra=["--gate-force"]))


def test_forced_run_keeps_l2_open_and_eval_only_keeps_the_floor(tiny, tmp_path):
    cfg = json.loads(open(tiny).read())
    cfg.update(addon_layers=[1, 2], addon_final=True)
    path = tmp_path / "mix.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "run"
    force = ["--gate-floor", "0.5", "--gate-force", "--gate-floor-start", "0",
             "--gate-floor-steps", "2"]
    G.main(_args(str(path), out, extra=force))
    r = json.loads((out / "results.json").read_text())
    assert r["config"]["gate_force"] is True
    assert min(r["test"]["gate_per_site"]) >= 0.5 - 1e-6
    # --eval-only must rebuild the forced model with its floor, not floor 0
    G.main(_args(str(path), out, extra=["--eval-only"]))
    r2 = json.loads((out / "results.json").read_text())
    assert r2["test"]["gate_per_site"] == r["test"]["gate_per_site"]
    assert r2["test"]["medquad_bench"]["nll"] == pytest.approx(r["test"]["medquad_bench"]["nll"])


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
