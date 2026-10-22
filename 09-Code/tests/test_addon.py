"""v2 add-on blocks: scaled debate width, size-matched MLP control, and the
zero-init gated residual that makes every arm equal the backbone at step 0
(spec 07-Management/30_Design_v2_Pretrained_Backbone.md §3)."""

import torch

from ekdant.model import GatedAddOn, mlp_hidden_for
from ekdant.sandbox import Sandbox

SBX = dict(K=4, rounds=4, steps=8, eta=0.0625, rho_min=0.05,
           readout="mean", mlp_ratio=4.0)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_sandbox_mlp_ratio_sets_debate_width():
    assert abs(_n(Sandbox(576, rounds=4, mlp_ratio=4.0)) - 19.59e6) < 0.01e6
    assert abs(_n(Sandbox(576)) - 8.30e6) < 0.01e6


def test_mlp_control_matches_sandbox_params():
    sbx = GatedAddOn(576, "sandbox", SBX, mlp_hidden=0)
    n_sbx = _n(sbx) - sbx.gate.numel()
    mlp = GatedAddOn(576, "mlp", SBX, mlp_hidden=mlp_hidden_for(n_sbx, 576))
    n_mlp = _n(mlp) - mlp.gate.numel()
    assert abs(n_mlp - n_sbx) / n_sbx < 0.01


def test_gated_addon_is_identity_at_init():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    small = dict(SBX, rounds=1, mlp_ratio=2.0)
    for kind in ("sandbox", "mlp"):
        out, _ = GatedAddOn(32, kind, small, mlp_hidden=48)(h, None)
        assert torch.equal(out, h), kind


def test_gate_gets_gradient_at_init():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    small = dict(SBX, rounds=1, mlp_ratio=2.0)
    for kind in ("sandbox", "mlp"):
        addon = GatedAddOn(32, kind, small, mlp_hidden=48)
        out, _ = addon(h, None)
        out.pow(2).sum().backward()
        assert addon.gate.grad.abs().sum() > 0, kind
