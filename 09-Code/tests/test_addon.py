"""v2 add-on: scaled debate width and the zero-init gated residual that
makes the sandbox arm equal the backbone at step 0
(spec 07-Management/30_Design_v2_Pretrained_Backbone.md §3)."""

import torch

from ekdant.model import GatedAddOn
from ekdant.sandbox import Sandbox

SBX = dict(K=4, rounds=4, steps=8, eta=0.0625, rho_min=0.05,
           readout="mean", mlp_ratio=4.0)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_sandbox_mlp_ratio_sets_debate_width():
    assert abs(_n(Sandbox(576, rounds=4, mlp_ratio=4.0)) - 19.59e6) < 0.01e6
    assert abs(_n(Sandbox(576)) - 8.30e6) < 0.01e6


def test_gated_addon_is_identity_at_init():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    small = dict(SBX, rounds=1, mlp_ratio=2.0)
    out, _ = GatedAddOn(32, small)(h, None)
    assert torch.equal(out, h)


def test_gate_gets_gradient_at_init():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    small = dict(SBX, rounds=1, mlp_ratio=2.0)
    addon = GatedAddOn(32, small)
    out, _ = addon(h, None)
    out.pow(2).sum().backward()
    assert addon.gate.grad.abs().sum() > 0
