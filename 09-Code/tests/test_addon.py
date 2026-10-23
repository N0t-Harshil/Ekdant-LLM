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


def test_addon_contribution_bounded_when_sandbox_output_explodes():
    # T32: v2's add-on sits after SmolLM2's final norm, so an unbounded L2
    # output went straight into the logits (sandbox run: loss in the
    # thousands, grad norm 1e11 -> inf). The consensus is RMS-normalised
    # before the gate, so L2's contribution is bounded however large its
    # internals grow.
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    addon = GatedAddOn(32, dict(SBX, rounds=1, mlp_ratio=2.0))
    with torch.no_grad():
        addon.gate.fill_(0.5)
    real = addon.sandbox.forward

    def exploding(x, mask=None):
        o = real(x, mask=mask)
        return o._replace(consensus=o.consensus * 1e6)

    addon.sandbox.forward = exploding
    out, _ = addon(h, None)
    bound = 0.5 * (32 ** 0.5 + h.abs().max()) + 1e-4
    assert torch.isfinite(out).all()
    assert (out - h).abs().max() <= bound
