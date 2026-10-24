"""v2.1: L2 inside the backbone (after decoder layer k), so its output feeds
the later layers and, through their attention, the other tokens - the
role the thesis gives the M3 write-back (spec 07-Management/30 §13).
A tiny random Llama stands in for SmolLM2."""

import copy

import pytest
import torch

import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _pair(layer, layers=3, seed=0):
    torch.manual_seed(seed)
    lm = tiny_llama(layers=layers)
    base = EkdantLM(V2Config(addon="none", addon_layer=layer, **SMALL), copy.deepcopy(lm)).eval()
    sbx = EkdantLM(V2Config(addon="sandbox", addon_layer=layer, **SMALL), copy.deepcopy(lm)).eval()
    return base, sbx


def _ids():
    torch.manual_seed(1)
    return torch.randint(3, 64, (2, 7))


def test_mid_addon_is_identity_at_init():
    base, sbx = _pair(layer=2)
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(sbx(ids).logits, base(ids).logits)


def _effect_on_later_tokens(layer):
    """Max change of the logits at positions > 2 when L2's output is
    perturbed at position 2 only."""
    _, m = _pair(layer=layer)
    with torch.no_grad():
        m.addon.gate.fill_(0.5)
    ids = _ids()
    with torch.no_grad():
        ref = m(ids).logits
    real = m.addon.sandbox.forward

    def bump(x, mask=None):
        o = real(x, mask=mask)
        c = o.consensus.clone()
        c[:, 2] += 3.0
        return o._replace(consensus=c)

    m.addon.sandbox.forward = bump
    with torch.no_grad():
        out = m(ids).logits
    return float((out[:, 3:] - ref[:, 3:]).abs().max()), float((out[:, :2] - ref[:, :2]).abs().max())


def test_mid_addon_reaches_later_tokens_but_final_placement_cannot():
    later_mid, earlier_mid = _effect_on_later_tokens(layer=2)
    later_end, _ = _effect_on_later_tokens(layer=-1)
    assert later_mid > 1e-4          # L2 at token 2 changes predictions at tokens 3+
    assert earlier_mid == 0.0        # ... and never the past (causal)
    assert later_end == 0.0          # at the end, L2 is a dead end


def test_mid_kv_cache_step_matches_full_forward():
    _, m = _pair(layer=2)
    with torch.no_grad():
        m.addon.gate.fill_(0.5)
    ids = _ids()[:1]
    with torch.no_grad():
        full = m(ids).logits[:, -1]
        pre = m(ids[:, :-1], use_cache=True)
        step = m(ids[:, -1:], past_key_values=pre.aux["past_key_values"],
                 use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_mid_addon_input_is_normalised_against_massive_activations():
    # SmolLM2's residual stream mid-network has rms ~100 and outliers ~2e4;
    # L2 reads RMSNorm(h), so its contribution stays bounded.
    from ekdant.model import GatedAddOn
    from ekdant.sandbox import RHO_MIN_DEFAULT  # noqa: F401
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    h[:, 0, 3] = 2e4
    addon = GatedAddOn(32, dict(K=4, rounds=1, steps=8, eta=0.0625, rho_min=0.05,
                                readout="mean", mlp_ratio=2.0, norm_inputs=True),
                       mode="mid")
    with torch.no_grad():
        addon.gate.fill_(0.5)
    out, _ = addon(h, None)
    assert torch.isfinite(out).all()
    assert (out - h).abs().max() <= 0.5 * 32 ** 0.5 + 1e-4


def test_addon_layer_is_validated():
    with pytest.raises(ValueError):
        EkdantLM(V2Config(addon="sandbox", addon_layer=0, **SMALL), tiny_llama(layers=3))
    with pytest.raises(ValueError):
        EkdantLM(V2Config(addon="sandbox", addon_layer=4, **SMALL), tiny_llama(layers=3))


def test_gate_has_its_own_learning_rate():
    _, m = _pair(layer=2)
    groups = m.param_groups(5e-5, 1e-4, gate_lr=1e-3)
    gate = [g for g in groups if any(p is m.addon.gate for p in g["params"])]
    assert len(gate) == 1 and gate[0]["lr"] == 1e-3 and len(gate[0]["params"]) == 1
    assert gate[0]["weight_decay"] == 0.0
    others = [p for g in groups if g is not gate[0] for p in g["params"]]
    assert sorted(map(id, others + [m.addon.gate])) == sorted(map(id, m.parameters()))


def test_gate_lr_flag_defaults_to_1e_3():
    args = G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"])
    assert args.gate_lr == 1e-3


def test_shipped_configs_put_l2_after_layer_20():
    import json
    for arm in ("baseline", "sandbox"):
        with open(f"configs/v2_{arm}.json", encoding="utf-8") as f:
            assert json.load(f)["addon_layer"] == 20
    with open("configs/v2_sandbox_final.json", encoding="utf-8") as f:
        assert json.load(f)["addon_layer"] == -1     # the v2.0 placement (the tie)
