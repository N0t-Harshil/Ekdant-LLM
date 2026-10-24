"""v2.2: the same L2 applied at several depths (spec 07-Management/30 §14).
One shared Sandbox; every site has its own zero-init gate and its own
input/output norms (the stream's scale differs a lot between depths).
A tiny random Llama stands in for SmolLM2."""

import copy
import json

import torch

from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _model(layers=(1, 2, 3), shared=True, addon="sandbox", seed=0):
    torch.manual_seed(seed)
    lm = tiny_llama(layers=3)
    return EkdantLM(V2Config(addon=addon, addon_layers=list(layers),
                             addon_shared=shared, **SMALL), lm).eval()


def _ids():
    torch.manual_seed(1)
    return torch.randint(3, 64, (2, 7))


def test_multisite_is_identity_at_init():
    torch.manual_seed(0)
    lm = tiny_llama(layers=3)
    base = EkdantLM(V2Config(addon="none", **SMALL), copy.deepcopy(lm)).eval()
    multi = EkdantLM(V2Config(addon="sandbox", addon_layers=[1, 2, 3], **SMALL),
                     copy.deepcopy(lm)).eval()
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(multi(ids).logits, base(ids).logits)


def test_shared_sites_reuse_one_sandbox():
    shared = _model(shared=True)
    single = EkdantLM(V2Config(addon="sandbox", addon_layer=2, **SMALL), tiny_llama(layers=3))
    per_site = 3 * 32                       # gate + in_norm + out_norm, dim 32
    assert shared.addon_params() == single.addon_params() + 2 * per_site
    sbx = {id(s.sandbox) for s in shared.addon.sites}
    assert len(sbx) == 1


def test_separate_copies_switch_gives_one_sandbox_per_site():
    sep = _model(shared=False)
    assert len({id(s.sandbox) for s in sep.addon.sites}) == 3
    assert sep.addon_params() > 2.5 * _model(shared=True).addon_params() / 1.0 - 1


def test_every_site_gate_gets_gradient():
    m = _model().train()
    ids = _ids()
    labels = ids.clone()
    m(ids, labels=labels).loss.backward()
    for s in m.addon.sites:
        assert s.gate.grad is not None and s.gate.grad.abs().sum() > 0


def test_multisite_kv_cache_step_matches_full_forward():
    m = _model()
    with torch.no_grad():
        for s in m.addon.sites:
            s.gate.fill_(0.3)
    ids = _ids()[:1]
    with torch.no_grad():
        full = m(ids).logits[:, -1]
        pre = m(ids[:, :-1], use_cache=True)
        step = m(ids[:, -1:], past_key_values=pre.aux["past_key_values"],
                 use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_all_site_gates_share_the_gate_lr_group():
    m = _model()
    groups = m.param_groups(5e-5, 1e-4, gate_lr=1e-3)
    gate_group = [g for g in groups if g["lr"] == 1e-3]
    assert len(gate_group) == 1
    assert {id(p) for p in gate_group[0]["params"]} == {id(s.gate) for s in m.addon.sites}
    seen = [id(p) for g in groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in m.parameters())


def test_gate_magnitudes_reported_per_site():
    m = _model()
    with torch.no_grad():
        for v, s in zip((0.1, 0.2, 0.3), m.addon.sites):
            s.gate.fill_(v)
    assert [round(x, 4) for x in m.addon.site_gate_magnitudes()] == [0.1, 0.2, 0.3]
    assert abs(m.addon.gate_magnitude() - 0.2) < 1e-6


def test_shipped_multi_config():
    with open("configs/v2_sandbox_multi.json", encoding="utf-8") as f:
        c = json.load(f)
    assert c["addon_layers"] == [10, 20, 30] and c["addon_shared"] is True
    with open("configs/v2_sandbox.json", encoding="utf-8") as f:
        mid = json.load(f)
    c.pop("addon_layers"), mid.pop("addon_layers", None)
    assert c == mid      # otherwise identical to the single mid-layer config
