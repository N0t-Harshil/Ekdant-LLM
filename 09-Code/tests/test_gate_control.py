"""v2.3 gate control (spec 07-Management/30 §15, T37): the gate opens fast at
the start (LR boost), can never exceed a soft cap, and is braked - gates
shrunk, L2 learning rates cut - when L2 shows the warning signs that
preceded the T32/T33 explosions."""

import json
import math

import pytest
import torch

import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, GatedAddOn, V2Config

KW = dict(K=4, rounds=1, steps=8, eta=0.0625, rho_min=0.05, readout="mean",
          mlp_ratio=2.0, norm_inputs=True)


def test_gate_cap_bounds_effective_gate_and_keeps_identity():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    a = GatedAddOn(32, KW, mode="mid", gate_cap=1.0)
    out, _ = a(h, None)
    assert torch.equal(out, h)                       # theta = 0 -> g = 0
    with torch.no_grad():
        a.gate.fill_(100.0)
    assert a.gate_magnitude() <= 1.0 + 1e-6          # never bursts past the cap
    with torch.no_grad():
        a.gate.fill_(0.01)
    assert a.gate_magnitude() == pytest.approx(math.tanh(0.01), rel=1e-5)


def test_capped_gate_moves_like_uncapped_at_start():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    grads = []
    for cap in (0.0, 1.0):
        torch.manual_seed(1)
        a = GatedAddOn(32, KW, mode="mid", gate_cap=cap)
        a(h, None)[0].pow(2).sum().backward()
        grads.append(a.gate.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_gate_boost_starts_high_and_fades():
    assert G.gate_boost(0, 10.0, 1000) == pytest.approx(10.0)
    assert 1.0 < G.gate_boost(500, 10.0, 1000) < 10.0
    assert G.gate_boost(1000, 10.0, 1000) == pytest.approx(1.0)
    assert G.gate_boost(5000, 10.0, 1000) == pytest.approx(1.0)
    assert G.gate_boost(0, 1.0, 1000) == pytest.approx(1.0)       # boost off


def test_brake_on_gradient_spike_and_recovery():
    b = G.GateBrake(spike=3.0, l2_max=50.0, cut=0.5, recover=1.01)
    for _ in range(30):
        assert b.update(1.0, 1.0) is False
    assert b.factor == 1.0
    assert b.update(10.0, 1.0) is True               # 10x the running average
    assert b.factor == pytest.approx(0.5) and b.events == 1
    for _ in range(10):
        b.update(1.0, 1.0)
    assert 0.5 < b.factor < 1.0                      # slow recovery
    for _ in range(500):
        b.update(1.0, 1.0)
    assert b.factor == 1.0


def test_brake_on_l2_size_and_floor():
    b = G.GateBrake(spike=3.0, l2_max=50.0, cut=0.5, min_factor=0.05)
    assert b.update(1.0, 60.0) is True               # L2's size above the limit
    for _ in range(20):
        b.update(1.0, 1e6)
    assert b.factor == pytest.approx(0.05)           # never below the floor
    assert b.update(float("nan"), 1.0) is False      # non-finite: the skip guard's job


def test_brake_state_round_trip():
    b = G.GateBrake()
    for x in (1.0, 1.2, 9.0, 1.0):
        b.update(x, 2.0)
    c = G.GateBrake()
    c.load_state_dict(b.state_dict())
    assert c.state_dict() == b.state_dict()


def test_param_groups_carry_their_kind():
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="sandbox", addon_layer=2, debate_rounds=1,
                          sandbox_mlp_ratio=2.0), tiny_llama(layers=3))
    kinds = {g["kind"] for g in m.param_groups(5e-5, 1e-4, gate_lr=1e-3)}
    assert kinds == {"backbone", "addon", "gate"}


def test_shipped_sandbox_configs_cap_the_gate():
    for name in ("v2_sandbox", "v2_sandbox_multi", "v2_sandbox_final", "v2_baseline"):
        with open(f"configs/{name}.json", encoding="utf-8") as f:
            assert json.load(f)["gate_cap"] == 1.0


# ---- T39: gate floor ramp - open the gate to a target, then hold it ----

def test_floor_zero_keeps_identity_and_old_gate():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 32)
    a = GatedAddOn(32, KW, mode="final", gate_cap=1.0)
    a.gate_floor = 0.0
    assert torch.equal(a(h, None)[0], h)


def test_floor_sets_the_gate_and_bounds_it():
    a = GatedAddOn(32, KW, mode="final", gate_cap=1.0)
    a.gate_floor = 0.7
    assert a.gate_magnitude() == pytest.approx(0.7, abs=1e-6)     # theta = 0
    with torch.no_grad():
        a.gate.fill_(-100.0)
    assert a.gate_magnitude() == pytest.approx(0.4, abs=1e-4)     # 2f - 1
    with torch.no_grad():
        a.gate.fill_(100.0)
    assert a.gate_magnitude() <= 1.0 + 1e-6


def test_ramp_opens_smoothly_then_holds():
    r = G.GateRamp(target=0.7, start=100, steps=1000)
    assert r.floor() == 0.0
    for _ in range(100):
        r.step(False)
    assert r.floor() == pytest.approx(0.0)
    for _ in range(500):
        r.step(False)
    assert 0.2 < r.floor() < 0.5                                  # half-way
    for _ in range(2000):
        r.step(False)
    assert r.floor() == pytest.approx(0.7)                        # then held


def test_ramp_backs_off_on_a_burst():
    r = G.GateRamp(target=0.7, start=0, steps=1000, backoff=0.1)
    for _ in range(800):
        r.step(False)
    before = r.floor()
    r.step(True)                                                  # brake fired
    assert r.floor() < before and r.pos == 800 - 100
    s = G.GateRamp(target=0.7, start=0, steps=1000)
    s.load_state_dict(r.state_dict())
    assert s.state_dict() == r.state_dict()


def test_ramp_off_by_default():
    args = G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"])
    assert args.gate_floor == 0.0


def test_lever_configs_change_exactly_one_setting():
    # each lever experiment changes one thing against the end placement
    with open("configs/v2_sandbox_final.json", encoding="utf-8") as f:
        base = json.load(f)
    for name, key, value in (("v2_sandbox_final_k8", "n_sandbox", 8),
                             ("v2_sandbox_final_lowest", "readout", "lowest_energy")):
        with open(f"configs/{name}.json", encoding="utf-8") as f:
            c = json.load(f)
        assert c[key] == value
        assert {k for k in c if c[k] != base.get(k)} == {key}
