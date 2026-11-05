"""T43: open the gate fast; on a burst go back to the last good state and
open it again more slowly and a little less far.  The trainer keeps an
in-memory snapshot (model + optimiser + ramp position) every N steps."""

import json

import pytest
import torch

import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def test_ramp_rollback_goes_back_slower_and_lower():
    r = G.GateRamp(target=0.5, start=0, steps=100)
    for _ in range(80):
        r.step(False)
    r.rollback(40)
    assert r.pos == 40 and r.steps == 200          # re-opens at half the speed
    assert r.target == pytest.approx(0.45)         # and a little less far
    assert r.rollbacks == 1
    fresh = G.GateRamp(target=0.5, start=0, steps=100)
    fresh.pos = 40
    assert r.floor() == pytest.approx(fresh.floor())   # resumes from the snapshot's floor
    for _ in range(20):
        r.step(False)
        fresh.step(False)
    assert r.floor() < fresh.floor()               # then rises more slowly
    for _ in range(10):
        r.rollback(40)
    assert r.target == pytest.approx(0.4)          # never below the minimum (0.8 x 0.5)


def test_ramp_minimum_can_be_set():
    r = G.GateRamp(target=0.5, start=0, steps=100, step_down=0.1, min_target=0.3)
    for _ in range(5):
        r.rollback(0)
    assert r.target == pytest.approx(0.3)


def test_ramp_state_round_trips_and_old_state_loads():
    r = G.GateRamp(target=0.5, start=0, steps=100)
    r.rollback(10)
    s = G.GateRamp(target=0.5, start=0, steps=100)
    s.load_state_dict(r.state_dict())
    assert s.state_dict() == r.state_dict()
    old = G.GateRamp(target=0.7, start=0, steps=100)
    old.load_state_dict({"pos": 7})                 # a T39 checkpoint
    assert old.pos == 7 and old.target == 0.7 and old.steps == 100


def test_snapshot_restores_model_and_optimizer_exactly():
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="sandbox", **SMALL), tiny_llama(layers=2))
    opt = torch.optim.AdamW(m.param_groups(1e-3, 1e-3, gate_lr=1e-2))
    ids = torch.randint(3, 64, (2, 7))

    def step():
        opt.zero_grad()
        m(ids, labels=ids.clone()).loss.backward()
        opt.step()

    step()
    snap = G.take_snapshot(m, opt)
    want = {k: v.clone() for k, v in m.state_dict().items()}
    assert all(v.device.type == "cpu" for v in snap["model"].values())
    step(); step()
    G.restore_snapshot(m, opt, snap)
    assert all(torch.equal(m.state_dict()[k], want[k]) for k in want)
    # the optimiser continues exactly as from the snapshot
    step()
    after_a = {k: v.clone() for k, v in m.state_dict().items()}
    G.restore_snapshot(m, opt, snap)
    step()
    assert all(torch.equal(m.state_dict()[k], after_a[k]) for k in after_a)


def test_rollback_is_off_by_default():
    a = G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"])
    assert a.gate_rollback is False and a.snapshot_every == 100


def test_burst_with_rollback_goes_back_and_is_recorded(tiny, tmp_path, monkeypatch):
    real = G.clip_grads
    calls = {"n": 0}

    def spiky(model, max_norm):
        bn, an = real(model, max_norm)
        calls["n"] += 1
        return (bn, an * 1e3) if calls["n"] == 5 else (bn, an)

    monkeypatch.setattr(G, "clip_grads", spiky)
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--brake-warmup", "2", "--gate-rollback",
                                   "--snapshot-every", "2", "--gate-floor", "0.5",
                                   "--gate-force", "--gate-floor-start", "0",
                                   "--gate-floor-steps", "4"]))
    r = json.loads((out / "results.json").read_text())
    assert r["cost"]["rollbacks"] == 1
    ramp = torch.load(out / "last.pt", weights_only=False)["gate_ramp"]
    assert ramp["rollbacks"] == 1 and ramp["target"] == pytest.approx(0.45)
    assert ramp["steps"] == 8
    assert min(r["test"]["gate_per_site"]) >= 0.45 - 1e-6



def test_repeated_rollbacks_never_drop_the_floor_towards_zero():
    # the v26_mix_forced log (T45): the ramp had finished (floor 0.45), then
    # one burst after another; each rollback doubled the ramp length and,
    # measured from the start, the floor at the snapshot fell to ~0 - the
    # forced gates were released and the model broke (loss 34)
    r = G.GateRamp(target=0.5, start=100, steps=500)
    for _ in range(2600):
        r.step(False)
    for _ in range(200):
        r.rollback(2600)
        assert r.floor() >= 0.4 - 1e-9             # never below the minimum
    assert r.floor() == pytest.approx(0.4)
    assert r.steps <= 16 * 500                     # the length is capped


def test_rollback_during_the_rise_continues_from_where_it_was():
    r = G.GateRamp(target=0.5, start=0, steps=100)
    for _ in range(30):
        r.step(False)
    was = r.floor()
    r.rollback(30)
    assert r.floor() == pytest.approx(was)
    for _ in range(1000):
        r.step(False)
    assert r.floor() == pytest.approx(0.45)        # reaches the lowered target


def test_brake_average_follows_a_lasting_rise():
    b = G.GateBrake(spike=3.0, warmup=2)
    for _ in range(30):
        b.update(0.4, 1.0)
    bursts = [b.update(2.0, 1.0) for _ in range(60)]
    assert bursts[0] is True
    assert bursts[-1] is False                     # 2.0 became the new normal


def test_rollback_limit_falls_back_to_the_brake(tiny, tmp_path, monkeypatch):
    # every step after the 4th bursts: without a limit this rolled back forever
    real = G.clip_grads
    calls = {"n": 0}

    def spiky(model, max_norm):
        bn, an = real(model, max_norm)
        calls["n"] += 1
        return (bn, an * 1e3) if calls["n"] >= 5 else (bn, an)

    monkeypatch.setattr(G, "clip_grads", spiky)
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--brake-warmup", "2", "--gate-rollback",
                                   "--snapshot-every", "2", "--gate-floor", "0.5",
                                   "--gate-force", "--gate-floor-start", "0",
                                   "--gate-floor-steps", "4", "--max-rollbacks", "2"]))
    r = json.loads((out / "results.json").read_text())
    # 2 in stage 1 (the limit), not hundreds; stage 2 re-learns what is
    # normal (T53 rewarm), and the steady 1e3x level is then no burst
    assert r["cost"]["rollbacks"] == 2
    assert min(r["test"]["gate_per_site"]) >= 0.4 - 1e-6

from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
