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
    assert r.floor() < fresh.floor()               # lower than before at the same point
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


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
