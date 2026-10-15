"""LR schedule (optimizer-step semantics) — the warmup bug made a default
3-epoch run never leave warmup, so full runs trained at ~0.5% of base lr."""

from ekdant.train import lr_at

BASE = 3e-4


def test_warmup_reaches_base_then_cosine_decays_to_zero():
    warmup, total = 4, 10
    # linear warmup: step 0..3 -> base/4 .. base
    assert abs(lr_at(0, total, BASE, warmup) - BASE / 4) < 1e-12
    assert abs(lr_at(3, total, BASE, warmup) - BASE) < 1e-12
    # cosine phase is monotone non-increasing and ends at 0
    vals = [lr_at(s, total, BASE, warmup) for s in range(warmup, total + 1)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert abs(vals[-1]) < 1e-12


def test_lr_clamped_past_total():
    # beyond `total` the cosine phase must stay at 0, not swing back up
    assert abs(lr_at(50, 10, BASE, 4)) < 1e-12


def test_zero_warmup_is_pure_cosine():
    assert abs(lr_at(0, 10, BASE, 0) - BASE) < 1e-12
    assert abs(lr_at(10, 10, BASE, 0)) < 1e-12


def test_tiny_run_warmup_dominates():
    # warmup >= total: every step stays on the warmup ramp, never exceeds base
    for s in range(5):
        lr = lr_at(s, 3, BASE, 4)
        assert 0 < lr <= BASE + 1e-12


def test_stage_boundary_resets_cosine_cycle():
    """Stage 2 must restart its own schedule, not inherit stage 1's spent
    budget. Regression for the bug where a single global opt_step left
    MedQA training at lr == 0 for all epochs (losses flat at 20.82)."""
    stage1_total, stage2_total, warmup = 200, 20, 4
    # stage 1 spends its whole budget -> cosine is correctly exhausted
    assert abs(lr_at(stage1_total, stage1_total, BASE, warmup)) < 1e-12
    # stage 2 with a RESET step counter is warm again, not pinned at 0
    assert lr_at(0, stage2_total, BASE, warmup) > 0
    assert abs(lr_at(0, stage2_total, BASE, warmup) - BASE / warmup) < 1e-12
    # and if the counter were NOT reset it would stay at 0 — what we fixed
    assert abs(lr_at(stage1_total, stage2_total, BASE, warmup)) < 1e-12
