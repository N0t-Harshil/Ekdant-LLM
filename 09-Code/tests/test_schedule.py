"""LR schedule (optimizer-step semantics) — the warmup bug made a default
3-epoch run never leave warmup, so full runs trained at ~0.5% of base lr.
Also covers the per-phase knobs (--ft-lr / --ft-epochs) and the flat
resume index they share across stages."""

import argparse

from ekdant.train import lr_at, stage_hparams, stage_offsets

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


def _args(**kw):
    base = {"lr": BASE, "epochs": 100, "ft_lr": None, "ft_epochs": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_stage_hparams_defaults_preserve_legacy_behavior():
    """Without --ft-lr/--ft-epochs both phases share --lr/--epochs — a run
    that doesn't pass the knobs behaves exactly as before they existed."""
    a = _args()
    assert stage_hparams(a, 0) == (BASE, 100)
    assert stage_hparams(a, 1) == (BASE, 100)


def test_stage_hparams_overrides_apply_to_medqa_finetune_only():
    """The anti-overfit knobs must never touch the MedMCQA phase."""
    a = _args(ft_lr=5e-5, ft_epochs=10)
    assert stage_hparams(a, 0) == (BASE, 100)
    assert stage_hparams(a, 1) == (5e-5, 10)


def test_stage_offsets_resume_arithmetic():
    """completed = offset + epoch + 1 stays a strictly rising flat index
    across stages; equal epochs reproduce the legacy si * epochs values so
    older checkpoints resume unchanged (no skipped/re-run epochs)."""
    assert stage_offsets([100, 100]) == [0, 100]
    offs = stage_offsets([100, 10])
    assert offs == [0, 100]
    totals = [offs[s] + e for s, e in enumerate((100, 10))]
    assert totals == [100, 110]


def test_train_stage_flushes_trailing_group_and_leaks_no_grads():
    """The schedule budgets ceil(steps/accum) optimizer steps per epoch;
    the trailing partial group used to be skipped, and its leftover grads
    leaked into the next epoch's first step (with small --limit runs,
    zero steps ever fired)."""
    import torch
    from ekdant.model import EkdantConfig, EkdantModel
    from ekdant.train import train_stage

    torch.manual_seed(0)
    cfg = EkdantConfig(vocab_size=64, dim=32, n_t=1, n_heads=2,
                       n_kv_heads=1, window=8, ffn_hidden=64,
                       max_seq_len=32, use_sandbox=False,
                       dropout=0.0).validate()
    model = EkdantModel(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    steps_taken = []
    real_step = opt.step
    opt.step = lambda *a, **k: (steps_taken.append(1), real_step(*a, **k))[1]
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    autocast = lambda: torch.autocast("cpu", enabled=False)

    def batch():
        ids = torch.randint(1, 60, (2, 8))
        return {"input_ids": ids, "labels": ids.clone(),
                "attention_mask": torch.ones_like(ids),
                "answer_pos": torch.full((2,), 3, dtype=torch.long),
                "answer_ids": torch.full((2,), 5, dtype=torch.long)}

    loader = [batch() for _ in range(5)]  # 5 micro-batches, accum 2
    opt_step = 0
    for epoch in range(2):
        _, _, opt_step, _, _ = train_stage(
            model, loader, opt, torch.device("cpu"), epoch, scaler,
            autocast, opt_step=opt_step,
            total_opt=3, base_lr=1e-2, warmup=0, accum=2,
            stage_name="t", log_every=0)
        # ceil(5/2) = 3 steps each epoch, including the flushed 1-micro tail
        assert opt_step == 3 * (epoch + 1)
        # flush clears grads — nothing leaks into the next epoch
        assert all(p.grad is None for p in model.parameters())
    assert len(steps_taken) == 6


def test_dataloader_shuffle_generator_is_seeded(monkeypatch):
    """Both A/B arms must see identical row order: model init consumes a
    different amount of global RNG per arm, so shuffle now uses a
    dedicated --seed-locked generator instead of the global RNG."""
    from types import SimpleNamespace

    import torch

    import ekdant.train as T

    monkeypatch.setattr(T, "LETTER_IDS", [1, 2, 3, 4])
    rows = [{"input_ids": [i + 1, i + 2], "answer_pos": 1, "answer_id": 1,
             "letter_idx": 0} for i in range(8)]
    cfg = SimpleNamespace(pad_id=0)

    def first_batch(seed):
        ld = T.build_dataloader(rows, cfg, batch_size=4, shuffle=True,
                                distributed=False, seed=seed, workers=0)
        assert ld.generator is not None
        return next(iter(ld))["input_ids"].tolist()

    assert first_batch(17) == first_batch(17)
    assert first_batch(17) != first_batch(99)
