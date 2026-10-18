"""Sequential A/B training: MedMCQA first, then a MedQA 60/40 fine-tune.

Phases (one invocation, strictly in order — never trained jointly):

    phase 1   train on the FULL MedMCQA train split (182,496),
              then test on the MedMCQA validation split (4,183 —
              public test split ships cop=-1, unlabelled, T18)
              -> results["phase1"]["medmcqa_val"]
    phase 2   fine-tune on 60% of MedQA train (FT_FRAC, seeded
              permutation — both A/B arms get the identical partition)
    final     benchmark the complementary 40% of MedQA exactly once
              (never trained on) + the official MedQA test split
              (also untouched) + the MedMCQA validation score again
              (retention after the fine-tune)

Checkpoint selection: best.pt tracks the untouched MedMCQA validation
during phase 2; at the end the better of best.pt vs last.pt (same
validation) is loaded and runs the final benchmark.

Both variants share the identical backbone.  Only the L2 sandbox layer
differs, so any accuracy gap is attributable to the sandbox alone.

Usage (single GPU):
    python -m ekdant.train --variant sandbox  --out runs/sandbox
    python -m ekdant.train --variant baseline --out runs/baseline

Multi-GPU (one process per GPU, run one variant at a time):
    torchrun --nproc_per_node=4 -m ekdant.train --variant sandbox --out runs/sandbox
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ekdant.data import (
    LETTER_IDS,
    MCQDataset,
    Tokenizer,
    bind_letter_ids,
    collate,
    format_prompt,
    load_examples,
)
from ekdant.model import EkdantConfig, EkdantModel, estimate_model_size

# Sequential protocol — phases in order, never trained jointly:
#   phase 1: full MedMCQA train (182,496), then the MedMCQA validation
#            result is recorded (public test split is unlabelled, T18)
#   phase 2: fine-tune on 60% of MedQA train (FT_FRAC)
#   final:   benchmark the complementary 40% of MedQA exactly once (it
#            is never trained on) + official MedQA test (untouched) +
#            the MedMCQA validation score again (retention)
# The 60/40 partition comes from a --seed-locked permutation, so both
# A/B arms train on identical data and benchmark the identical holdout
# (fairness rule).
STAGES = (("medmcqa", "train"), ("medqa_ft", "train"))
# Sandbox health numbers averaged per epoch into history / results.json
# (sandbox arm only; the baseline has no aux).
DIAG_KEYS = ("anchor_energy", "trajectory_variance", "spread_ratio")
FT_FRAC = 0.6
# MedMCQA's public test split ships cop=-1 (unlabelled) — bug T18 made the
# metric degenerate to "A-prediction rate". The labelled validation split
# (4,183 rows) is never trained on and serves as the untouched
# checkpoint-selection set (best.pt / final better-of selection).


def _accumulate_diag(acc: Dict[str, float], out, weight: int) -> None:
    aux = getattr(out, "aux", None) or {}
    for k in DIAG_KEYS:
        v = aux.get(k)
        if v is not None:
            acc[k] = acc.get(k, 0.0) + float(v.detach()) * weight


def _mean_diag(acc: Dict[str, float], n: int) -> Dict[str, float]:
    return {k: v / max(1, n) for k, v in acc.items()}


def log(msg: str) -> None:
    if dist.get_rank() == 0 if dist.is_initialized() else True:
        print(msg, flush=True)


def setup_dist() -> bool:
    if "RANK" not in os.environ:
        return False
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True


def build_dataloader(ds, cfg, batch_size, shuffle, distributed, seed=0,
                     workers=4):
    sampler = (DistributedSampler(ds, shuffle=shuffle, seed=seed)
               if distributed else None)
    # Dedicated generator: model init consumes a different amount of global
    # RNG per arm (sandbox vs baseline have different param counts), so
    # shuffle order would otherwise diverge between the A/B arms. Seeded
    # with --seed, both arms see identical row order. (After a --resume the
    # generator restarts, so order after the break can differ from an
    # uninterrupted run — order noise only, never the partition.)
    gen = None
    if shuffle and sampler is None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
        sampler=sampler, num_workers=workers, pin_memory=True,
        drop_last=shuffle, collate_fn=partial(collate, pad_id=cfg.pad_id),
        persistent_workers=(workers > 0), generator=gen,
    )


def make_dataset(name, split, tokenizer, cfg, limit=None):
    ex = load_examples(name, split, limit=limit)
    return MCQDataset(ex, tokenizer, max_len=cfg.max_seq_len)


def split_frac(ds, frac: float, seed: int = 17):
    """Seed-locked (head, tail) partition of a dataset.

    ``head`` = ``int(len * frac)`` rows (the training role), ``tail`` = the
    complement (the benchmark role — never trained on). One permutation,
    seeded with ``--seed``: both A/B arms get the identical partition
    (fairness rule).
    """
    n = len(ds)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(n * frac)
    return Subset(ds, perm[:cut]), Subset(ds, perm[cut:])


def load_stage(name, split, tokenizer, cfg, limit=None, seed=17):
    """Dataset for a stage. ``medqa_ft`` = the 60% fine-tune partition of
    the MedQA train split (the complementary 40% stays untouched until
    the final benchmark)."""
    ds = make_dataset("medqa" if name == "medqa_ft" else name,
                      split, tokenizer, cfg, limit)
    if name == "medqa_ft":
        head, _tail = split_frac(ds, FT_FRAC, seed)
        return head
    return ds


def stage_hparams(args, si: int):
    """(lr, epochs) for stage ``si``.  The MedQA fine-tune phase has its own
    knobs (``--ft-lr`` / ``--ft-epochs``) that default to ``--lr`` /
    ``--epochs`` — a run without them behaves exactly as before."""
    ft = STAGES[si][0] == "medqa_ft"
    lr = args.ft_lr if ft and args.ft_lr is not None else args.lr
    epochs = args.ft_epochs if ft and args.ft_epochs is not None \
        else args.epochs
    return lr, epochs


def stage_offsets(epochs_each):
    """Flat ``completed``-index offset of each stage (sum of earlier stages'
    epoch counts).  With equal per-stage epochs this is the legacy
    ``si * epochs`` arithmetic, so older checkpoints resume unchanged."""
    return [sum(epochs_each[:si]) for si in range(len(epochs_each))]


def lr_at(step, total, base, warmup):
    """LR for an optimizer step: linear warmup, then cosine to 0."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def train_stage(model, loader, opt, device, epoch, scaler, autocast,
                opt_step, total_opt, base_lr, warmup, accum,
                stage_name="", log_every=100, answer_weight=0.0):
    """One epoch. LR is set per optimizer step, not per epoch (the old
    per-epoch counter made warmup never complete — fixed; see run.md §8).

    Emits a progress line every ``log_every`` optimizer steps:
        [stage] ep0 step 1200/3809 (32%) loss=.. lr=.. gnorm=.. tok/s=.. eta=..
    ``loss`` is an EMA of the per-batch CE (LM loss + ``answer_weight`` ×
    answer-letter CE when that weight is > 0); ``gnorm`` is the pre-clip
    global grad norm (clip threshold 1.0 — a sustained value far above 1
    means the clip is active every step; near 0 means training stalled).
    """
    model.train()
    opt.zero_grad(set_to_none=True)  # never inherit a prior epoch's tail
    total, ntok, steps, nsamp = 0.0, 0, 0, 0
    gnorm_sum, gnorm_n = 0.0, 0
    diag: Dict[str, float] = {}
    loss_ema = None
    last_lr = opt.param_groups[0]["lr"]
    start_opt = opt_step
    t_start = time.time()
    t_window, ntok_window, next_at = t_start, 0, opt_step + log_every
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast():
            out = model(batch["input_ids"], labels=batch["labels"],
                        answer_pos=batch["answer_pos"],
                        answer_ids=batch["answer_ids"],
                        answer_weight=answer_weight,
                        attention_mask=batch["attention_mask"])
            loss = out.loss / accum
        scaler.scale(loss).backward()
        bsz_tok = int(batch["attention_mask"].sum())
        if (steps + 1) % accum == 0:
            lr = lr_at(opt_step, total_opt, base_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            last_lr = lr
            opt_step += 1
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            gnorm_sum += gnorm
            gnorm_n += 1
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            lv = out.loss.item()
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if log_every > 0 and opt_step >= next_at:
                now = time.time()
                tok_s = ntok_window / max(1e-9, now - t_window)
                made = max(1, opt_step - start_opt)
                eta = (now - t_start) / made * max(0, total_opt - opt_step)
                log(f"[{stage_name}] ep{epoch} step {opt_step}/{total_opt} "
                    f"({100.0 * opt_step / max(1, total_opt):.0f}%) "
                    f"loss={loss_ema:.4f} lr={lr:.2e} gnorm={gnorm:.2f} "
                    f"tok/s={tok_s:.0f} eta={fmt_duration(eta)}")
                next_at += log_every
                t_window, ntok_window = now, 0
        total += out.loss.item() * batch["input_ids"].size(0)
        _accumulate_diag(diag, out, batch["input_ids"].size(0))
        nsamp += batch["input_ids"].size(0)
        ntok += bsz_tok
        ntok_window += bsz_tok
        steps += 1
    if steps > 0 and steps % accum != 0:
        # Flush the trailing partial accumulation group. The schedule
        # budgets ceil(steps/accum) optimizer steps per epoch; without
        # this the tail micro-batches' grads leaked into the NEXT epoch's
        # first step (and the budgeted step was never taken — with
        # small --limit runs, zero steps ever fired).
        lr = lr_at(opt_step, total_opt, base_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        last_lr = lr
        opt_step += 1
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        gnorm_sum += gnorm
        gnorm_n += 1
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    avg_gnorm = gnorm_sum / max(1, gnorm_n)
    return (total / max(1, nsamp), ntok, opt_step, last_lr, avg_gnorm,
            _mean_diag(diag, nsamp))


@torch.no_grad()
def evaluate(model, loader, device, autocast,
             answer_weight: float = 0.0) -> Dict[str, float]:
    model.eval()
    correct = seen = 0
    loss_sum = n = 0
    diag: Dict[str, float] = {}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast():
            out = model(batch["input_ids"], labels=batch["labels"],
                        answer_pos=batch["answer_pos"],
                        answer_ids=batch["answer_ids"],
                        answer_weight=answer_weight,
                        attention_mask=batch.get("attention_mask"))
            logits = out.logits
        loss_sum += out.loss.item() * batch["input_ids"].size(0)
        _accumulate_diag(diag, out, batch["input_ids"].size(0))
        n += batch["input_ids"].size(0)
        # logits[t] predicts token t+1 -> score the answer at p via logits[p-1]
        got = logits[torch.arange(len(logits), device=device),
                     batch["answer_pos"] - 1]
        pred = got[:, LETTER_IDS].argmax(-1)
        correct += int((pred == batch["answer_target"]).sum())
        seen += len(pred)
    r = {"loss": loss_sum / max(1, n), "accuracy": correct / max(1, seen),
         "n": seen}
    if diag:
        r["sandbox"] = _mean_diag(diag, n)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sandbox", "baseline"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ft-lr", type=float, default=None,
                    help="phase-2 (MedQA fine-tune) cosine base LR; "
                         "default: same as --lr")
    ap.add_argument("--ft-epochs", type=int, default=None,
                    help="phase-2 epochs; default: same as --epochs")
    ap.add_argument("--answer-weight", type=float, default=0.0,
                    help="weight of the answer-letter CE added to the LM "
                         "loss (0 = legacy pure LM loss; 10 recommended - "
                         "the letter is only ~1 token in ~34 MedMCQA / "
                         "~173 MedQA tokens, so plain CE dilutes the "
                         "decision signal ~97-99%)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=100,
                    help="print a progress line every N optimizer steps "
                         "(0 = epoch summaries only)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate the untouched MedMCQA validation set "
                         "every N epochs: in phase 1 for monitoring only "
                         "(history), in the MedQA fine-tune also for "
                         "best.pt selection (0 = final only)")
    ap.add_argument("--resume", action="store_true",
                    help="continue <out>/last.pt (skips finished stage/epochs)")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training; run the final protocol "
                         "(better-of selection + benchmark)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    distributed = setup_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() \
        else torch.device("cpu")
    is_main = (not distributed) or dist.get_rank() == 0

    resume_mode = args.resume or args.eval_only
    if not resume_mode and is_main:
        stale = os.path.join(args.out, "last.pt")
        if os.path.exists(stale):
            log(f"error: {stale} exists but --resume was not passed - a fresh "
                f"run would overwrite an existing training state. Pass "
                f"--resume, use a new --out directory, or delete {args.out}")
            raise SystemExit(1)
    run_cfg = os.path.join(args.out, "config.json")
    if resume_mode and os.path.exists(run_cfg):
        with open(run_cfg, "r", encoding="utf-8") as f:
            cfg = EkdantConfig.from_dict(json.load(f))
        if is_main:
            log(f"config from {run_cfg} (exact config the run started with)")
    else:
        cfg_path = args.config or os.path.join(
            "configs", f"{args.variant}_50m.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = EkdantConfig.from_dict(json.load(f))
    if args.variant == "baseline":
        cfg.use_sandbox = False

    if is_main:
        os.makedirs(args.out, exist_ok=True)
        log(f"variant={args.variant}  use_sandbox={cfg.use_sandbox}")

    tok_path = os.path.join(args.out, "tokenizer.json")
    if not os.path.exists(tok_path) and args.eval_only:
        log(f"error: {tok_path} not found — an eval-only run needs an "
            f"existing --out directory from a finished/paused training run")
        raise SystemExit(1)
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.load(tok_path)
        if is_main:
            log(f"loaded existing tokenizer ({tokenizer.vocab_size})")
    else:
        probe = load_examples("medmcqa", "train", limit=20000)
        texts = [format_prompt(e.question, e.options) + f" {e.answer}"
                 for e in probe]
        tokenizer = Tokenizer.train(texts, cfg.vocab_size, tok_path)
        if is_main:
            log(f"trained tokenizer -> {tok_path} ({tokenizer.vocab_size})")
    cfg.pad_id = tokenizer.pad_id
    bind_letter_ids(tokenizer)

    model = EkdantModel(cfg).to(device)
    start_pair, step, history = 0, 0, []
    opt_step = 0
    best_acc = -1.0
    if resume_mode:
        ckpt_path = os.path.join(args.out, "last.pt")
        if not os.path.exists(ckpt_path):
            log(f"error: {ckpt_path} not found — nothing to resume/evaluate")
            raise SystemExit(1)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        step = int(ckpt.get("step", 0))
        start_pair = int(ckpt.get("completed", 0))
        history = list(ckpt.get("history", []))
        opt_step = int(ckpt.get("opt_step", 0))
        best_acc = float(ckpt.get("best_medmcqa", -1.0))
        if is_main:
            verb = "eval-only: loaded" if args.eval_only else "resumed"
            log(f"{verb} {ckpt_path}  completed_pairs={start_pair} "
                f"step={step} opt_step={opt_step}")
    if is_main:
        s = estimate_model_size(cfg)
        log(f"params total={s['total']/1e6:.2f}M "
            f"T={s['t']/1e6:.2f}M L2={s['sandbox']/1e6:.2f}M")
    if distributed:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=bool(cfg.use_sandbox))
    raw = model.module if distributed else model

    if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            if is_main:
                log("torch.compile enabled")
        except Exception as exc:
            if is_main:
                log(f"compile skipped: {exc}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    use_amp = device.type == "cuda"
    if use_amp:
        bf16_ok = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        if is_main and not bf16_ok:
            log("bf16 unsupported on this GPU (e.g. T4/P100) — "
                "using fp16 autocast + loss scaling")
        autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype))
        use_scaler = not bf16_ok  # fp16 needs loss scaling; bf16 does not
    else:
        autocast = (lambda: torch.autocast("cpu", enabled=False))
        use_scaler = False
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:  # torch < 2.3 (Kaggle images ship 2.2)
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if is_main:
        with open(os.path.join(args.out, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    if args.eval_only and is_main:
        log("eval-only: skipping training, running the final benchmark protocol")
    results = {"variant": args.variant, "use_sandbox": cfg.use_sandbox,
               "params": estimate_model_size(cfg)["total"], "history": history,
               "test": {}}
    if resume_mode:
        # keep the phase-1 record across resume / eval-only rewrites
        prev_path = os.path.join(args.out, "results.json")
        if os.path.exists(prev_path):
            try:
                with open(prev_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if isinstance(prev.get("phase1"), dict):
                    results["phase1"] = prev["phase1"]
            except Exception:
                pass

    def _write_results() -> None:
        if is_main:
            with open(os.path.join(args.out, "results.json"), "w",
                      encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    def eval_phase1() -> None:
        """Phase-1 record: MedMCQA validation right after MedMCQA training
        (before the MedQA fine-tune). Runs once; crash-safe via
        results.json."""
        if args.eval_only or not is_main:
            return
        if results.get("phase1", {}).get("medmcqa_val"):
            return
        tds = make_dataset("medmcqa", "validation", tokenizer, cfg, args.limit)
        tloader = build_dataloader(tds, cfg, args.batch_size, False, False,
                                   seed=args.seed, workers=0)
        r = evaluate(raw, tloader, device, autocast, args.answer_weight)
        results["phase1"] = {"medmcqa_val": r}
        log(f"[phase1 test medmcqa_val] acc={r['accuracy']*100:.2f}% "
            f"loss={r['loss']:.4f} n={r['n']}  "
            f"(after MedMCQA training, before MedQA fine-tune)")
        _write_results()

    if not args.eval_only:
        epochs_each = [stage_hparams(args, si)[1] for si in range(len(STAGES))]
        offsets = stage_offsets(epochs_each)
        for si, (stage_name, split) in enumerate(STAGES):
            stage_lr, stage_epochs = stage_hparams(args, si)
            first_epoch = max(0, start_pair - offsets[si])
            if first_epoch >= stage_epochs:
                if is_main:
                    log(f"stage {stage_name}: already complete, skipping")
                if si == 0:
                    eval_phase1()
                continue
            ds = load_stage(stage_name, split, tokenizer, cfg, args.limit,
                            seed=args.seed)
            loader = build_dataloader(ds, cfg, args.batch_size, True,
                                      distributed, seed=args.seed)
            steps_per_epoch = max(1, len(loader))
            opt_per_epoch = max(1, math.ceil(steps_per_epoch / args.grad_accum))
            total_opt = opt_per_epoch * stage_epochs
            if is_main:
                log(f"\n=== stage {stage_name}  n={len(ds)} "
                    f"(epoch {first_epoch}+) | {opt_per_epoch} opt/epoch x "
                    f"{stage_epochs} epochs = {total_opt} steps, "
                    f"lr {stage_lr:g} cosine, warmup {args.warmup} ===")
            stage_opt_step = 0  # each stage gets its own cosine cycle
            for epoch in range(first_epoch, stage_epochs):
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(epoch)
                t0 = time.time()
                avg, ntok, stage_opt_step, last_lr, avg_gnorm, diag = train_stage(
                    model, loader, opt, device, epoch, scaler, autocast,
                    stage_opt_step, total_opt, stage_lr, args.warmup,
                    args.grad_accum, stage_name=stage_name,
                    log_every=args.log_every,
                    answer_weight=args.answer_weight)
                opt_step = stage_opt_step  # for resume/ckpt bookkeeping
                step += 1
                completed = offsets[si] + epoch + 1
                if is_main:
                    log(f"[{stage_name}] ep{epoch} done loss={avg:.4f} "
                        f"gnorm={avg_gnorm:.2f} lr_end={last_lr:.2e} "
                        f"tok/s={ntok/max(1,time.time()-t0):.0f} "
                        f"time={fmt_duration(time.time()-t0)}")
                    history.append({"stage": stage_name, "epoch": epoch,
                                    "loss": avg, "lr": last_lr,
                                    "gnorm": avg_gnorm})
                    if diag:
                        history[-1]["sandbox"] = diag
                        log(f"[{stage_name}] ep{epoch} sandbox "
                            + " ".join(f"{k}={v:.4g}" for k, v in diag.items()))
                    torch.save({"model": raw.state_dict(),
                                "cfg": cfg.to_dict(),
                                "step": step, "completed": completed,
                                "opt_step": opt_step, "history": history,
                                "best_medmcqa": best_acc},
                               os.path.join(args.out, "last.pt"))
                    if (args.eval_every > 0 and stage_name == "medmcqa"
                            and ((epoch + 1) % args.eval_every == 0
                                 or epoch + 1 == stage_epochs)):
                        # Monitoring only (history): makes phase-1
                        # overfitting visible.  Never selects a checkpoint.
                        vds = make_dataset("medmcqa", "validation",
                                           tokenizer, cfg, args.limit)
                        vloader = build_dataloader(vds, cfg, args.batch_size,
                                                   False, False,
                                                   seed=args.seed, workers=0)
                        r = evaluate(raw, vloader, device, autocast,
                                     args.answer_weight)
                        history[-1]["test"] = {"medmcqa_val": r}
                        log(f"[{stage_name}] ep{epoch} monitor medmcqa_val "
                            f"acc={r['accuracy']*100:.2f}% "
                            f"loss={r['loss']:.4f} n={r['n']}")
                        _write_results()
                    if (args.eval_every > 0
                            and stage_name == "medqa_ft"
                            and ((epoch + 1) % args.eval_every == 0
                                 or epoch + 1 == stage_epochs)):
                        # Selection set: the untouched MedMCQA validation
                        # split.  The 40% benchmark partition is scored
                        # exactly once, at the end.
                        vds = make_dataset("medmcqa", "validation",
                                           tokenizer, cfg, args.limit)
                        vloader = build_dataloader(vds, cfg, args.batch_size,
                                                   False, False,
                                                   seed=args.seed, workers=0)
                        r = evaluate(raw, vloader, device, autocast,
                                     args.answer_weight)
                        history[-1]["test"] = {"medmcqa_val": r}
                        results["test"] = {"medmcqa_val": r}
                        log(f"[{stage_name}] ep{epoch} eval medmcqa_val "
                            f"acc={r['accuracy']*100:.2f}% "
                            f"loss={r['loss']:.4f} n={r['n']}")
                        cur = r["accuracy"]
                        if cur > best_acc:
                            best_acc = cur
                            torch.save({"model": raw.state_dict(),
                                        "cfg": cfg.to_dict(),
                                        "step": step, "completed": completed,
                                        "opt_step": opt_step,
                                        "history": history,
                                        "best_medmcqa": best_acc},
                                       os.path.join(args.out, "best.pt"))
                            log(f"[{stage_name}] ep{epoch} new best "
                                f"medmcqa_val {cur*100:.2f}% -> best.pt")
                        _write_results()
            if si == 0:
                eval_phase1()

    # Final protocol: checkpoint selection = better of best.pt vs last.pt
    # on the untouched MedMCQA validation ("change the best with last if
    # it is good"), then score the never-trained-on 40% of MedQA exactly
    # once, plus the official MedQA test split (also untouched).
    val_ds = make_dataset("medmcqa", "validation", tokenizer, cfg, args.limit)
    val_loader = build_dataloader(val_ds, cfg, args.batch_size, False, False,
                                  seed=args.seed, workers=0)
    scores = {"last.pt": evaluate(raw, val_loader, device, autocast,
                                  args.answer_weight)}
    last_state = {k: v.detach().cpu().clone()
                  for k, v in raw.state_dict().items()}
    best_path = os.path.join(args.out, "best.pt")
    bckpt = None
    if os.path.exists(best_path):
        bckpt = torch.load(best_path, map_location="cpu")
        raw.load_state_dict(bckpt["model"])
        scores["best.pt"] = evaluate(raw, val_loader, device, autocast,
                                     args.answer_weight)
    if is_main:
        parts = "  ".join(f"{k}={v['accuracy']*100:.2f}%"
                          for k, v in scores.items())
        log(f"selection on medmcqa_val: {parts}")
    # keep best.pt unless last.pt is strictly better on the selection set
    chosen = "best.pt" if bckpt is not None else "last.pt"
    if bckpt is not None and \
            scores["last.pt"]["accuracy"] > scores["best.pt"]["accuracy"]:
        chosen = "last.pt"
    raw.load_state_dict(bckpt["model"] if chosen == "best.pt" else last_state)
    selected = {"checkpoint": chosen, "selected_on": "medmcqa_val",
                "best_val_accuracy":
                    scores.get("best.pt", {}).get("accuracy"),
                "last_val_accuracy": scores["last.pt"]["accuracy"]}
    if is_main:
        log(f"final benchmark uses {chosen} (medmcqa_val "
            f"{scores[chosen]['accuracy']*100:.2f}%)")
        if args.eval_only and "medmcqa_val" not in results.get("phase1", {}):
            log("note: phase-1 MedMCQA-validation result not in "
                "results.json (only a training run records it)")
    results["selected"] = selected
    results["test"]["medmcqa_val"] = scores[chosen]
    eval_phase1()  # no-op when already recorded / for eval-only

    # The benchmark: 40% of MedQA train, never trained on — scored once.
    medqa_full = make_dataset("medqa", "train", tokenizer, cfg, args.limit)
    _, bench = split_frac(medqa_full, FT_FRAC, args.seed)
    finals = [("medqa_bench", bench),
              ("medqa_test", make_dataset("medqa", "test", tokenizer, cfg,
                                          args.limit))]
    for name, ds_ in finals:
        loader = build_dataloader(ds_, cfg, args.batch_size, False, False,
                                  seed=args.seed, workers=0)
        r = evaluate(raw, loader, device, autocast, args.answer_weight)
        results["test"][name] = r
        if is_main:
            log(f"[test {name}] acc={r['accuracy']*100:.2f}% "
                f"loss={r['loss']:.4f} n={r['n']}")

    if is_main:
        _write_results()
        log(f"\nwrote {args.out}/results.json")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
