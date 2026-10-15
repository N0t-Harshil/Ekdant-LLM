"""Two-stage A/B training: MedMCQA then MedQA, baseline vs sandbox.

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
from torch.utils.data import DataLoader, DistributedSampler

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

STAGES = (("medmcqa", "train"), ("medqa", "train"))
# MedMCQA's public test split ships cop=-1 (unlabelled) — bug T18 made the
# metric degenerate to "A-prediction rate". Standard practice: eval on the
# labelled validation split (4,183 rows).
TESTS = (("medmcqa", "validation"), ("medqa", "test"))


def log(msg: str) -> None:
    if dist.get_rank() == 0 if dist.is_initialized() else True:
        print(msg, flush=True)


def setup_dist() -> bool:
    if "RANK" not in os.environ:
        return False
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True


def build_dataloader(ds, cfg, batch_size, shuffle, distributed, seed=0):
    sampler = (DistributedSampler(ds, shuffle=shuffle, seed=seed)
               if distributed else None)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
        sampler=sampler, num_workers=4, pin_memory=True,
        drop_last=shuffle, collate_fn=partial(collate, pad_id=cfg.pad_id),
        persistent_workers=True,
    )


def make_dataset(name, split, tokenizer, cfg, limit=None):
    ex = load_examples(name, split, limit=limit)
    return MCQDataset(ex, tokenizer, max_len=cfg.max_seq_len)


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
                stage_name="", log_every=100):
    """One epoch. LR is set per optimizer step, not per epoch (the old
    per-epoch counter made warmup never complete — fixed; see run.md §8).

    Emits a progress line every ``log_every`` optimizer steps:
        [stage] ep0 step 1200/3809 (32%) loss=.. lr=.. gnorm=.. tok/s=.. eta=..
    ``loss`` is an EMA of the per-batch CE; ``gnorm`` is the pre-clip global
    grad norm (clip threshold 1.0 — a sustained value far above 1 means the
    clip is active every step; near 0 means training has stalled).
    """
    model.train()
    total, ntok, steps, nsamp = 0.0, 0, 0, 0
    gnorm_sum, gnorm_n = 0.0, 0
    loss_ema = None
    last_lr = opt.param_groups[0]["lr"]
    start_opt = opt_step
    t_start = time.time()
    t_window, ntok_window, next_at = t_start, 0, opt_step + log_every
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast():
            out = model(batch["input_ids"], labels=batch["labels"])
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
        nsamp += batch["input_ids"].size(0)
        ntok += bsz_tok
        ntok_window += bsz_tok
        steps += 1
    avg_gnorm = gnorm_sum / max(1, gnorm_n)
    return total / max(1, nsamp), ntok, opt_step, last_lr, avg_gnorm


@torch.no_grad()
def evaluate(model, loader, device, autocast) -> Dict[str, float]:
    model.eval()
    correct = seen = 0
    loss_sum = n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast():
            out = model(batch["input_ids"], labels=batch["labels"])
            logits = out.logits
        loss_sum += out.loss.item()
        n += 1
        # logits[t] predicts token t+1 -> score the answer at p via logits[p-1]
        got = logits[torch.arange(len(logits), device=device),
                     batch["answer_pos"] - 1]
        pred = got[:, LETTER_IDS].argmax(-1)
        correct += int((pred == batch["answer_target"]).sum())
        seen += len(pred)
    return {"loss": loss_sum / max(1, n), "accuracy": correct / max(1, seen),
            "n": seen}


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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=100,
                    help="print a progress line every N optimizer steps "
                         "(0 = epoch summaries only)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate held-out splits every N epochs, append "
                         "accuracy to history and rewrite results.json "
                         "incrementally, and keep best.pt on medmcqa "
                         "(0 = final evaluation only)")
    ap.add_argument("--resume", action="store_true",
                    help="continue <out>/last.pt (skips finished stage/epochs)")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training; evaluate <out>/last.pt on both test sets")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    distributed = setup_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() \
        else torch.device("cpu")
    is_main = (not distributed) or dist.get_rank() == 0

    resume_mode = args.resume or args.eval_only
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
        log("eval-only: skipping training, running held-out tests")
    results = {"variant": args.variant, "use_sandbox": cfg.use_sandbox,
               "params": estimate_model_size(cfg)["total"], "history": history,
               "test": {}}
    if not args.eval_only:
        for si, (stage_name, split) in enumerate(STAGES):
            first_epoch = 0
            while si * args.epochs + first_epoch < start_pair:
                first_epoch += 1
            if first_epoch >= args.epochs:
                if is_main:
                    log(f"stage {stage_name}: already complete, skipping")
                continue
            ds = make_dataset(stage_name, split, tokenizer, cfg, args.limit)
            loader = build_dataloader(ds, cfg, args.batch_size, True,
                                      distributed, seed=args.seed)
            steps_per_epoch = max(1, len(loader))
            opt_per_epoch = max(1, math.ceil(steps_per_epoch / args.grad_accum))
            total_opt = opt_per_epoch * args.epochs
            if is_main:
                log(f"\n=== stage {stage_name}  n={len(ds)} "
                    f"(epoch {first_epoch}+) | {opt_per_epoch} opt/epoch x "
                    f"{args.epochs} epochs = {total_opt} steps, "
                    f"lr {args.lr:g} cosine, warmup {args.warmup} ===")
            stage_opt_step = 0  # each stage gets its own cosine cycle
            for epoch in range(first_epoch, args.epochs):
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(epoch)
                t0 = time.time()
                avg, ntok, stage_opt_step, last_lr, avg_gnorm = train_stage(
                    model, loader, opt, device, epoch, scaler, autocast,
                    stage_opt_step, total_opt, args.lr, args.warmup,
                    args.grad_accum, stage_name=stage_name,
                    log_every=args.log_every)
                opt_step = stage_opt_step  # for resume/ckpt bookkeeping
                step += 1
                completed = si * args.epochs + epoch + 1
                if is_main:
                    log(f"[{stage_name}] ep{epoch} done loss={avg:.4f} "
                        f"gnorm={avg_gnorm:.2f} lr_end={last_lr:.2e} "
                        f"tok/s={ntok/max(1,time.time()-t0):.0f} "
                        f"time={fmt_duration(time.time()-t0)}")
                    history.append({"stage": stage_name, "epoch": epoch,
                                    "loss": avg, "lr": last_lr,
                                    "gnorm": avg_gnorm})
                    torch.save({"model": raw.state_dict(),
                                "cfg": cfg.to_dict(),
                                "step": step, "completed": completed,
                                "opt_step": opt_step, "history": history,
                                "best_medmcqa": best_acc},
                               os.path.join(args.out, "last.pt"))
                    if args.eval_every > 0 and (
                            (epoch + 1) % args.eval_every == 0
                            or epoch + 1 == args.epochs):
                        tests_now = {}
                        for tname, tsplit in TESTS:
                            tds = make_dataset(tname, tsplit, tokenizer, cfg,
                                               args.limit)
                            tloader = build_dataloader(tds, cfg,
                                                       args.batch_size, False,
                                                       False, seed=args.seed)
                            tests_now[tname] = evaluate(raw, tloader, device,
                                                        autocast)
                            log(f"[{stage_name}] ep{epoch} eval {tname} "
                                f"acc={tests_now[tname]['accuracy']*100:.2f}% "
                                f"loss={tests_now[tname]['loss']:.4f} "
                                f"n={tests_now[tname]['n']}")
                        history[-1]["test"] = tests_now
                        results["test"] = tests_now
                        cur = tests_now["medmcqa"]["accuracy"]
                        if cur > best_acc:
                            best_acc = cur
                            torch.save({"model": raw.state_dict(),
                                        "cfg": cfg.to_dict(),
                                        "step": step, "completed": completed,
                                        "opt_step": opt_step,
                                        "history": history,
                                        "best_medmcqa": best_acc},
                                       os.path.join(args.out, "best.pt"))
                            log(f"[{stage_name}] ep{epoch} new best medmcqa "
                                f"{cur*100:.2f}% -> best.pt")
                        with open(os.path.join(args.out, "results.json"),
                                  "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2)

    for name, split in TESTS:
        ds = make_dataset(name, split, tokenizer, cfg, args.limit)
        loader = build_dataloader(ds, cfg, args.batch_size, False,
                                  distributed, seed=args.seed)
        r = evaluate(model, loader, device, autocast)
        results["test"][name] = r
        if is_main:
            log(f"[test {name}] acc={r['accuracy']*100:.2f}% "
                f"loss={r['loss']:.4f} n={r['n']}")

    if is_main:
        with open(os.path.join(args.out, "results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log(f"\nwrote {args.out}/results.json")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
