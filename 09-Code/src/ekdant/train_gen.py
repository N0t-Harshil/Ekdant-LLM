"""v2 training: pretrained backbone + add-on on generative medical QA
(spec 07-Management/30).  Started through ``python -m ekdant.train --task gen``.

    zero-shot   perplexity of the untrained arm on every evaluation set
    phase 1     PubMedQA (question + abstract -> conclusion), validation on
                2,000 held-out rows every --eval-fraction of the epoch
    phase 2     MedQuAD fine-tune half; validation slice each epoch -> best.pt
    final       better of best.pt / last.pt on the validation slice, then the
                MedQuAD benchmark half (scored once) + PubMedQA held-out

Arms: baseline (the backbone alone) and sandbox (backbone + L2).  Both start
identical (zero-init gate) and see identical data in identical order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import torch

from ekdant.backbone import apply_hf_token, load_backbone, load_tokenizer
from ekdant.gen_data import (
    GenDataset,
    gen_collate,
    holdout,
    load_medquad,
    load_pubmedqa,
    question_key,
    split_by_question,
)
from ekdant.gen_eval import answer_nll, by_type, generate_greedy, score_generations
from ekdant.model import EkdantLM, V2Config, count_parameters
from ekdant.train import fmt_duration, log, lr_at

STAGES = ("pubmedqa", "medquad_ft")
ADDON = {"baseline": "none", "sandbox": "sandbox"}
ROW_KEYS = ("row_nll", "row_tokens", "row_idx")


def pick_device(gpu: str, min_free_gb: float) -> torch.device:
    """The requested GPU (``auto`` = most free memory), refusing to start
    below ``min_free_gb`` free - the pod's GPUs are shared and GPU 0 is often
    nearly full.  CPU only when asked for (``--gpu cpu``): a broken CUDA must
    not turn a GPU run into a silent multi-day CPU run."""
    if gpu == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        log("error: CUDA is not available - check the GPU/driver, or pass "
            "--gpu cpu for a CPU run (tests and smoke runs only)")
        raise SystemExit(1)
    if gpu == "auto":
        free = {i: torch.cuda.mem_get_info(i)[0]
                for i in range(torch.cuda.device_count())}
        idx = max(free, key=free.get)
    else:
        idx = int(gpu)
    f, t = torch.cuda.mem_get_info(idx)
    log(f"GPU {idx}: {f/1e9:.1f} GB free of {t/1e9:.1f} GB")
    if f / 1e9 < min_free_gb:
        log(f"error: GPU {idx} has only {f/1e9:.1f} GB free (< --min-free-gb "
            f"{min_free_gb:g}). Other jobs are using it - pick a free GPU "
            f"with CUDA_VISIBLE_DEVICES=<n> (check nvidia-smi) or --gpu auto")
        raise SystemExit(1)
    torch.cuda.set_device(idx)
    return torch.device("cuda", idx)


# Flags that fix the data order, the schedule or the arm: a resume with any
# of them changed would skip the wrong batches or mislabel the run.
RUN_KEYS = ("variant", "batch_size", "grad_accum", "lr", "addon_lr", "gate_lr",
            "warmup",
            "epochs", "ft_epochs", "eval_fraction", "pubmed_holdout", "limit",
            "seed")


def phase1_eval_points(nb: int, eval_fraction: float, grad_accum: int) -> set:
    """Batch counts (on optimiser-step boundaries, before the epoch end) at
    which phase 1 validates and checkpoints mid-epoch."""
    k = int(round(1 / eval_fraction))
    pts = {math.ceil(i * nb / k / grad_accum) * grad_accum for i in range(1, k)}
    return {p for p in pts if 0 < p < nb}


def clip_grads(model: EkdantLM, max_norm: float):
    """Clip the backbone and the add-on separately, so the add-on's gradient
    can never shrink the backbone's update (the baseline has no add-on)."""
    addon = list(model.addon.parameters()) if model.addon is not None else []
    ids = {id(p) for p in addon}
    backbone = [p for p in model.parameters() if id(p) not in ids]
    bn = float(torch.nn.utils.clip_grad_norm_(backbone, max_norm))
    an = float(torch.nn.utils.clip_grad_norm_(addon, max_norm)) if addon else 0.0
    return bn, an


def save_checkpoint(state: Dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ekdant.train --task gen")
    ap.add_argument("--variant", choices=sorted(ADDON), required=True)
    ap.add_argument("--config", default=None,
                    help="default: configs/v2_<variant>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone LR")
    ap.add_argument("--addon-lr", type=float, default=1e-4,
                    help="add-on + gate LR (1e-3 kept inflating L2's weights, "
                         "T34)")
    ap.add_argument("--gate-lr", type=float, default=1e-3,
                    help="LR of L2's 576-entry gate alone (T35); its "
                         "matrices keep --addon-lr")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1, help="phase-1 epochs")
    ap.add_argument("--ft-epochs", type=int, default=3, help="phase-2 epochs")
    ap.add_argument("--eval-fraction", type=float, default=0.25,
                    help="phase-1 validation every this fraction of an epoch")
    ap.add_argument("--pubmed-holdout", type=int, default=2000)
    ap.add_argument("--gen-subset", type=int, default=1000,
                    help="MedQuAD benchmark rows scored by generation")
    ap.add_argument("--gen-subset-pubmed", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-new-pubmed", type=int, default=128)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="rows per dataset (smoke runs only, never results)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--gpu", default="auto",
                    help="auto (most free memory), an index, or cpu")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--max-bad-steps", type=int, default=20,
                    help="stop after this many consecutive optimiser steps "
                         "skipped for a non-finite loss or gradient")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    return ap


def _batches(ds: GenDataset, bs: int, order: Optional[List[int]] = None):
    order = list(range(len(ds))) if order is None else order
    return [order[i:i + bs] for i in range(0, len(order), bs)]


def _strip_rows(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if k not in ROW_KEYS}


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    last_path = os.path.join(args.out, "last.pt")
    best_path = os.path.join(args.out, "best.pt")
    res_path = os.path.join(args.out, "results.json")
    resume = args.resume or args.eval_only
    if not resume and os.path.exists(last_path):
        log(f"error: {last_path} exists but --resume was not passed - pass "
            f"--resume, or use a new --out directory")
        raise SystemExit(1)
    if resume and not os.path.exists(last_path):
        log(f"error: {last_path} not found - nothing to resume/evaluate")
        raise SystemExit(1)

    device = pick_device(args.gpu, args.min_free_gb)
    ckpt = torch.load(last_path, map_location="cpu", weights_only=False) \
        if resume else None
    if ckpt is not None:
        saved = ckpt.get("run_args", {})
        keys = (("variant", "limit", "seed", "pubmed_holdout")
                if args.eval_only else RUN_KEYS)
        diff = [f"--{k.replace('_', '-')} {saved[k]} (now {getattr(args, k)})"
                for k in keys if k in saved and saved[k] != getattr(args, k)]
        if diff:
            log("error: this run was started with different settings - resume "
                "with the original ones: " + ", ".join(diff))
            raise SystemExit(1)
        cfg = V2Config.from_dict(ckpt["cfg"])
    else:
        cfg_path = args.config or os.path.join("configs",
                                               f"v2_{args.variant}.json")
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
        if "addon" in raw and raw["addon"] != ADDON[args.variant]:
            log(f"error: {cfg_path} has addon={raw['addon']!r} but --variant "
                f"{args.variant} needs addon={ADDON[args.variant]!r}")
            raise SystemExit(1)
        cfg = V2Config.from_dict({**V2Config().to_dict(), **raw})
        cfg.addon = ADDON[args.variant]

    log(f"hugging face token: {apply_hf_token(cfg.hf_token_file)}")
    torch.manual_seed(args.seed)
    tok = load_tokenizer(cfg.backbone)
    model = EkdantLM(cfg, load_backbone(cfg.backbone)).to(device)
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.addon_lr,
                                               gate_lr=args.gate_lr),
                            betas=(0.9, 0.95))
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) \
        if use_bf16 else (lambda: torch.autocast(device.type, enabled=False))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log(f"variant={args.variant} addon={cfg.addon} backbone={cfg.backbone} "
        f"params total={count_parameters(model)/1e6:.2f}M "
        f"addon={model.addon_params()/1e6:.2f}M device={device}")

    # ---- data (identical for every arm: same seed, same splits) ----
    pub = load_pubmedqa(args.limit, args.seed)
    pub_train, pub_held = holdout(pub, min(args.pubmed_holdout, len(pub) // 10),
                                  args.seed)
    ft, val, bench = split_by_question(load_medquad(args.limit, args.seed),
                                       args.seed)
    mk = lambda ex: GenDataset(ex, tok, cfg.max_seq_len, cfg.max_answer)  # noqa: E731
    data = {"pubmedqa": mk(pub_train), "pubmedqa_held": mk(pub_held),
            "medquad_ft": mk(ft), "medquad_val": mk(val),
            "medquad_bench": mk(bench)}
    log("rows " + " ".join(f"{k}={len(v)}(-{v.dropped})" for k, v in data.items()))

    def collate(rows):
        return gen_collate(rows, pad)

    def score(name: str) -> Dict:
        ds = data[name]
        loader = (collate([ds[i] for i in b]) for b in _batches(ds, args.batch_size))
        return answer_nll(model, loader, device, autocast)

    def gen_scores(name: str, n: int, max_new: int) -> Dict:
        ds = data[name]
        rows = sorted(random.Random(args.seed).sample(range(len(ds)),
                                                      min(n, len(ds))))
        outs = generate_greedy(model, [ds.prompt_ids(i) for i in rows], max_new,
                               device, tok.eos_token_id, pad, args.gen_batch,
                               autocast)
        preds = [tok.decode(o, skip_special_tokens=True).strip() for o in outs]
        g = score_generations(preds, [ds.example(i).answer for i in rows])
        g["gen_n"] = g.pop("n")  # keep "n" = rows scored for perplexity
        return g

    results: Dict = {}
    if resume and os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            results = json.load(f)
    results.update({"variant": args.variant, "addon": cfg.addon,
                    "config": cfg.to_dict(), "params_total": count_parameters(model),
                    "params_addon": model.addon_params()})

    def write() -> None:
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

    if ckpt is None:  # loaded weights are trained: never label them zero-shot
        results["zero_shot"] = {k: _strip_rows(score(k)) for k in
                                ("pubmedqa_held", "medquad_val", "medquad_bench")}
        log("zero-shot ppl " + " ".join(
            f"{k}={v['ppl']:.3f}" for k, v in results["zero_shot"].items()))
        write()
    elif "zero_shot" not in results:
        log("note: no zero-shot record in results.json (it is only measured "
            "at the start of a fresh run)")

    st = (ckpt or {})
    st_stage, st_epoch, st_done = st.get("stage", 0), st.get("epoch", 0), \
        st.get("batches_done", 0)
    history: List[Dict] = list(st.get("history", []))
    best_val = float(st.get("best_val_nll", float("inf")))
    tokens_seen = int(st.get("tokens_seen", 0))
    opt_step = int(st.get("opt_step", 0))
    wall_prev = float(st.get("wall_clock_s", 0.0))
    skipped_steps = int(st.get("skipped_steps", 0))
    bad_streak = 0
    t_start = time.time()

    def checkpoint(stage: int, epoch: int, done: int) -> None:
        save_checkpoint({"model": model.state_dict(), "optimizer": opt.state_dict(),
                         "cfg": cfg.to_dict(), "stage": stage, "epoch": epoch,
                         "batches_done": done, "opt_step": opt_step,
                         "history": history, "best_val_nll": best_val,
                         "tokens_seen": tokens_seen,
                         "skipped_steps": skipped_steps,
                         "wall_clock_s": wall_prev + time.time() - t_start,
                         "run_args": {k: getattr(args, k) for k in RUN_KEYS}},
                        last_path)

    for si, stage in enumerate(STAGES if not args.eval_only else ()):
        epochs = args.epochs if si == 0 else args.ft_epochs
        if si < st_stage:
            continue
        if si > st_stage:
            st_epoch, st_done, opt_step = 0, 0, 0
        ds = data[stage]
        nb = math.ceil(len(ds) / args.batch_size)
        total_opt = math.ceil(nb / args.grad_accum) * epochs
        eval_at = (phase1_eval_points(nb, args.eval_fraction, args.grad_accum)
                   if si == 0 else set())
        log(f"\n=== stage {stage} n={len(ds)} | {nb} batches x {epochs} epochs, "
            f"{total_opt} opt steps ===")
        for epoch in range(st_epoch, epochs):
            g = torch.Generator().manual_seed(args.seed * 1000 + si * 100 + epoch)
            batches = _batches(ds, args.batch_size,
                               torch.randperm(len(ds), generator=g).tolist())
            skip = st_done if (si, epoch) == (st_stage, st_epoch) else 0
            model.train()
            opt.zero_grad(set_to_none=True)
            t0, loss_ema, group_bad, l2_rms = time.time(), None, False, None
            for bi in range(skip, nb):
                b = collate([ds[i] for i in batches[bi]])
                ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
                with autocast():
                    out = model(ids, attention_mask=mask,
                                labels=b["labels"].to(device))
                loss = out.loss
                if "consensus_rms" in out.aux:
                    l2_rms = float(out.aux["consensus_rms"])
                tokens_seen += int(mask.sum())
                lv = loss.item()
                # T32 guard: a non-finite loss or gradient never reaches the
                # weights; the whole accumulation group is dropped.
                if math.isfinite(lv):
                    (loss / args.grad_accum).backward()
                    loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
                else:
                    group_bad = True
                if (bi + 1) % args.grad_accum == 0 or bi + 1 == nb:
                    gn_backbone, gn_addon = clip_grads(model, 1.0)
                    if group_bad or not (math.isfinite(gn_backbone)
                                         and math.isfinite(gn_addon)):
                        skipped_steps += 1
                        bad_streak += 1
                        log(f"[{stage}] warning: non-finite loss/gradient at "
                            f"step {opt_step + 1} - update skipped "
                            f"({bad_streak} in a row, {skipped_steps} total)")
                        if bad_streak >= args.max_bad_steps:
                            log(f"error: {bad_streak} consecutive non-finite "
                                f"steps - training has diverged; stopping "
                                f"(last.pt holds the last good checkpoint)")
                            raise SystemExit(1)
                    else:
                        f = lr_at(opt_step, total_opt, 1.0, args.warmup)
                        for grp in opt.param_groups:
                            grp["lr"] = grp["base_lr"] * f
                        opt.step()
                        bad_streak = 0
                    opt.zero_grad(set_to_none=True)
                    group_bad = False
                    opt_step += 1
                    if (args.log_every and opt_step % args.log_every == 0
                            and loss_ema is not None):
                        el = time.time() - t0
                        rate = (bi + 1 - skip) / max(el, 1e-9)
                        log(f"[{stage}] ep{epoch} step {opt_step}/{total_opt} "
                            f"loss={loss_ema:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                            f"gnorm={gn_backbone:.2f}/{gn_addon:.2f} "
                            f"|g|={model.addon.gate_magnitude() if model.addon else 0:.4f} "
                            + (f"l2={l2_rms:.3g} " if l2_rms is not None else "") +
                            f"eta={fmt_duration((nb - bi - 1) / max(rate, 1e-9))}")
                if si == 0 and bi + 1 in eval_at:
                    r = score("pubmedqa_held")
                    history.append({"stage": stage, "epoch": epoch,
                                    "at": (bi + 1) / nb, "val": _strip_rows(r)})
                    log(f"[{stage}] ep{epoch} {100*(bi+1)/nb:.0f}% "
                        f"val pubmedqa_held ppl={r['ppl']:.3f}")
                    checkpoint(si, epoch, bi + 1)
                    model.train()
            val_name = "pubmedqa_held" if si == 0 else "medquad_val"
            r = score(val_name)
            entry = {"stage": stage, "epoch": epoch, "at": 1.0,
                     "train_loss": loss_ema, "val": _strip_rows(r),
                     "l2_rms": l2_rms,
                     "gate_magnitude": model.addon.gate_magnitude()
                     if model.addon else 0.0,
                     "gate_per_site": model.addon.site_gate_magnitudes()
                     if model.addon else []}
            history.append(entry)
            log(f"[{stage}] ep{epoch} done loss={loss_ema or float('nan'):.4f} val {val_name} "
                f"ppl={r['ppl']:.3f} time={fmt_duration(time.time()-t0)}")
            if si == 1 and r["nll"] < best_val:
                best_val = r["nll"]
                save_checkpoint({"model": model.state_dict(), "epoch": epoch,
                                 "val_nll": best_val}, best_path)
                log(f"[{stage}] ep{epoch} new best medquad_val -> best.pt")
            checkpoint(si, epoch + 1, 0)
        st_epoch, st_done = 0, 0
        if si == 0 and "phase1" not in results:
            results["phase1"] = {k: _strip_rows(score(k))
                                 for k in ("pubmedqa_held", "medquad_val")}
            results["history"] = history
            write()

    # ---- final protocol: better of best/last on the validation slice ----
    scores = {"last.pt": score("medquad_val")["nll"]}
    chosen = "last.pt"
    if os.path.exists(best_path):
        last_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(torch.load(best_path, map_location="cpu",
                                         weights_only=False)["model"])
        scores["best.pt"] = score("medquad_val")["nll"]
        if scores["best.pt"] <= scores["last.pt"]:
            chosen = "best.pt"
        else:
            model.load_state_dict(last_state)
    results["selected"] = {"checkpoint": chosen, "selected_on": "medquad_val",
                           "val_nll": scores}
    log(f"final benchmark uses {chosen} (medquad_val nll {scores[chosen]:.4f})")

    b = score("medquad_bench")
    exb = data["medquad_bench"].examples
    b["by_type"] = by_type(b, exb)
    b["row_group"] = [question_key(exb[i].question) for i in b["row_idx"]]
    b.update(gen_scores("medquad_bench", args.gen_subset, args.max_new))
    p = _strip_rows(score("pubmedqa_held"))
    p.update(gen_scores("pubmedqa_held", args.gen_subset_pubmed,
                        args.max_new_pubmed))
    wall = wall_prev + time.time() - t_start
    results["history"] = history
    results["test"] = {"medquad_bench": b, "pubmedqa_held": p,
                       "gate_magnitude": model.addon.gate_magnitude()
                       if model.addon else 0.0,
                       "gate_per_site": model.addon.site_gate_magnitudes()
                       if model.addon else []}
    results["cost"] = {"params_total": count_parameters(model),
                       "params_addon": model.addon_params(),
                       "tokens_seen": tokens_seen, "wall_clock_s": wall,
                       "skipped_steps": skipped_steps,
                       "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9
                       if device.type == "cuda" else 0.0}
    log(f"[test medquad_bench] ppl={b['ppl']:.3f} rouge_l={b['rouge_l']:.4f} "
        f"token_f1={b['token_f1']:.4f} n={b['n']} generated={b['gen_n']}")
    log(f"[test pubmedqa_held] ppl={p['ppl']:.3f} rouge_l={p['rouge_l']:.4f} "
        f"token_f1={p['token_f1']:.4f}")
    write()
    log(f"\nwrote {res_path}")
