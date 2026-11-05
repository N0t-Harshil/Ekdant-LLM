"""v2 evaluation (spec 07-Management/30 §5).

    answer_nll        reference-answer NLL / perplexity (primary metric)
    by_type           perplexity per MedQuAD question type
    generate_greedy   batched greedy decoding with the KV cache
    rouge_l, token_f1 text overlap of generated vs reference answers
    paired_bootstrap  CI of the per-answer NLL difference between two arms,
                      resampling questions (rows sharing a question move
                      together)

CLI:  python -m ekdant.gen_eval compare runs/v2_baseline runs/v2_sandbox ...
      python -m ekdant.gen_eval mcq runs/v25_baseline runs/v25_end ... (T46,
      multiple-choice accuracy; see ``MCQ_DOC``)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ekdant.backbone import _offline_first, apply_hf_token, load_backbone, load_tokenizer
from ekdant.data import LETTERS, load_examples
from ekdant.model import EkdantLM, V2Config

DIAG_KEYS = ("anchor_energy", "trajectory_variance", "spread_ratio",
             # L2 v3 (T50/T51); absent for older L2s
             "confidence", "learned_confidence", "vote_support",
             "verifier_top_weight", "token_gate")


def auroc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """Area under the ROC curve: the chance that a random positive scores
    above a random negative (ties count half); 0.5 = no signal."""
    scores, positive = scores.double().flatten(), positive.bool().flatten()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    end = counts.cumsum(0).double()
    r = ((end - counts.double() + 1.0 + end) / 2.0)[inv]   # average rank of ties
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _conf_logits(model) -> Optional[torch.Tensor]:
    """The learned confidence logits [B,T] of the last forward (mean over
    the L2 sites), or None for a model without them (T53)."""
    sites = model._all_sites() if hasattr(model, "_all_sites") else []
    zs = [s.last_conf_logit for s in sites if getattr(s, "last_conf_logit", None) is not None]
    return torch.stack([z.float() for z in zs]).mean(0) if zs else None


def calibration(conf: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    """Does the learned confidence know when the model is right (T53)?
    AUROC (0.5 = no idea, 1 = perfect), mean confidence on right vs wrong
    tokens, and the Brier score."""
    conf, right = conf.double(), right.double()
    return {"auroc": auroc(conf, right.bool()),
            "conf_right": float(conf[right.bool()].mean()) if right.any() else float("nan"),
            "conf_wrong": float(conf[~right.bool()].mean()) if (~right.bool()).any() else float("nan"),
            "accuracy": float(right.mean()), "brier": float((conf - right).pow(2).mean()),
            "tokens": int(right.numel())}


@torch.no_grad()
def answer_nll(model, loader, device, autocast) -> Dict:
    """Token-weighted mean NLL over answer tokens (labels != -100), plus
    per-row sums for the per-type breakdown and the bootstrap."""
    model.eval()
    row_nll: List[float] = []
    row_tok: List[int] = []
    row_idx: List[int] = []
    diag: Dict[str, float] = {}
    confs: List[torch.Tensor] = []
    rights: List[torch.Tensor] = []
    own: List[torch.Tensor] = []
    own_right: List[torch.Tensor] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            out = model(ids, attention_mask=mask, labels=labels)
        logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
        tgt = labels[:, 1:]
        valid = tgt != -100
        top_lp, top = logp.max(-1)
        right = (top == tgt)[valid].cpu()
        own.append(top_lp.exp()[valid].cpu())
        own_right.append(right)
        z = _conf_logits(model)
        if z is not None:
            confs.append(torch.sigmoid(z[:, :-1].float())[valid].cpu())
            rights.append(right)
        tok_nll = -logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tok_nll = tok_nll * valid
        row_nll += tok_nll.sum(-1).tolist()
        row_tok += valid.sum(-1).tolist()
        row_idx += batch["idx"].tolist()
        for k in DIAG_KEYS:
            v = out.aux.get(k)
            if v is not None:
                diag[k] = diag.get(k, 0.0) + float(v) * ids.size(0)
    tokens = sum(row_tok)
    nll = sum(row_nll) / max(1, tokens)
    r = {"nll": nll, "ppl": math.exp(nll), "tokens": tokens, "n": len(row_idx),
         "row_nll": row_nll, "row_tokens": row_tok, "row_idx": row_idx}
    if diag:
        r["addon"] = {k: v / max(1, len(row_idx)) for k, v in diag.items()}
    if confs:
        r["confidence"] = calibration(torch.cat(confs), torch.cat(rights))
    if own:
        # the yardstick (T54): the model's own top-word probability; L2's
        # confidence only adds something if it beats this
        r["own_confidence"] = calibration(torch.cat(own), torch.cat(own_right))
    return r


def by_type(result: Dict, examples: Sequence) -> Dict[str, float]:
    """Perplexity per ``qtype`` (``examples`` indexed by ``row_idx``)."""
    acc: Dict[str, List[float]] = {}
    for i, nll, tok in zip(result["row_idx"], result["row_nll"],
                           result["row_tokens"]):
        a = acc.setdefault(examples[i].qtype or "-", [0.0, 0])
        a[0] += nll
        a[1] += tok
    return {k: math.exp(s / max(1, n)) for k, (s, n) in sorted(acc.items())}


_PUNCT = str.maketrans("", "", string.punctuation)


def normalise_text(s: str) -> List[str]:
    return s.lower().translate(_PUNCT).split()


def _lcs(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _f(overlap: int, n_pred: int, n_ref: int) -> float:
    if n_pred == 0 and n_ref == 0:
        return 1.0
    if overlap == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_l(pred: str, ref: str) -> float:
    a, b = normalise_text(pred), normalise_text(ref)
    return _f(_lcs(a, b), len(a), len(b))


def token_f1(pred: str, ref: str) -> float:
    a, b = normalise_text(pred), normalise_text(ref)
    common = 0
    counts: Dict[str, int] = {}
    for w in b:
        counts[w] = counts.get(w, 0) + 1
    for w in a:
        if counts.get(w, 0) > 0:
            counts[w] -= 1
            common += 1
    return _f(common, len(a), len(b))


def score_generations(preds: List[str], refs: List[str]) -> Dict[str, float]:
    n = max(1, len(preds))
    return {"rouge_l": sum(map(rouge_l, preds, refs)) / n,
            "token_f1": sum(map(token_f1, preds, refs)) / n, "n": len(preds)}


@torch.no_grad()
def generate_greedy(model, prompts: List[List[int]], max_new: int, device,
                    eos_id: int, pad_id: int, batch_size: int = 16,
                    autocast=None) -> List[List[int]]:
    """Greedy continuation of each prompt (token ids), EOS excluded.

    Batches are left-padded; positions count real tokens only
    (``cumsum(mask) - 1``), so a row's output does not depend on its batch.
    """
    model.eval()
    autocast = autocast or (lambda: torch.autocast("cpu", enabled=False))
    out: List[List[int]] = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, p in enumerate(chunk):
            ids[r, L - len(p):] = torch.tensor(p)
            mask[r, L - len(p):] = 1
        ids, mask = ids.to(device), mask.to(device)
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        gen = [[] for _ in chunk]
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        with autocast():
            o = model(ids, attention_mask=mask, position_ids=pos, use_cache=True)
        past = o.aux["past_key_values"]
        nxt = o.logits[:, -1].argmax(-1)
        last_pos = pos[:, -1]
        for _ in range(max_new):
            done |= nxt == eos_id
            for r in range(len(chunk)):
                if not done[r]:
                    gen[r].append(int(nxt[r]))
            if bool(done.all()):
                break
            step = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            last_pos = last_pos + 1
            with autocast():
                o = model(step.unsqueeze(1), attention_mask=mask,
                          position_ids=last_pos.unsqueeze(1),
                          past_key_values=past, use_cache=True)
            past = o.aux["past_key_values"]
            nxt = o.logits[:, -1].argmax(-1)
        out += gen
    return out


def paired_bootstrap(a: List[float], b: List[float], groups: List[str],
                     n: int = 1000, seed: int = 17,
                     weights: Optional[List[float]] = None) -> Dict[str, float]:
    """sum(b - a) / sum(weights) with a 95 % percentile CI, resampling whole
    groups (questions) with replacement.  Without weights this is the mean
    per-row difference; with per-row NLL sums and token counts it is the
    per-token NLL difference - the weighting of the headline perplexity."""
    w = weights if weights is not None else [1.0] * len(a)
    per: Dict[str, List[float]] = {}
    for x, y, g, k in zip(a, b, groups, w):
        s = per.setdefault(g, [0.0, 0.0])
        s[0] += y - x
        s[1] += k
    sums = list(per.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        pick = [sums[rng.randrange(len(sums))] for _ in sums]
        stats.append(sum(p[0] for p in pick) / sum(p[1] for p in pick))
    stats.sort()
    return {"diff": sum(y - x for x, y in zip(a, b)) / max(1e-12, sum(w)),
            "lo": stats[int(0.025 * (n - 1))], "hi": stats[int(0.975 * (n - 1))]}


def _rows(t: Dict) -> Dict[int, tuple]:
    return {i: (s, k) for i, s, k in
            zip(t["row_idx"], t["row_nll"], t["row_tokens"])}


def compare(run_dirs: List[str]) -> None:
    runs = []
    for d in run_dirs:
        with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
            runs.append((d, json.load(f)))
    print(f"{'run':28s} {'bench ppl':>10s} {'ROUGE-L':>8s} {'F1':>7s} "
          f"{'|g|':>8s} {'pubmed ppl':>11s} {'L2 off ppl':>11s}")
    for d, r in runs:
        b, p = r["test"]["medquad_bench"], r["test"]["pubmedqa_held"]
        off = r["test"].get("l2_off")
        print(f"{os.path.basename(d.rstrip('/')):28s} {b['ppl']:10.3f} "
              f"{b.get('rouge_l', float('nan')):8.4f} "
              f"{b.get('token_f1', float('nan')):7.4f} "
              f"{r['test'].get('gate_magnitude', 0.0):8.4f} {p['ppl']:11.3f} "
              + (f"{off['medquad_bench']['ppl']:11.3f}" if off else f"{'-':>11s}"))
    ref_d, ref = runs[0]
    ra = _rows(ref["test"]["medquad_bench"])
    groups = dict(zip(ref["test"]["medquad_bench"]["row_idx"],
                      ref["test"]["medquad_bench"]["row_group"]))
    for d, r in runs[1:]:
        rb = _rows(r["test"]["medquad_bench"])
        keys = sorted(set(ra) & set(rb))
        ci = paired_bootstrap([ra[k][0] for k in keys], [rb[k][0] for k in keys],
                              [groups[k] for k in keys],
                              weights=[ra[k][1] for k in keys])
        verdict = "real difference" if ci["lo"] > 0 or ci["hi"] < 0 else "tie"
        print(f"{os.path.basename(d.rstrip('/'))} - "
              f"{os.path.basename(ref_d.rstrip('/'))}: per-token NLL "
              f"{ci['diff']:+.4f}  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
              f"({verdict}; negative = better)")
    # T44: how much each trained model depends on its L2
    for d, r in runs:
        off = r["test"].get("l2_off")
        if off:
            imp = off["importance"]
            verdict = ("the model depends on L2" if imp["lo"] > 0 else
                       "the model is better without L2" if imp["hi"] < 0 else
                       "no measurable dependence on L2")
            print(f"{os.path.basename(d.rstrip('/'))}: L2 off -> per-token NLL "
                  f"{imp['diff']:+.4f}  95% CI [{imp['lo']:+.4f}, {imp['hi']:+.4f}]  "
                  f"({verdict}; positive = L2 matters)")


# ---------------------------------------------------------------------------
# Multiple-choice accuracy (T46)
# ---------------------------------------------------------------------------

MCQ_DOC = """Multiple-choice accuracy of finished v2 runs (T46).

The v2 models write free-text answers, so their benchmark has no
"% correct".  This section asks every finished run the same multiple-choice
questions and reports the share it gets right:

    medmcqa   MedMCQA validation (4 options; the test split has no labels)
    medqa     MedQA-USMLE test   (4 options)
    pubmedqa  PubMedQA pqa_labeled, 1,000 expert-labelled questions with
              their abstract (yes / no / maybe)

Main method - option likelihood (as in lm-evaluation-harness): the prompt
is the training format ``Question: ...\\nAnswer:`` (plus ``Context:`` for
PubMedQA) and the model's choice is the option text with the highest mean
log-probability per token.  Second method - letter: the options are listed
as A-D in the prompt and the model's choice is the letter with the highest
next-token probability (the models were never trained on this format, so
expect it near chance).  For L2 runs both are also scored with L2 switched
off (the L2-off test, T44).  Nothing is trained; results.json is never
changed - each run gets its own ``mcq.json``.

CLI:
    python -m ekdant.gen_eval mcq runs/v25_baseline runs/v25_end ... [--zero-shot]
    python -m ekdant.gen_eval mcq --compare runs/v25_baseline runs/v25_end ...
"""

MCQ_DATASETS = ("medmcqa", "medqa", "pubmedqa")
# mcq.json format: 1 = % right (T46), 2 = + thinking time (T47), 3 = + MedQuAD
# perplexity on / off / thinking, facts (T48), 4 = forced runs scored with
# their trained gate floor (T49), 5 = + the confidence test (T54), 6 = + the
# explanation benchmark (T56).  An older file is scored again (a version-3
# file only for a forced run).
MCQ_VERSION = 6
MCQ_CHANCE = {"medmcqa": 0.25, "medqa": 0.25, "pubmedqa": 1 / 3}
PUBMED_OPTIONS = ["yes", "no", "maybe"]


class Question:
    def __init__(self, question: str, options: List[str], answer: int,
                 context: str = ""):
        self.question, self.options = question, options
        self.answer, self.context = answer, context


def load_questions(name: str, limit: Optional[int] = None,
                   seed: int = 17) -> List[Question]:
    if name in ("medmcqa", "medqa"):
        split = "validation" if name == "medmcqa" else "test"
        out = [Question(e.question, list(e.options), LETTERS.index(e.answer))
               for e in load_examples(name, split)]
    elif name == "pubmedqa":
        from datasets import load_dataset

        ds = _offline_first(lambda: load_dataset("qiaojin/PubMedQA",
                                                 "pqa_labeled", split="train"))
        out = [Question(r["question"], PUBMED_OPTIONS,
                        PUBMED_OPTIONS.index(r["final_decision"]),
                        " ".join(r["context"]["contexts"]))
               for r in ds if r["final_decision"] in PUBMED_OPTIONS]
    else:
        raise ValueError(f"unknown dataset {name!r}")
    if limit is not None and limit < len(out):
        out = random.Random(seed).sample(out, limit)
    return out


def _enc(tok, s: str) -> List[int]:
    return tok(s, add_special_tokens=False)["input_ids"]


def _mcq_prompt(tok, q: Question, max_prompt: int, letters: bool) -> Optional[List[int]]:
    """Training-format prompt; a long context is cut to fit, a question that
    still does not fit is skipped (None)."""
    head = f"Question: {' '.join(q.question.split())}"
    if letters:
        head += "\n" + "\n".join(f"{L}. {' '.join(o.split())}"
                                 for L, o in zip(LETTERS, q.options))
    suf = _enc(tok, "\nAnswer:")
    if q.context:
        pre = _enc(tok, head + "\nContext:")
        ctx = _enc(tok, " " + q.context)
        keep = max_prompt - len(pre) - len(suf)
        return None if keep < 0 else pre + ctx[:keep] + suf
    ids = _enc(tok, head) + suf
    return ids if len(ids) <= max_prompt else None


@torch.no_grad()
def _forward_rows(model, rows: List[List[int]], pad: int, device, autocast,
                  batch_size: int):
    """Yields (row index, logits [len, V] on the device, L2's confidence
    logits [len] or None) for every row, batched by length so little compute
    is spent on padding."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        L = max(len(rows[i]) for i in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, i in enumerate(chunk):
            ids[r, :len(rows[i])] = torch.tensor(rows[i])
            mask[r, :len(rows[i])] = 1
        with autocast():
            out = model(ids.to(device), attention_mask=mask.to(device))
        z = _conf_logits(model)
        for r, i in enumerate(chunk):
            yield i, out.logits[r, :len(rows[i])], \
                (None if z is None else z[r, :len(rows[i])])


def score_questions(model, tok, questions: Sequence[Question], device, autocast,
                    max_len: int = 1024, batch_size: int = 16) -> Dict:
    """Per question: the option chosen by option likelihood and by letter,
    and whether each is right.  Questions that do not fit are skipped."""
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    letter_ids = [_enc(tok, " " + L)[-1] for L in LETTERS]
    opt_rows, opt_meta, let_rows, let_q = [], [], [], []
    kept: List[int] = []
    for qi, q in enumerate(questions):
        opts = [_enc(tok, " " + " ".join(o.split())) for o in q.options]
        p = _mcq_prompt(tok, q, max_len - max(len(o) for o in opts), letters=False)
        pl = _mcq_prompt(tok, q, max_len, letters=True) if len(q.options) == 4 else None
        if p is None:
            continue
        kept.append(qi)
        for oi, o in enumerate(opts):
            opt_rows.append(p + o)
            opt_meta.append((qi, oi, len(p), len(o)))
        if pl is not None:
            let_rows.append(pl)
            let_q.append(qi)
    mean_lp = {qi: [float("-inf")] * len(questions[qi].options) for qi in kept}
    l2_conf: Dict[int, List[float]] = {}
    for i, lp, z in _forward_rows(model, opt_rows, pad, device, autocast, batch_size):
        qi, oi, n_p, n_o = opt_meta[i]
        tgt = torch.tensor(opt_rows[i][n_p:], device=lp.device)
        # position t predicts token t+1: the option's tokens are read from
        # positions n_p-1 .. n_p+n_o-2
        sel = F.log_softmax(lp[n_p - 1:n_p - 1 + n_o].float(), dim=-1)
        mean_lp[qi][oi] = float(sel.gather(-1, tgt.unsqueeze(-1)).mean())
        if z is not None:
            # L2's confidence while it reads this option (T54)
            l2_conf.setdefault(qi, [0.0] * len(questions[qi].options))[oi] = \
                float(torch.sigmoid(z[n_p - 1:n_p - 1 + n_o].float()).mean())
    letter_pick: Dict[int, int] = {}
    for i, lp, _ in _forward_rows(model, let_rows, pad, device, autocast, batch_size):
        letter_pick[let_q[i]] = int(lp[-1, letter_ids].float().argmax())
    rows = []
    for qi in kept:
        q, scores = questions[qi], mean_lp[qi]
        pick = max(range(len(scores)), key=scores.__getitem__)
        row = {"q": qi, "pick": pick, "right": int(pick == q.answer),
               # the model's own confidence: its probability of the option
               # it picked, among the options (T54)
               "own_conf": float(torch.tensor(scores).softmax(0)[pick])}
        if qi in l2_conf:
            row["l2_conf"] = l2_conf[qi][pick]
        if qi in letter_pick:
            row["letter_pick"] = letter_pick[qi]
            row["letter_right"] = int(letter_pick[qi] == q.answer)
        rows.append(row)
    return mcq_summarise(rows, len(questions))


def _acc_ci(right: List[int], n: int = 1000, seed: int = 17) -> Tuple[float, float]:
    rng = random.Random(seed)
    stats = sorted(sum(right[rng.randrange(len(right))] for _ in right) / len(right)
                   for _ in range(n))
    return stats[int(0.025 * (n - 1))], stats[int(0.975 * (n - 1))]


def mcq_summarise(rows: List[Dict], n_total: int) -> Dict:
    right = [r["right"] for r in rows]
    out = {"n": len(rows), "skipped": n_total - len(rows), "rows": rows,
           "accuracy": sum(right) / max(1, len(right))}
    if right:
        out["ci"] = list(_acc_ci(right))
    lr = [r["letter_right"] for r in rows if "letter_right" in r]
    if lr:
        out["letter_accuracy"] = sum(lr) / len(lr)
    for sig in ("own_conf", "l2_conf"):
        if rows and all(sig in r for r in rows):
            out.setdefault("know", {})[sig] = knows([r[sig] for r in rows], right)
    return out


def knows(conf: Sequence[float], right: Sequence[int]) -> Dict[str, float]:
    """Does a confidence know when the answer is right (T54)?  AUROC
    (0.5 = no idea) and the % right on the half of the questions it is
    surest about vs the other half: a model that knows when it knows can
    answer the sure half and say 'not sure' to the rest."""
    n = len(right)
    order = sorted(range(n), key=lambda i: -conf[i])
    half = (n + 1) // 2
    sure, unsure = order[:half], order[half:]
    return {"auroc": auroc(torch.tensor(list(conf)), torch.tensor(list(right)).bool()),
            "sure_half": sum(right[i] for i in sure) / max(1, len(sure)),
            "unsure_half": sum(right[i] for i in unsure) / max(1, len(unsure)),
            "n": n}


def _autocast(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return lambda: torch.autocast(device.type, enabled=False)


def load_run(run_dir: str, with_args: bool = False):
    """(cfg, weights, checkpoint name) of a finished run: the checkpoint its
    final benchmark used (results.json ``selected``), else best.pt, else
    last.pt.  ``with_args``: also the run's training settings (seed, limit)."""
    last = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu",
                      weights_only=False)
    cfg = V2Config.from_dict(last["cfg"])
    if with_args:
        run_args = dict(last.get("run_args", {}))
        run_args["trained_gate_floor"] = trained_gate_floor(last)
        return (*load_run_weights(run_dir, last, cfg), run_args)
    return load_run_weights(run_dir, last, cfg)


def trained_gate_floor(last: Dict) -> float:
    """The gate floor the run's final benchmark used (T49).  The forced floor
    (T42) is set by the trainer's ramp and is not part of the weights, so it
    is rebuilt from last.pt exactly as ``--eval-only`` does; 0 for free runs."""
    from ekdant.train_gen import GateRamp

    fl = last.get("run_args", {})
    if not fl.get("gate_floor"):
        return 0.0
    ramp = GateRamp(fl["gate_floor"], fl.get("gate_floor_start", 200),
                    fl.get("gate_floor_steps", 2000),
                    slowdown=fl.get("floor_slowdown", 2.0),
                    step_down=fl.get("floor_step_down", 0.05),
                    min_target=fl.get("gate_floor_min", -1.0))
    if "gate_ramp" in last:
        ramp.load_state_dict(last["gate_ramp"])
    return ramp.floor()


def load_run_weights(run_dir: str, last: Dict, cfg: V2Config):
    chosen = "best.pt"
    res = os.path.join(run_dir, "results.json")
    if os.path.exists(res):
        with open(res, encoding="utf-8") as f:
            chosen = json.load(f).get("selected", {}).get("checkpoint", chosen)
    path = os.path.join(run_dir, chosen)
    if chosen == "last.pt" or not os.path.exists(path):
        return cfg, last["model"], "last.pt"
    return cfg, torch.load(path, map_location="cpu", weights_only=False)["model"], chosen


def think_settings(steps: int, passes: int = 1
                   ) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """(label, debate passes, descent steps) of the thinking-time test
    (T47): less and more of each of L2's two loops; the trained setting
    (``passes`` passes, ``steps`` steps) is the reference and is not
    repeated.  A model trained with 2 passes (T54) is also tried with 1 and 3."""
    grid = sorted({0, steps // 2, 2 * steps, 4 * steps} - {steps})
    pgrid = sorted({0, passes + 1} | ({passes - 1} if passes > 1 else set()))
    return ([(f"debate x{p}", p, None) for p in pgrid]
            + [(f"steps {s}", None, s) for s in grid])


def mcq_evaluate(model, tok, qs: Dict[str, List[Question]], device,
                 max_len: int, batch_size: int, l2_off: bool,
                 think: Sequence[str] = (), log=print) -> Dict:
    """Per test: % right; for L2 models also with L2 off and, for the tests
    in ``think``, with less / more L2 thinking (``EkdantLM.think``)."""
    autocast = _autocast(device)
    out: Dict = {}

    def sc(questions):
        return score_questions(model, tok, questions, device, autocast,
                               max_len, batch_size)

    for name, questions in qs.items():
        out[name] = sc(questions)
        msg = f"  {name}: {100 * out[name]['accuracy']:.1f}% right"
        if model.addon is not None and l2_off:
            with model.l2_off():
                off = sc(questions)
            out[name]["l2_off"] = off
            msg += f" (L2 off {100 * off['accuracy']:.1f}%)"
        log(msg + f" of {out[name]['n']}")
        if model.addon is not None and name in think:
            box = model.sandboxes()[0]
            out[name]["think"] = {"trained": f"debate x{box.debate_passes}, "
                                             f"steps {box.steps}"}
            for label, passes, steps in think_settings(box.steps, box.debate_passes):
                with model.think(passes, steps):
                    r = sc(questions)
                out[name]["think"][label] = r
                log(f"    thinking {label}: {100 * r['accuracy']:.1f}% right")
    return out


def bench_dataset(tok, cfg: V2Config, seed: int = 17, limit: Optional[int] = None):
    """The MedQuAD benchmark half exactly as training built it (same seed,
    same split), tokenised for ``cfg``; plus each row's question group."""
    from ekdant.gen_data import GenDataset, load_medquad, question_key, split_by_question

    _, _, bench = split_by_question(load_medquad(limit, seed), seed)
    ds = GenDataset(bench, tok, cfg.max_seq_len, cfg.max_answer)
    return ds, [question_key(e.question) for e in bench]


def bench_nll(model, ds, pad: int, device, batch_size: int = 16) -> Dict:
    """Reference-answer NLL on ``ds`` (rows kept for paired CIs), batched by
    length; L2's internal diagnostics are averaged in ``addon``."""
    from ekdant.gen_data import gen_collate

    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["input_ids"]))
    loader = (gen_collate([ds[i] for i in order[s:s + batch_size]], pad)
              for s in range(0, len(order), batch_size))
    return answer_nll(model, loader, device, _autocast(device))


def ppl_evaluate(model, ds, groups: List[str], pad: int, device, batch_size: int,
                 l2_off: bool, think: bool, log=print) -> Dict:
    """MedQuAD benchmark perplexity with L2 on, off, and with less / more
    L2 thinking - the perplexity half of the one-run benchmark (T48)."""
    out = {"on": bench_nll(model, ds, pad, device, batch_size)}
    out["row_group"] = {str(i): groups[i] for i in out["on"]["row_idx"]}
    msg = f"  medquad_bench ppl {out['on']['ppl']:.3f}"
    if model.addon is not None and l2_off:
        with model.l2_off():
            out["off"] = bench_nll(model, ds, pad, device, batch_size)
        msg += f" (L2 off {out['off']['ppl']:.3f})"
    log(msg)
    if model.addon is not None and think:
        box = model.sandboxes()[0]
        out["think"] = {"trained": f"debate x{box.debate_passes}, steps {box.steps}"}
        for label, passes, steps in think_settings(box.steps, box.debate_passes):
            with model.think(passes, steps):
                out["think"][label] = bench_nll(model, ds, pad, device, batch_size)
            log(f"    thinking {label}: ppl {out['think'][label]['ppl']:.3f}")
    return out


# --- the second benchmark: the model writes the explanation (T56) ----------
# medmcqa_explain  MedMCQA validation: the question with its four options ->
#                  "<right option>. <the expert's explanation>" (out of
#                  domain: never trained on)
# pubmedqa_explain PubMedQA pqa_labeled: question + abstract -> the authors'
#                  conclusion (the format of phase 1, but these 1,000
#                  expert-labelled questions are never trained on)
EXPLAIN_SETS = ("medmcqa_explain", "pubmedqa_explain")
EXPLAIN_TITLE = {"medmcqa_explain": "MedMCQA explanation",
                 "pubmedqa_explain": "PubMedQA conclusion"}


def _clean(s) -> str:
    return " ".join(str(s or "").split())


def explain_examples(name: str, limit: Optional[int] = 1000, seed: int = 17):
    """(GenExamples, metas): for MedMCQA each meta is (options, right index)
    so a generated explanation can be checked for naming the right answer."""
    from datasets import load_dataset
    from ekdant.gen_data import GenExample

    out = []
    if name == "medmcqa_explain":
        ds = _offline_first(lambda: load_dataset("openlifescienceai/medmcqa",
                                                 split="validation"))
        for r in ds:
            opts = [_clean(r[k]) for k in ("opa", "opb", "opc", "opd")]
            exp, cop = _clean(r.get("exp")), r.get("cop")
            if cop not in (0, 1, 2, 3) or len(exp) < 60 or not all(opts):
                continue
            q = (f"{_clean(r['question'])}\nOptions: {opts[0]}; {opts[1]}; "
                 f"{opts[2]}; {opts[3]}\nExplain the answer.")
            out.append((GenExample(q, f"{opts[cop]}. {exp}"), (opts, cop)))
    elif name == "pubmedqa_explain":
        ds = _offline_first(lambda: load_dataset("qiaojin/PubMedQA", "pqa_labeled",
                                                 split="train"))
        for r in ds:
            ans = _clean(r["long_answer"])
            if ans:
                out.append((GenExample(_clean(r["question"]), ans,
                                       context=" ".join(r["context"]["contexts"])), None))
    else:
        raise ValueError(f"unknown explanation set {name!r}")
    if limit is not None and limit < len(out):
        out = random.Random(seed).sample(out, limit)
    return [e for e, _ in out], [m for _, m in out]


def names_answer(pred: str, options: Sequence[str], right: int) -> int:
    """1 if the generated text names the right option first: it contains the
    right option's words (in order, contiguous) earlier than any other
    option; an option that is part of a longer one counts for the longer."""
    words = normalise_text(pred)

    def first(opt: str) -> Optional[int]:
        o = normalise_text(opt)
        if not o:
            return None
        for i in range(len(words) - len(o) + 1):
            if words[i:i + len(o)] == o:
                return i
        return None

    pos = [(first(o), -len(normalise_text(o)), k) for k, o in enumerate(options)]
    found = sorted(p for p in pos if p[0] is not None)
    return int(bool(found) and found[0][2] == right)


def explain_datasets(tok, cfg: V2Config, limit: Optional[int] = 1000, seed: int = 17,
                     max_answer: int = 256) -> Dict:
    from ekdant.gen_data import GenDataset

    out = {}
    for name in EXPLAIN_SETS:
        ex, metas = explain_examples(name, limit, seed)
        out[name] = (GenDataset(ex, tok, cfg.max_seq_len, max_answer), metas)
    return out


def explain_evaluate(model, tok, sets: Dict, device, batch_size: int, l2_off: bool,
                     n_gen: int = 200, max_new: int = 128, log=print) -> Dict:
    """Per set: the reference explanation's perplexity (L2 on / off; rows kept
    for paired CIs) and, on the first ``n_gen`` rows, a greedy generation
    scored against the reference (ROUGE-L, token F1) and, for MedMCQA,
    whether it names the right answer."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out: Dict = {}
    for name, (ds, metas) in sets.items():
        r = {"on": bench_nll(model, ds, pad, device, batch_size)}
        msg = f"  {name}: ppl {r['on']['ppl']:.3f}"
        if model.addon is not None and l2_off:
            with model.l2_off():
                r["off"] = bench_nll(model, ds, pad, device, batch_size)
            msg += f" (L2 off {r['off']['ppl']:.3f})"
        rows = list(range(min(n_gen, len(ds))))
        outs = generate_greedy(model, [ds.prompt_ids(i) for i in rows], max_new, device,
                               tok.eos_token_id, pad, batch_size, _autocast(device))
        preds = [tok.decode(o, skip_special_tokens=True).strip() for o in outs]
        gen_rows = []
        for i, p in zip(rows, preds):
            ref = ds.example(i).answer
            row = {"i": i, "rouge_l": rouge_l(p, ref), "token_f1": token_f1(p, ref)}
            meta = metas[ds.rows[i][2]]
            if meta is not None:
                row["named"] = names_answer(p, meta[0], meta[1])
            gen_rows.append(row)
        n = max(1, len(gen_rows))
        g = {"rows": gen_rows, "n": len(gen_rows),
             "rouge_l": sum(x["rouge_l"] for x in gen_rows) / n,
             "token_f1": sum(x["token_f1"] for x in gen_rows) / n,
             "examples": [{"question": ds.example(i).question[:300], "model": p[:500],
                           "reference": ds.example(i).answer[:500]}
                          for i, p in zip(rows[:3], preds[:3])]}
        if any("named" in x for x in gen_rows):
            g["named_right"] = sum(x.get("named", 0) for x in gen_rows) / n
        r["gen"] = g
        msg += f", ROUGE-L {100 * g['rouge_l']:.1f}, F1 {100 * g['token_f1']:.1f}"
        if "named_right" in g:
            msg += f", names the right answer {100 * g['named_right']:.1f}%"
        log(msg + f" ({len(ds)} rows, {g['n']} generated)")
        out[name] = r
    return out


def finished_runs(runs_dir: str) -> List[str]:
    """Run folders whose training finished (last.pt and a final test in
    results.json); a broken or zero-shot folder is skipped."""
    out = []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if "broken" in name or "zero_shot" in name or not os.path.isdir(d):
            continue
        rp = os.path.join(d, "results.json")
        if not (os.path.exists(os.path.join(d, "last.pt")) and os.path.exists(rp)):
            continue
        try:
            with open(rp, encoding="utf-8") as f:
                if "test" in json.load(f):
                    out.append(d)
        except (OSError, ValueError):
            continue
    return out


def _score_state(run_dir: str, names: Sequence[str]) -> str:
    """'complete' if ``run_dir`` already has an up-to-date mcq.json with every
    requested test, 'missing' if it has none, else why it is scored again."""
    path = os.path.join(run_dir, "mcq.json")
    if not os.path.exists(path):
        return "missing"
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return "unreadable mcq.json"
    v = m.get("version", 1)
    if v < 3:
        return "scored with an older version (no perplexity / thinking numbers)"
    if v < MCQ_VERSION and "forced" in str(m.get("l2", "")):
        # version 3 scored forced runs without their forced gate floor (T49);
        # free runs have floor 0, so their numbers stay valid
        return "scored without its forced gate floor (T49)"
    if v < 5:
        return "scored without the confidence test (T54)"
    if v < 6:
        return "scored without the explanation benchmark (T56)"
    if any(n not in m for n in names):
        return "some tests missing"
    return "complete"


def _log(msg: str) -> None:
    print(msg, flush=True)


def mcq_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ekdant.gen_eval mcq",
                                 description=MCQ_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="every finished run in --runs-dir, plus plain SmolLM2; "
                         "runs already scored are kept (see --rescore)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--rescore", action="store_true",
                    help="with --all: score runs that already have mcq.json again")
    ap.add_argument("--compare", action="store_true",
                    help="only print the tables from existing mcq.json files")
    ap.add_argument("--zero-shot", action="store_true",
                    help="also score plain SmolLM2 (no fine-tuning) -> <runs-dir>/zero_shot_smollm2")
    ap.add_argument("--datasets", default=",".join(MCQ_DATASETS))
    ap.add_argument("--think-datasets", default="pubmedqa",
                    help="tests that also get the thinking-time test (T47); empty = none")
    ap.add_argument("--limit", type=int, default=None,
                    help="questions per dataset (default: all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-l2-off", action="store_true")
    ap.add_argument("--no-ppl", action="store_true",
                    help="skip the MedQuAD perplexity part (L2 on / off / thinking)")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--baseline", default=None,
                    help="run name the leaderboard compares against "
                         "(default: the last run by name with 'baseline' in it)")
    ap.add_argument("--markdown", default=None,
                    help="also write the tables to this markdown file")
    ap.add_argument("--no-explain", action="store_true",
                    help="skip the explanation benchmark (T56)")
    ap.add_argument("--explain-only", action="store_true",
                    help="add the explanation benchmark to runs already scored "
                         "(their mcq.json keeps every other number)")
    ap.add_argument("--explain-limit", type=int, default=1000,
                    help="rows per explanation set (perplexity)")
    ap.add_argument("--explain-gen", type=int, default=200,
                    help="rows per explanation set that are also generated")
    ap.add_argument("--explain-max-new", type=int, default=128)
    args = ap.parse_args(argv)
    zs_dir = os.path.join(args.runs_dir, "zero_shot_smollm2")
    runs = list(args.runs)
    if args.all:
        runs = [zs_dir] + finished_runs(args.runs_dir)
        if args.markdown is None:
            args.markdown = "leaderboard.md"
    elif args.zero_shot:
        runs.insert(0, zs_dir)
    if not args.compare:
        from ekdant.train_gen import pick_device

        names = [d for d in args.datasets.split(",") if d]
        think = [d for d in args.think_datasets.split(",") if d]
        think_ppl = ["medquad"]                    # perplexity thinking test: always
        bench_cache: Dict = {}
        explain_cache: Dict = {}
        device = pick_device(args.gpu, args.min_free_gb)
        qs = None
        _log("models: " + ", ".join(os.path.basename(r) for r in runs))
        for k, run_dir in enumerate(runs, 1):
            if args.all and not args.rescore:
                state_ = _score_state(run_dir, names)
                if state_ == "complete":
                    _log(f"[{k}/{len(runs)}] {run_dir}: already scored - kept")
                    continue
                if state_ != "missing":
                    _log(f"[{k}/{len(runs)}] {run_dir}: {state_} - scoring again")
            try:
                if run_dir == zs_dir:
                    cfg, state, chosen = V2Config(addon="none"), None, "pretrained"
                    run_args: Dict = {}
                    os.makedirs(run_dir, exist_ok=True)
                else:
                    cfg, state, chosen, run_args = load_run(run_dir, with_args=True)
                _log(f"[{k}/{len(runs)}] {run_dir} ({chosen}, L2: {describe_l2(cfg)})")
                apply_hf_token(cfg.hf_token_file)
                tok = load_tokenizer(cfg.backbone)
                if qs is None and not args.explain_only:
                    qs = {n: load_questions(n, args.limit, args.seed) for n in names}
                model = EkdantLM(cfg, load_backbone(cfg.backbone))
                if state is not None:
                    model.load_state_dict(state)
                floor = float(run_args.get("trained_gate_floor", 0.0))
                if model.addon is not None and floor > 0:
                    model.addon.set_gate_floor(floor)    # forced runs (T49)
                    _log(f"  gate floor {floor:.3f} (as in training)")
                model.to(device)
                if args.explain_only:
                    with open(os.path.join(run_dir, "mcq.json"), encoding="utf-8") as f:
                        res = json.load(f)
                    if res.get("version", 1) < 5:
                        raise ValueError(f"{run_dir}: mcq.json is version {res.get('version')}; "
                                         "score it fully first (without --explain-only)")
                else:
                    res = mcq_evaluate(model, tok, qs, device, cfg.max_seq_len,
                                       args.batch_size, not args.no_l2_off, think, _log)
                res["checkpoint"] = chosen
                res["l2"] = describe_l2(cfg)
                res["version"] = MCQ_VERSION
                res["gate_floor"] = floor
                res["seed"] = int(run_args.get("seed", 17))
                res["params_total"] = sum(p.numel() for p in model.parameters())
                res["params_l2"] = model.addon_params()
                if not args.no_explain:
                    ekey = (cfg.backbone, cfg.max_seq_len)
                    if ekey not in explain_cache:
                        explain_cache.clear()
                        explain_cache[ekey] = explain_datasets(tok, cfg, args.explain_limit)
                    res["explain"] = explain_evaluate(
                        model, tok, explain_cache[ekey], device, args.batch_size,
                        not args.no_l2_off, args.explain_gen, args.explain_max_new, _log)
                if not args.no_ppl and not args.explain_only:
                    key = (res["seed"], run_args.get("limit"), cfg.backbone,
                           cfg.max_seq_len, cfg.max_answer)
                    if key not in bench_cache:
                        bench_cache.clear()          # keep one tokenised copy
                        bench_cache[key] = bench_dataset(tok, cfg, key[0], key[1])
                    ds, groups = bench_cache[key]
                    pad = tok.pad_token_id if tok.pad_token_id is not None \
                        else tok.eos_token_id
                    res["medquad_ppl"] = ppl_evaluate(
                        model, ds, groups, pad, device, args.batch_size,
                        not args.no_l2_off, "medquad" in think_ppl, _log)
                with open(os.path.join(run_dir, "mcq.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f)
                del model
            except Exception as e:  # one broken run must not stop the others
                if not args.all:
                    raise
                _log(f"FAILED {run_dir}: {type(e).__name__}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _log("=== ALL DONE")
    runs = [r for r in runs if os.path.exists(os.path.join(r, "mcq.json"))]
    mcq_compare(runs, args.baseline, args.markdown)


MCQ_TITLE = {"medmcqa": "MedMCQA", "medqa": "MedQA", "pubmedqa": "PubMedQA"}


def describe_l2(cfg: V2Config) -> str:
    """Short label of where L2 sits and whether it is forced."""
    if cfg.addon == "none":
        return "-"
    if cfg.addon_layers:
        where = f"mix ({len(cfg.addon_layers) + int(cfg.addon_final)} sites)"
    elif cfg.addon_layer >= 1:
        where = f"after layer {cfg.addon_layer}"
    else:
        where = "end"
    v3 = [n for n, on in (("aligned", getattr(cfg, "sandbox_aligned", False)),
                          ("context", getattr(cfg, "sandbox_context", False)),
                          ("confidence", getattr(cfg, "sandbox_confidence", False)),
                          ("free views", getattr(cfg, "sandbox_free_views", False))) if on]
    return (f"{where}, {'forced' if cfg.gate_force else 'free'}"
            + (f", v3 {'+'.join(v3)}" if v3 else "")
            + (f", thinks x{cfg.sandbox_train_passes}"
               if getattr(cfg, "sandbox_train_passes", 1) > 1 else ""))


def _paired_acc(ref: Dict, run: Dict, names: Sequence[str]) -> Optional[Dict]:
    """Paired bootstrap of the accuracy difference (run - ref), averaged over
    ``names`` with equal weight per test (the leaderboard average)."""
    a, b, g, w = [], [], [], []
    for n in names:
        if n not in ref or n not in run:
            return None
        ra = {row["q"]: row["right"] for row in ref[n]["rows"]}
        rb = {row["q"]: row["right"] for row in run[n]["rows"]}
        keys = sorted(set(ra) & set(rb))
        if not keys:
            return None
        k = 1.0 / (len(keys) * len(names))
        for q in keys:
            a.append(ra[q] * k)
            b.append(rb[q] * k)
            g.append(f"{n}:{q}")
            w.append(k)
    return paired_bootstrap(a, b, g, weights=w)


def _star(ci: Optional[Dict]) -> str:
    return "*" if ci and (ci["lo"] > 0 or ci["hi"] < 0) else ""


def mcq_leaderboard(run_dirs: List[str], baseline: Optional[str] = None) -> str:
    """A model-card style leaderboard (markdown): % right per test, the
    average, the gain over the baseline (* = paired 95 % CI excludes 0) and
    the generative benchmark from results.json where the run has one."""
    runs = []
    for d in run_dirs:
        with open(os.path.join(d, "mcq.json"), encoding="utf-8") as f:
            m = json.load(f)
        res = None
        rp = os.path.join(d, "results.json")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                res = json.load(f)
        runs.append((os.path.basename(d.rstrip("/\\")), m, res))
    names = [n for n in MCQ_DATASETS if any(n in m for _, m, _ in runs)]
    # default: the newest baseline by name (v25_baseline over v22_baseline)
    base_name = baseline or next((n for n, _, _ in reversed(sorted(runs, key=lambda r: r[0]))
                                  if "baseline" in n), runs[0][0])
    base = next(r for r in runs if r[0] == base_name)

    def avg(m):
        return (sum(m[n]["accuracy"] for n in names) / len(names)
                if all(n in m for n in names) else None)

    rows = sorted(runs, key=lambda r: -(avg(r[1]) if avg(r[1]) is not None else -1))
    head = (["#", "Model", "L2"] + [MCQ_TITLE[n] for n in names]
            + ["Average", "vs baseline", "MedQuAD ppl (lower = better)", "ROUGE-L", "F1"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    bres = (base[2] or {}).get("test", {}).get("medquad_bench") \
        or base[1].get("medquad_ppl", {}).get("on")
    for i, (name, m, res) in enumerate(rows, 1):
        is_base = name == base_name
        cells = [str(i), f"**{name}** (baseline)" if is_base else name, m.get("l2", "?")]
        for n in names:
            if n not in m:
                cells.append("-")
                continue
            c = f"{100 * m[n]['accuracy']:.1f}"
            if not is_base and n in base[1]:
                ci = _paired_acc(base[1], m, [n])
                if ci:
                    c += f" ({100 * ci['diff']:+.1f}{_star(ci)})"
            cells.append(c)
        a = avg(m)
        cells.append(f"**{100 * a:.1f}**" if a is not None else "-")
        if is_base:
            cells.append("-")
        else:
            ci = _paired_acc(base[1], m, names)
            cells.append(f"{100 * ci['diff']:+.1f} points{_star(ci)}" if ci else "-")
        b = (res or {}).get("test", {}).get("medquad_bench") \
            or m.get("medquad_ppl", {}).get("on")
        if b is None:
            cells += ["-", "-", "-"]
        else:
            c = f"{b['ppl']:.3f}"
            same_split = m.get("seed", 17) == base[1].get("seed", 17)
            if not same_split and not is_base:
                c += " (other seed: different split)"
            if (bres is not None and not is_base and same_split
                    and "row_nll" in b and "row_nll" in bres):
                ra, rb = _rows(bres), _rows(b)
                keys = sorted(set(ra) & set(rb))
                if "row_group" in bres:
                    grp = dict(zip(bres["row_idx"], bres["row_group"]))
                else:
                    rg = base[1].get("medquad_ppl", {}).get("row_group", {})
                    grp = {k: rg.get(str(k), k) for k in ra}
                ci = paired_bootstrap([ra[k][0] for k in keys], [rb[k][0] for k in keys],
                                      [str(grp[k]) for k in keys],
                                      weights=[ra[k][1] for k in keys])
                c += f" ({100 * (b['ppl'] / bres['ppl'] - 1):+.1f}%{_star(ci)})"
            cells.append(c)
            cells.append(f"{100 * b['rouge_l']:.1f}" if "rouge_l" in b else "-")
            cells.append(f"{100 * b['token_f1']:.1f}" if "token_f1" in b else "-")
        lines.append("| " + " | ".join(cells) + " |")
    n_q = {n: next((m[n]["n"] for _, m, _ in runs if n in m), 0) for n in names}
    lines += ["",
              "% of questions answered right ("
              + ", ".join(f"{MCQ_TITLE[n]} {n_q[n]:,} questions, chance "
                          f"{100 * MCQ_CHANCE[n]:.0f}%" for n in names)
              + "). Average = mean of the tests. (+x) = points more than the "
              f"baseline `{base_name}`; perplexity (+x%) = change vs the baseline. "
              "* = real difference (paired 95% bootstrap CI excludes 0); "
              "no * = within noise. ROUGE-L / F1 = word overlap of generated "
              "answers with the reference (MedQuAD, %)."]
    return "\n".join(lines)


def think_table(run_dirs: List[str]) -> str:
    """The thinking-time test (T47) as a markdown table: % right with less
    or more L2 thinking, change vs the trained setting (* = real)."""
    out = []
    for d in run_dirs:
        with open(os.path.join(d, "mcq.json"), encoding="utf-8") as f:
            m = json.load(f)
        for n in MCQ_DATASETS:
            if n in m and "think" in m[n]:
                out.append((os.path.basename(d.rstrip("/\\")), n, m))
    if not out:
        return ""
    labels: List[str] = []
    for _, n, m in out:
        labels += [k for k in m[n]["think"] if k != "trained" and k not in labels]
    head = ["Model", "Test", "trained setting"] + labels + ["L2 off"]
    lines = ["### Thinking time: % right", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, n, m in out:
        ref = m[n]
        cells = [name, MCQ_TITLE[n],
                 f"{100 * ref['accuracy']:.1f} ({ref['think']['trained']})"]
        for lab in labels + ["L2 off"]:
            r = ref.get("l2_off") if lab == "L2 off" else ref["think"].get(lab)
            if not isinstance(r, dict):
                cells.append("-")
                continue
            ci = _paired_acc({n: ref}, {n: r}, [n])
            cells.append(f"{100 * r['accuracy']:.1f} ({100 * ci['diff']:+.1f}{_star(ci)})"
                         if ci else f"{100 * r['accuracy']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Thinking-time test: the same trained model answers with L2's "
              "debate run more or fewer times than trained (the trained "
              "setting column shows how many) and with fewer or more "
              "anchored-descent steps (same step size), no retraining. (+x) = "
              "points vs the trained setting; * = real difference (paired 95% "
              "CI excludes 0). If L2 reasons step by step, less thinking should "
              "cost accuracy and more may help."]
    return "\n".join(lines)


def know_table(run_dirs: List[str]) -> str:
    """Does the model know when it is right (T54)?  Per test, how well the
    model's own probability and L2's confidence tell right answers from
    wrong ones, and the % right on the half each is surest about."""
    lines = []
    for d in run_dirs:
        name, m, _ = _load_both(d)
        for n in MCQ_DATASETS:
            k = m.get(n, {}).get("know")
            if not k:
                continue
            cells = [name, MCQ_TITLE[n], f"{100 * m[n]['accuracy']:.1f}"]
            for sig in ("own_conf", "l2_conf"):
                s = k.get(sig)
                cells += ([f"{s['auroc']:.3f}",
                           f"{100 * s['sure_half']:.1f} / {100 * s['unsure_half']:.1f}"]
                          if s else ["-", "-"])
            lines.append("| " + " | ".join(cells) + " |")
    tok = []
    for d in run_dirs:
        name, m, _ = _load_both(d)
        on = m.get("medquad_ppl", {}).get("on", {})
        if "own_confidence" in on:
            l2 = on.get("confidence", {}).get("auroc")
            tok.append(f"| {name} | {on['own_confidence']['auroc']:.3f} | "
                       + (f"{l2:.3f}" if l2 is not None else "-") + " |")
    if not lines and not tok:
        return ""
    out = ["### Does the model know when it is right? (T54)", ""]
    if lines:
        head = ["Model", "Test", "% right (all)", "Own probability: AUROC",
                "Own probability: % right sure / unsure half", "L2 confidence: AUROC",
                "L2 confidence: % right sure / unsure half"]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + lines
        out += ["", "Per question: the model's own confidence = its probability of the "
                "option it picked; L2 confidence = L2's learned confidence while reading "
                "that option. AUROC = chance that a right answer gets a higher confidence "
                "than a wrong one (0.5 = no idea, 1 = perfect). Sure / unsure half = % right "
                "on the half of the questions the signal is surest / least sure about: a "
                "big gap means the model can answer when sure and say 'not sure' otherwise. "
                "L2 adds something only if its numbers beat the model's own probability."]
    if tok:
        out += ["", "| Model | Next-word AUROC: own top-word probability | "
                "Next-word AUROC: L2 confidence |", "|---|---|---|"] + tok
        out += ["", "MedQuAD benchmark, every answer token: how well each signal tells the "
                "model's right next words from its wrong ones."]
    return "\n".join(out)


def _load_both(d: str) -> Tuple[str, Dict, Optional[Dict]]:
    with open(os.path.join(d, "mcq.json"), encoding="utf-8") as f:
        m = json.load(f)
    res = None
    rp = os.path.join(d, "results.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            res = json.load(f)
    return os.path.basename(d.rstrip("/\\")), m, res


def _nll_ci(a: Dict, b: Dict, groups: Dict[str, str]) -> Optional[Dict]:
    """Paired CI of the per-token NLL difference b - a (same benchmark rows)."""
    ra, rb = _rows(a), _rows(b)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return None
    return paired_bootstrap([ra[k][0] for k in keys], [rb[k][0] for k in keys],
                            [str(groups.get(str(k), k)) for k in keys],
                            weights=[ra[k][1] for k in keys])


def dependence_table(run_dirs: List[str]) -> str:
    """Does the trained model depend on L2?  Everything with L2 on vs off."""
    lines = []
    for d in run_dirs:
        name, m, _ = _load_both(d)
        mp = m.get("medquad_ppl", {})
        if "off" not in mp and not any("l2_off" in m.get(n, {}) for n in MCQ_DATASETS):
            continue
        cells = [name, m.get("l2", "?")]
        if "off" in mp:
            ci = _nll_ci(mp["on"], mp["off"], mp.get("row_group", {}))
            cells.append(f"{mp['on']['ppl']:.3f} -> {mp['off']['ppl']:.3f}")
            cells.append(f"{ci['diff']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]{_star(ci)}"
                         if ci else "-")
        else:
            cells += ["-", "-"]
        for n in MCQ_DATASETS:
            r = m.get(n, {})
            if "l2_off" not in r:
                cells.append("-")
                continue
            ci = _paired_acc({n: r}, {n: r["l2_off"]}, [n])
            cells.append(f"{100 * r['accuracy']:.1f} -> {100 * r['l2_off']['accuracy']:.1f}"
                         + (f" ({100 * ci['diff']:+.1f}{_star(ci)})" if ci else ""))
        lines.append("| " + " | ".join(cells) + " |")
    if not lines:
        return ""
    head = ["Model", "L2", "MedQuAD ppl: with L2 -> without",
            "NLL rise without L2 [95% CI]"] + [f"{MCQ_TITLE[n]} % right: with -> without"
                                             for n in MCQ_DATASETS]
    return "\n".join(["### Does the model depend on L2? (L2 switched off after training)", "",
                      "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + lines
                     + ["", "Same trained model, every L2 site switched off (exactly the "
                        "backbone). A positive NLL rise with * = the model needs L2 for its "
                        "answers; a negative one = it does better without L2. (+x) = points "
                        "of % right; * = real difference (paired 95% CI excludes 0)."])


def think_ppl_table(run_dirs: List[str]) -> str:
    """The thinking-time test on MedQuAD perplexity (T48)."""
    out = []
    for d in run_dirs:
        name, m, _ = _load_both(d)
        if "think" in m.get("medquad_ppl", {}):
            out.append((name, m["medquad_ppl"]))
    if not out:
        return ""
    labels: List[str] = []
    for _, mp in out:
        labels += [k for k in mp["think"] if k != "trained" and k not in labels]
    head = ["Model", "trained setting"] + labels + ["L2 off"]
    lines = ["### Thinking time: MedQuAD perplexity (lower = better)", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, mp in out:
        cells = [name, f"{mp['on']['ppl']:.3f} ({mp['think']['trained']})"]
        for lab in labels + ["L2 off"]:
            r = mp.get("off") if lab == "L2 off" else mp["think"].get(lab)
            if not isinstance(r, dict):
                cells.append("-")
                continue
            ci = _nll_ci(mp["on"], r, mp.get("row_group", {}))
            cells.append(f"{r['ppl']:.3f} ({100 * (r['ppl'] / mp['on']['ppl'] - 1):+.1f}%"
                         f"{_star(ci)})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "(+x%) = perplexity change vs the trained setting; * = real "
              "difference (paired 95% CI of the per-token NLL excludes 0)."]
    return "\n".join(lines)


def facts_table(run_dirs: List[str]) -> str:
    """Size, cost, gate and L2's inner state of every model."""
    head = ["Model", "L2", "Params", "L2 params", "Training time", "Final gate (mean size)",
            "Brake events", "Rollbacks", "L2 spread ratio", "L2 anchor energy",
            "Confidence AUROC"]
    lines = ["### Model facts", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d in run_dirs:
        name, m, res = _load_both(d)
        cost = (res or {}).get("cost", {})
        test = (res or {}).get("test", {})
        diag = m.get("medquad_ppl", {}).get("on", {}).get("addon", {})
        pt = cost.get("params_total", m.get("params_total"))
        pl = cost.get("params_addon", m.get("params_l2"))
        wall = cost.get("wall_clock_s")
        cells = [name, m.get("l2", "?"),
                 f"{pt / 1e6:.1f}M" if pt else "-",
                 f"{pl / 1e6:.1f}M" if pl else "-",
                 (f"{wall / 3600:.1f} h" if wall else "-"),
                 f"{test['gate_magnitude']:.3f}" if test.get("gate_magnitude") else "-",
                 str(cost["brake_events"]) if "brake_events" in cost else "-",
                 str(cost["rollbacks"]) if "rollbacks" in cost else "-",
                 f"{diag['spread_ratio']:.3g}" if "spread_ratio" in diag else "-",
                 f"{diag['anchor_energy']:.3g}" if "anchor_energy" in diag else "-"]
        cal = test.get("medquad_bench", {}).get("confidence", {})
        cells.append(f"{cal['auroc']:.3f}" if cal.get("auroc") == cal.get("auroc")
                     and "auroc" in cal else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Training time = wall clock of the whole run (both phases). Spread "
              "ratio = how far L2's 4 trajectories stay apart relative to the input "
              "(about 0 = they agree on everything, no real debate); anchor energy = "
              "their distance to the anchor after the descent. Both on the MedQuAD "
              "benchmark, averaged. Confidence AUROC (L2 v3) = how well L2's learned "
              "confidence tells the model's right next tokens from its wrong ones "
              "(0.5 = no idea, 1 = perfect)."]
    return "\n".join(lines)


def full_report(run_dirs: List[str], baseline: Optional[str] = None) -> str:
    """Every table of the one-run benchmark, ready for the paper."""
    parts = ["## Benchmark of every model (T46-T48)", "",
             "### Leaderboard", "", mcq_leaderboard(run_dirs, baseline)]
    for part in (explain_table(run_dirs, baseline), dependence_table(run_dirs),
                 think_table(run_dirs),
                 think_ppl_table(run_dirs), know_table(run_dirs),
                 facts_table(run_dirs)):
        if part:
            parts += ["", part]
    return "\n".join(parts)


def mcq_compare(run_dirs: List[str], baseline: Optional[str] = None,
                markdown: Optional[str] = None) -> None:
    board = full_report(run_dirs, baseline)
    print("\n" + board + "\n")
    if markdown:
        with open(markdown, "w", encoding="utf-8") as f:
            f.write(board + "\n")
        print(f"leaderboard written to {markdown}")
    runs = []
    for d in run_dirs:
        with open(os.path.join(d, "mcq.json"), encoding="utf-8") as f:
            runs.append((os.path.basename(d.rstrip("/\\")), json.load(f)))
    names = [n for n in MCQ_DATASETS if any(n in r for _, r in runs)]
    print("details:")
    print("\n% right (95% CI); chance: "
          + ", ".join(f"{n} {100 * MCQ_CHANCE[n]:.0f}%" for n in names))
    print(f"{'run':24s}" + "".join(f"{n:>22s}" for n in names))
    for name, r in runs:
        cells = []
        for n in names:
            if n not in r:
                cells.append(f"{'-':>22s}")
                continue
            a, ci = r[n]["accuracy"], r[n].get("ci", [0, 0])
            cells.append(f"{100 * a:6.1f} [{100 * ci[0]:4.1f},{100 * ci[1]:5.1f}]".rjust(22))
        print(f"{name:24s}" + "".join(cells))
    extra = [(name, r) for name, r in runs if any("l2_off" in r.get(n, {}) for n in names)]
    if extra:
        print("\nwith L2 switched off:")
        for name, r in extra:
            print(f"{name:24s}" + "".join(
                f"{100 * r[n]['l2_off']['accuracy']:22.1f}" if "l2_off" in r.get(n, {})
                else f"{'-':>22s}" for n in names))
    print("\nletter method (options listed A-D; never trained on, expect ~chance):")
    for name, r in runs:
        print(f"{name:24s}" + "".join(
            f"{100 * r[n]['letter_accuracy']:22.1f}" if "letter_accuracy" in r.get(n, {})
            else f"{'-':>22s}" for n in names))
    ref_name, ref = runs[0]
    for name, r in runs[1:]:
        for n in names:
            if n not in r or n not in ref:
                continue
            a = {row["q"]: row["right"] for row in ref[n]["rows"]}
            b = {row["q"]: row["right"] for row in r[n]["rows"]}
            keys = sorted(set(a) & set(b))
            if not keys:
                continue
            ci = paired_bootstrap([a[k] for k in keys], [b[k] for k in keys],
                                  [str(k) for k in keys])
            verdict = "real difference" if ci["lo"] > 0 or ci["hi"] < 0 else "tie"
            print(f"{name} - {ref_name} on {n}: {100 * ci['diff']:+.1f} points "
                  f"95% CI [{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}] "
                  f"({verdict}; positive = more right)")


def _pooled_acc(pairs: List[Tuple[Dict, Dict]], names: Sequence[str]) -> Optional[Dict]:
    """Paired bootstrap of the accuracy gain (model - baseline) averaged over
    every seed pair and over ``names`` (equal weight per seed and per test).
    The questions are the same in every seed, so one question in every seed
    is one resampling group: the CI sees seed-to-seed and question noise."""
    a, b, g, w = [], [], [], []
    for base, model in pairs:
        for n in names:
            if n not in base or n not in model:
                return None
            ra = {row["q"]: row["right"] for row in base[n]["rows"]}
            rb = {row["q"]: row["right"] for row in model[n]["rows"]}
            keys = sorted(set(ra) & set(rb))
            if not keys:
                return None
            k = 1.0 / (len(keys) * len(names) * len(pairs))
            for q in keys:
                a.append(ra[q] * k)
                b.append(rb[q] * k)
                g.append(f"{n}:{q}")
                w.append(k)
    return paired_bootstrap(a, b, g, weights=w)


def _pooled_nll(pairs: List[Tuple[Dict, Dict]]) -> Optional[Dict]:
    """Per-token NLL change (model - baseline), the mean over the seed pairs.
    Each seed has its own MedQuAD benchmark half, so groups are per seed."""
    a, b, g, w = [], [], [], []
    for p, (base, model) in enumerate(pairs):
        ma, mb = base.get("medquad_ppl"), model.get("medquad_ppl")
        if not ma or not mb:
            return None
        ra, rb = _rows(ma["on"]), _rows(mb["on"])
        keys = sorted(set(ra) & set(rb))
        if not keys:
            return None
        c = 1.0 / (len(pairs) * sum(ra[k][1] for k in keys))
        groups = ma.get("row_group", {})
        for k in keys:
            a.append(ra[k][0] * c)
            b.append(rb[k][0] * c)
            g.append(f"{p}:{groups.get(str(k), k)}")
            w.append(ra[k][1] * c)
    return paired_bootstrap(a, b, g, weights=w)


def _pool(pairs: List[Tuple[Dict, Dict]], get, per_seed: bool) -> Optional[Dict]:
    """Paired bootstrap of (model - baseline), the mean over the seed pairs of
    each pair's weighted mean difference.  ``get(m)`` -> {row: (value,
    weight)} or None; ``per_seed`` = the rows differ between seeds."""
    a, b, g, w = [], [], [], []
    for p, (base, model) in enumerate(pairs):
        ra, rb = get(base), get(model)
        if not ra or not rb:
            return None
        keys = sorted(set(ra) & set(rb))
        if not keys:
            return None
        c = 1.0 / (len(pairs) * sum(ra[k][1] for k in keys))
        for k in keys:
            a.append(ra[k][0] * c)
            b.append(rb[k][0] * c)
            g.append(f"{p}:{k}" if per_seed else str(k))
            w.append(ra[k][1] * c)
    return paired_bootstrap(a, b, g, weights=w)


def _explain_nll(name: str):
    def get(m):
        r = m.get("explain", {}).get(name)
        return _rows(r["on"]) if r else None
    return get


def _explain_gen(name: str, key: str):
    def get(m):
        r = m.get("explain", {}).get(name)
        if not r or not any(key in x for x in r["gen"]["rows"]):
            return None
        return {x["i"]: (x[key], 1.0) for x in r["gen"]["rows"] if key in x}
    return get


def explain_table(run_dirs: List[str], baseline: Optional[str] = None) -> str:
    """The explanation benchmark (T56): how well each model writes the
    explanation / conclusion, vs the baseline (* = paired 95 % CI excludes 0)."""
    runs = [(n, m) for n, m, _ in map(_load_both, run_dirs) if "explain" in m]
    if not runs:
        return ""
    base_name = baseline or next((n for n, _ in sorted(runs, key=lambda r: r[0])[::-1]
                                  if "baseline" in n), runs[0][0])
    base = next((m for n, m in runs if n == base_name), None)
    lines = []
    for name, m in runs:
        for s in EXPLAIN_SETS:
            r = m["explain"].get(s)
            if not r:
                continue
            b = None if (base is None or name == base_name) else base.get("explain", {}).get(s)
            ppl = f"{r['on']['ppl']:.3f}"
            if b:
                ci = _nll_ci(b["on"], r["on"], {})
                if ci:
                    ppl += f" ({100 * (math.exp(ci['diff']) - 1):+.1f}%{_star(ci)})"
            cells = [f"**{name}** (baseline)" if name == base_name else name,
                     EXPLAIN_TITLE[s], ppl,
                     f"{r['off']['ppl']:.3f}" if "off" in r else "-"]
            for mean_key, row_key in (("rouge_l", "rouge_l"), ("token_f1", "token_f1"),
                                      ("named_right", "named")):
                g = r["gen"]
                if mean_key not in g:
                    cells.append("-")
                    continue
                c = f"{100 * g[mean_key]:.1f}"
                if b:
                    ci = _pool([(base, m)], _explain_gen(s, row_key), False)
                    if ci:
                        c += f" ({100 * ci['diff']:+.1f}{_star(ci)})"
                cells.append(c)
            lines.append("| " + " | ".join(cells) + " |")
    head = ["Model", "Task", "Reference ppl (lower = better)", "ppl with L2 off",
            "ROUGE-L", "F1", "Names the right answer %"]
    return "\n".join(
        ["### Can the model explain? Written answers (T56)", "",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + lines
        + ["", "The model writes the answer itself. MedMCQA explanation: the question "
           "and its four options -> the right option and the expert's explanation "
           "(never trained on). PubMedQA conclusion: question + abstract -> the "
           "authors' conclusion (1,000 expert-labelled questions, never trained on). "
           "Reference ppl = how well the model predicts the reference text (lower = "
           "better; (+x%) vs the baseline). ROUGE-L / F1 = word overlap of its own "
           "greedy answer with the reference (%). Names the right answer = its own "
           "explanation names the right option before any other (%). * = paired 95% "
           "CI excludes 0."])


def gain_report(baselines: List[str], models: List[Tuple[str, List[str]]]) -> str:
    """How much better is each model than the baseline, over all seeds?

    ``baselines`` = the baseline run of every seed; ``models`` = (name, the
    model's runs in the same seed order).  Per test: % right of both (mean
    over seeds), the gain in points with a 95 % CI over seeds and questions,
    the gain relative to the baseline (+x %), the share of the baseline's
    wrong answers the model gets right, and the gain of each seed."""
    def load(d):
        with open(os.path.join(d, "mcq.json"), encoding="utf-8") as f:
            return json.load(f)

    base = [load(d) for d in baselines]
    seeds = [m.get("seed", "?") for m in base]
    lines = []
    for name, dirs in models:
        if len(dirs) != len(baselines):
            raise ValueError(f"{name}: {len(dirs)} runs for {len(baselines)} baselines")
        runs = [load(d) for d in dirs]
        for d, m, s in zip(dirs, runs, seeds):
            if m.get("seed", "?") != s:
                raise ValueError(f"{d} has seed {m.get('seed')}, its baseline {s}: "
                                 "give the runs in the same seed order")
        pairs = list(zip(base, runs))
        names = [n for n in MCQ_DATASETS if all(n in b and n in r for b, r in pairs)]
        for label, tests in [(MCQ_TITLE[n], [n]) for n in names] + [("Average", names)]:
            if not tests:
                continue
            acc_b = sum(sum(b[n]["accuracy"] for n in tests) / len(tests) for b in base) / len(base)
            acc_m = sum(sum(r[n]["accuracy"] for n in tests) / len(tests) for r in runs) / len(runs)
            ci = _pooled_acc(pairs, tests)
            per = []
            for p in pairs:
                c = _pooled_acc([p], tests)
                per.append(f"{100 * c['diff']:+.1f}{_star(c)}" if c else "-")
            gain = ci["diff"] if ci else acc_m - acc_b
            cells = [name, label, f"{100 * acc_b:.1f}", f"{100 * acc_m:.1f}",
                     (f"{100 * gain:+.2f} [{100 * ci['lo']:+.2f}, {100 * ci['hi']:+.2f}]{_star(ci)}"
                      if ci else f"{100 * gain:+.2f}"),
                     f"{100 * gain / max(acc_b, 1e-12):+.1f} %",
                     f"{100 * gain / max(1 - acc_b, 1e-12):+.1f} %",
                     " / ".join(per)]
            lines.append("| " + " | ".join(cells) + " |")
        nl = _pooled_nll(pairs)
        if nl:
            pb = sum(b["medquad_ppl"]["on"]["ppl"] for b in base) / len(base)
            pm = sum(r["medquad_ppl"]["on"]["ppl"] for r in runs) / len(runs)
            per = []
            for p in pairs:
                c = _pooled_nll([p])
                per.append(f"{100 * (math.exp(c['diff']) - 1):+.1f} %{_star(c)}" if c else "-")
            lines.append("| " + " | ".join([
                name, "MedQuAD ppl (lower = better)", f"{pb:.3f}", f"{pm:.3f}",
                f"{100 * (math.exp(nl['diff']) - 1):+.2f} % [{100 * (math.exp(nl['lo']) - 1):+.2f}, "
                f"{100 * (math.exp(nl['hi']) - 1):+.2f}]{_star(nl)}",
                "-", "-", " / ".join(per)]) + " |")
        # the explanation benchmark (T56): the same rows in every seed
        for s in EXPLAIN_SETS:
            get = _explain_nll(s)
            if not all(get(x) for p in pairs for x in p):
                continue
            ci = _pool(pairs, get, False)
            per = [(lambda c: f"{100 * (math.exp(c['diff']) - 1):+.1f} %{_star(c)}" if c else "-")(
                _pool([p], get, False)) for p in pairs]
            pb = sum(b["explain"][s]["on"]["ppl"] for b in base) / len(base)
            pm = sum(r["explain"][s]["on"]["ppl"] for r in runs) / len(runs)
            lines.append("| " + " | ".join([
                name, f"{EXPLAIN_TITLE[s]} ppl (lower = better)", f"{pb:.3f}", f"{pm:.3f}",
                f"{100 * (math.exp(ci['diff']) - 1):+.2f} % [{100 * (math.exp(ci['lo']) - 1):+.2f}, "
                f"{100 * (math.exp(ci['hi']) - 1):+.2f}]{_star(ci)}", "-", "-",
                " / ".join(per)]) + " |")
            for key, row_key, label in (("named_right", "named", "names the right answer"),
                                        ("rouge_l", "rouge_l", "ROUGE-L")):
                gget = _explain_gen(s, row_key)
                if not all(gget(x) for p in pairs for x in p):
                    continue
                ci = _pool(pairs, gget, False)
                vb = sum(b["explain"][s]["gen"][key] for b in base) / len(base)
                vm = sum(r["explain"][s]["gen"][key] for r in runs) / len(runs)
                per = [(lambda c: f"{100 * c['diff']:+.1f}{_star(c)}" if c else "-")(
                    _pool([p], gget, False)) for p in pairs]
                lines.append("| " + " | ".join([
                    name, f"{EXPLAIN_TITLE[s]}: {label}", f"{100 * vb:.1f}", f"{100 * vm:.1f}",
                    f"{100 * ci['diff']:+.2f} [{100 * ci['lo']:+.2f}, {100 * ci['hi']:+.2f}]{_star(ci)}",
                    f"{100 * ci['diff'] / max(vb, 1e-12):+.1f} %",
                    f"{100 * ci['diff'] / max(1 - vb, 1e-12):+.1f} %" if key == "named_right" else "-",
                    " / ".join(per)]) + " |")
    head = ["Model", "Test", "Baseline % right", "Model % right",
            "Gain, points [95% CI]", "Relative gain", "Baseline's errors fixed",
            "Gain per seed (" + " / ".join(str(s) for s in seeds) + ")"]
    return "\n".join(
        [f"### How much better than the baseline? (seeds {', '.join(str(s) for s in seeds)} together)",
         "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + lines
        + ["", "% right and perplexity = mean over the seeds. Gain = model minus "
           "baseline in points of % right, mean over the seeds, with a paired 95% "
           "bootstrap CI (each question resampled in every seed at once); * = the "
           "CI excludes 0 (a real gain, not noise). Relative gain = gain / the "
           "baseline's % right (+5 % = five percent more right answers). "
           "Baseline's errors fixed = gain / the baseline's % wrong. Perplexity: "
           "change of the per-token NLL as a % change of perplexity, CI over the "
           "benchmark questions of every seed. Explanation rows (T56): the model "
           "writes the explanation itself; ppl of the reference explanation, and "
           "the % of its own explanations that name the right answer / ROUGE-L "
           "with the reference. Chance: MedMCQA and MedQA 25 %, PubMedQA 33 %."])


GAIN_DOC = """How much better than the baseline, all seeds together.

  python -m ekdant.gen_eval gain --baseline runs/v22_baseline runs/v22s23_baseline \\
      --model v22 runs/v22_end runs/v22s23_end \\
      --model v3b runs/v3b_end runs/v3b_s23_end --markdown gain.md

Give every model's runs in the same seed order as the baselines (checked).
Reads the mcq.json files only (score the runs with `mcq` first).  To compare
two L2 models, pass one of them as --baseline."""


def gain_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ekdant.gen_eval gain", description=GAIN_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="the baseline run of every seed")
    ap.add_argument("--model", nargs="+", action="append", required=True,
                    metavar=("NAME", "RUN"),
                    help="a name, then its runs in the same seed order (repeatable)")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)
    models = []
    for m in args.model:
        if len(m) < 2:
            ap.error("--model needs a name and at least one run")
        models.append((m[0], m[1:]))
    table = gain_report(args.baseline, models)
    print("\n" + table + "\n")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"written to {args.markdown}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "gain":
        gain_main(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == "mcq":
        mcq_main(sys.argv[2:])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
    else:
        print("usage: python -m ekdant.gen_eval compare <run> <run> [...]\n"
              "       python -m ekdant.gen_eval mcq <run> [...] [--zero-shot]\n"
              "       python -m ekdant.gen_eval gain --baseline <runs> --model NAME <runs> [...]")
        raise SystemExit(2)
