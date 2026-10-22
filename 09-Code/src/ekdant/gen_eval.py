"""v2 evaluation (spec 07-Management/30 §5).

    answer_nll        reference-answer NLL / perplexity (primary metric)
    by_type           perplexity per MedQuAD question type
    generate_greedy   batched greedy decoding with the KV cache
    rouge_l, token_f1 text overlap of generated vs reference answers
    paired_bootstrap  CI of the per-answer NLL difference between two arms,
                      resampling questions (rows sharing a question move
                      together)

CLI:  python -m ekdant.gen_eval compare runs/v2_baseline runs/v2_sandbox ...
"""

from __future__ import annotations

import json
import math
import os
import random
import string
import sys
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

DIAG_KEYS = ("anchor_energy", "trajectory_variance", "spread_ratio")


@torch.no_grad()
def answer_nll(model, loader, device, autocast) -> Dict:
    """Token-weighted mean NLL over answer tokens (labels != -100), plus
    per-row sums for the per-type breakdown and the bootstrap."""
    model.eval()
    row_nll: List[float] = []
    row_tok: List[int] = []
    row_idx: List[int] = []
    diag: Dict[str, float] = {}
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            out = model(ids, attention_mask=mask, labels=labels)
        logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
        tgt = labels[:, 1:]
        valid = tgt != -100
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
        with autocast():
            o = model(ids, attention_mask=mask, position_ids=pos, use_cache=True)
        past = o.aux["past_key_values"]
        nxt = o.logits[:, -1].argmax(-1)
        last_pos = pos[:, -1]
        gen = [[] for _ in chunk]
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
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
          f"{'|g|':>8s} {'pubmed ppl':>11s}")
    for d, r in runs:
        b, p = r["test"]["medquad_bench"], r["test"]["pubmedqa_held"]
        print(f"{os.path.basename(d.rstrip('/')):28s} {b['ppl']:10.3f} "
              f"{b.get('rouge_l', float('nan')):8.4f} "
              f"{b.get('token_f1', float('nan')):7.4f} "
              f"{r['test'].get('gate_magnitude', 0.0):8.4f} {p['ppl']:11.3f}")
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


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
    else:
        print("usage: python -m ekdant.gen_eval compare <run> <run> [...]")
        raise SystemExit(2)
