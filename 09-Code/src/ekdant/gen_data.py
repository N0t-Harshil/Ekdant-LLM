"""Generative medical QA data for v2 (spec 07-Management/30 §4).

    phase 1   PubMedQA pqa_artificial: question + abstract -> conclusion
    phase 2   MedQuAD fine-tune half (question -> answer)
    bench     MedQuAD benchmark half, split by unique question

The loss covers answer tokens only (prompt labels are -100).  Answers are
cut at ``max_answer`` tokens (no EOS when cut); a too-long context is
trimmed from its end so the row fits ``max_len``.
"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ekdant.backbone import _offline_first

PUBMEDQA = ("qiaojin/PubMedQA", "pqa_artificial")
MEDQUAD = "keivalya/MedQuad-MedicalQnADataset"


@dataclass
class GenExample:
    question: str
    answer: str
    context: str = ""
    qtype: str = ""


def format_gen_prompt(ex: GenExample) -> str:
    if ex.context:
        return f"Question: {ex.question}\nContext: {ex.context}\nAnswer:"
    return f"Question: {ex.question}\nAnswer:"


def _rows_to_examples(rows: Iterable[Dict],
                      convert: Callable[[Dict], GenExample]) -> List[GenExample]:
    """Convert rows, dropping any with an empty question or answer (an empty
    answer would score zero tokens and break perplexity)."""
    out = []
    for r in rows:
        ex = convert(r)
        if ex.question.strip() and ex.answer.strip():
            out.append(GenExample(ex.question.strip(), ex.answer.strip(),
                                  ex.context.strip(), ex.qtype))
    return out


def _subsample(ex: List[GenExample], limit: Optional[int], seed: int):
    if limit is not None and limit < len(ex):
        return random.Random(seed).sample(ex, limit)
    return ex


def load_pubmedqa(limit: Optional[int] = None, seed: int = 17) -> List[GenExample]:
    from datasets import load_dataset

    ds = _offline_first(lambda: load_dataset(*PUBMEDQA, split="train"))
    ex = _rows_to_examples(ds, lambda r: GenExample(
        r["question"], r["long_answer"] or "",
        context=" ".join(r["context"]["contexts"])))
    return _subsample(ex, limit, seed)


def load_medquad(limit: Optional[int] = None, seed: int = 17) -> List[GenExample]:
    from datasets import load_dataset

    ds = _offline_first(lambda: load_dataset(MEDQUAD, split="train"))
    ex = _rows_to_examples(ds, lambda r: GenExample(
        r["Question"] or "", r["Answer"] or "", qtype=r["qtype"] or ""))
    return _subsample(ex, limit, seed)


class GenDataset(Dataset):
    """Tokenizes each row once.  Item: ``input_ids``, ``labels`` (-100 on the
    prompt), ``idx`` (row index into ``examples``).  Rows that cannot fit
    ``max_len`` even with the context removed are dropped (``dropped``)."""

    def __init__(self, examples: List[GenExample], tokenizer,
                 max_len: int = 1024, max_answer: int = 512):
        self.examples = examples
        # ids as array('i'): 209k PubMedQA rows as Python int lists would
        # take ~3.4 GB per process (three arms run in parallel)
        self.rows: List[Tuple[array, int, int]] = []  # ids, n_prompt, idx
        self.dropped = 0
        enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        for i, ex in enumerate(examples):
            ans = enc(" " + ex.answer)
            ans = ans[:max_answer] if len(ans) > max_answer \
                else ans + [tokenizer.eos_token_id]
            if ex.context:
                # Pieces keep the leading space on the context, so the split
                # tokenization matches tokenizing the whole prompt.
                pre = enc(f"Question: {ex.question}\nContext:")
                ctx = enc(" " + ex.context)
                suf = enc("\nAnswer:")
                keep = max_len - len(pre) - len(suf) - len(ans)
                if keep < 0:
                    self.dropped += 1
                    continue
                prompt = pre + ctx[:keep] + suf
            else:
                prompt = enc(format_gen_prompt(ex))
                if len(prompt) + len(ans) > max_len:
                    self.dropped += 1
                    continue
            self.rows.append((array("i", prompt + ans), len(prompt), i))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        ids, n_prompt, idx = self.rows[i]
        ids_t = torch.frombuffer(ids, dtype=torch.int32).long()
        labels = ids_t.clone()
        labels[:n_prompt] = -100
        return {"input_ids": ids_t, "labels": labels, "idx": idx}

    def prompt_ids(self, i: int) -> List[int]:
        """The (trimmed) prompt of row ``i`` - what generation starts from."""
        ids, n_prompt, _ = self.rows[i]
        return ids[:n_prompt].tolist()

    def example(self, i: int) -> GenExample:
        return self.examples[self.rows[i][2]]


def gen_collate(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad to the batch maximum; pad labels are -100."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for r, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[r, :L] = b["input_ids"]
        labels[r, :L] = b["labels"]
        mask[r, :L] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels,
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long)}


def question_key(q: str) -> str:
    return q.strip().lower()


def split_by_question(examples: List[GenExample], seed: int = 17,
                      bench_frac: float = 0.5, val_frac: float = 0.1
                      ) -> Tuple[List[GenExample], List[GenExample],
                                 List[GenExample]]:
    """(fine-tune, validation, benchmark).  One seeded permutation over the
    sorted unique question keys, so every row sharing a question lands in
    the same set and all arms get identical sets."""
    keys = sorted({question_key(e.question) for e in examples})
    perm = torch.randperm(len(keys),
                          generator=torch.Generator().manual_seed(seed)).tolist()
    n_bench = round(len(keys) * bench_frac)
    bench_k = {keys[j] for j in perm[:n_bench]}
    ft_order = perm[n_bench:]
    val_k = {keys[j] for j in ft_order[:round(len(ft_order) * val_frac)]}
    ft, val, bench = [], [], []
    for e in examples:
        k = question_key(e.question)
        (bench if k in bench_k else val if k in val_k else ft).append(e)
    return ft, val, bench


def holdout(examples: List[GenExample], n: int = 2000, seed: int = 17
            ) -> Tuple[List[GenExample], List[GenExample]]:
    """(train, held): ``n`` seeded rows held out, never trained on."""
    perm = torch.randperm(len(examples),
                          generator=torch.Generator().manual_seed(seed)).tolist()
    held = set(perm[:n])
    return ([e for i, e in enumerate(examples) if i not in held],
            [examples[i] for i in sorted(held)])
