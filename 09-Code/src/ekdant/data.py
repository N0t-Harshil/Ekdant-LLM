"""Medical MCQ data pipeline: MedMCQA then MedQA.

Both datasets are reformatted into one prompt template and trained as a
causal LM whose next token after ``Answer:`` is the correct letter.  The
answer-token index is tracked exactly, so evaluation reads the letter
logits at one position instead of scoring four continuations.

MedMCQA  openlifescienceai/medmcqa     182,822 train / 4,183 validation
          (the public test split ships cop=-1 — unlabelled, never eval on it)
MedQA    GBaker/MedQA-USMLE-4-options   10,178 train /  1,273 test
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

LETTERS = ("A", "B", "C", "D")

# Datasets are downloaded once, then read from the local cache. Hub etag
# checks on every load are a network round-trip per split — with bad
# connectivity the request can stall indefinitely, which looks like
# training hanging silently right before an eval (observed: SynSent to a
# dead route). Default to offline reads; load_examples flips online for
# exactly one attempt when the cache is empty (first run). Override with
# HF_HUB_OFFLINE=0 ... to force online.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _hub_offline(off: bool) -> None:
    """Toggle hub offline mode after datasets/hub are already imported
    (their constants are read from the environment once, at import)."""
    val = "1" if off else "0"
    os.environ["HF_HUB_OFFLINE"] = val
    os.environ["HF_DATASETS_OFFLINE"] = val
    for mod_name, attr in (("huggingface_hub.constants", "HF_HUB_OFFLINE"),
                           ("datasets.config", "HF_DATASETS_OFFLINE")):
        try:
            setattr(__import__(mod_name, fromlist=[attr]), attr, off)
        except Exception:
            pass


def format_prompt(question: str, options: Sequence[str]) -> str:
    q = " ".join(str(question).split())
    lines = [q]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {' '.join(str(opt).split())}")
    return "\n".join(lines) + "\nAnswer:"


def normalise_options(raw) -> List[str]:
    if isinstance(raw, dict):
        keys = sorted(k for k in raw.keys() if k in LETTERS)
        return [str(raw[k]) for k in keys]
    if isinstance(raw, (list, tuple)):
        return [str(o) for o in raw][:4]
    raise TypeError(f"unsupported options container {type(raw)}")


def letter_from_answer(answer, options: Sequence[str]) -> Optional[str]:
    """Map a raw answer field to A-D, or None if it cannot be resolved.

    Never fabricates a default: MedMCQA's public test split ships cop=-1
    for every row, and silently mapping that to "A" made the eval metric
    degenerate into "how often does the model predict A" (bug T18).
    """
    if isinstance(answer, str) and answer.strip().upper() in LETTERS:
        return answer.strip().upper()
    if isinstance(answer, bool):
        return None
    if isinstance(answer, Integral):  # int, numpy integer
        n = int(answer)
        return LETTERS[n] if 0 <= n < len(LETTERS) else None
    if isinstance(answer, Real):  # parquet/pandas floats (2.0), reject 1.5
        if not float(answer).is_integer():
            return None
        n = int(answer)
        return LETTERS[n] if 0 <= n < len(LETTERS) else None
    if isinstance(answer, str) and answer.strip().isdigit():
        n = int(answer.strip())
        return LETTERS[n] if 0 <= n < len(LETTERS) else None
    text = str(answer).strip()
    for i, opt in enumerate(options):
        if text == str(opt).strip():
            return LETTERS[i]
    return None


@dataclass
class MCQExample:
    question: str
    options: List[str]
    answer: str


def _normalise_row(row: Dict, source: str) -> Optional[MCQExample]:
    if "options" in row:
        # MedQA (GBaker/MedQA-USMLE-4-options): options dict + answer_idx
        try:
            opts = normalise_options(row["options"])
        except TypeError:
            return None
        ans_raw = row.get("answer_idx", row.get("answer"))
    elif all(k in row for k in ("opa", "opb", "opc", "opd")):
        # MedMCQA (openlifescienceai/medmcqa): flat opa..opd, cop = 0..3
        opts = [str(row[k]) for k in ("opa", "opb", "opc", "opd")]
        ans_raw = row.get("cop")
    else:
        return None
    if len(opts) != 4 or any(not o.strip() for o in opts):
        return None
    if ans_raw is None:
        return None
    letter = letter_from_answer(ans_raw, opts)
    if letter is None:
        return None
    q = str(row.get("question", "")).strip()
    if len(q) < 12:
        return None
    return MCQExample(question=q, options=opts, answer=letter)


def load_examples(name: str, split: str, limit: Optional[int] = None,
                  seed: int = 17) -> List[MCQExample]:
    from datasets import load_dataset

    if name == "medmcqa":
        pkg = "openlifescienceai/medmcqa"
    elif name == "medqa":
        pkg = "GBaker/MedQA-USMLE-4-options"
    else:
        raise ValueError(f"unknown dataset {name!r}")

    # Offline-first (module header): cached reads never touch the network.
    # First run (empty cache) flips online for one attempt, then returns
    # to offline so later per-epoch evals stay local.
    try:
        ds = load_dataset(pkg, split=split)
    except Exception:
        _hub_offline(False)
        try:
            ds = load_dataset(pkg, split=split)
        finally:
            _hub_offline(True)

    rows = [_normalise_row(r, name) for r in ds]
    out = [r for r in rows if r is not None]
    if not out:
        raise ValueError(
            f"{name}/{split}: no usable labelled rows. MedMCQA's public "
            f"test split ships cop=-1 for every row (no released labels) — "
            f"evaluate on split='validation' instead."
        )
    if limit is not None and limit < len(out):
        rng = random.Random(seed)
        out = rng.sample(out, limit)
    return out


class Tokenizer:
    """Thin wrapper so the rest of the pipeline never imports `tokenizers`."""

    def __init__(self, tok):
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.bos_id = tok.token_to_id("<bos>") or 1
        self.eos_id = tok.token_to_id("<eos>") or 2
        self.pad_id = tok.token_to_id("<pad>") or 0

    @classmethod
    def train(cls, texts: Sequence[str], vocab_size: int, path: str,
              min_freq: int = 2) -> "Tokenizer":
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFD, Sequence as NormSeq, Lowercase
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="<unk>"))
        tok.normalizer = NormSeq([NFD(), Lowercase()])
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_freq,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tok.save(path)
        return cls(tok)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        from tokenizers import Tokenizer as HFTokenizer
        return cls(HFTokenizer.from_file(path))

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


class MCQDataset(Dataset):
    """Tokenises each example once and caches the answer-token index."""

    def __init__(self, examples: List[MCQExample], tokenizer: Tokenizer,
                 max_len: int = 1024):
        self.max_len = max_len
        self.rows: List[Dict] = []
        for ex in examples:
            prefix = format_prompt(ex.question, ex.options)
            ids_p = tokenizer.encode(prefix, add_eos=False)
            ids_a = tokenizer.encode(f" {ex.answer}", add_eos=False)
            if not ids_a:
                continue
            ids = ids_p + ids_a + [tokenizer.eos_id]
            if len(ids) > max_len:
                continue
            self.rows.append({
                "input_ids": ids,
                "answer_pos": len(ids_p),
                "answer_id": ids_a[0],
                "letter": ex.answer,
                "letter_idx": LETTERS.index(ex.answer),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        return self.rows[i]


def collate(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    pos = torch.zeros(len(batch), dtype=torch.long)
    tgt = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        mask[i, :n] = 1
        pos[i] = b["answer_pos"]
        tgt[i] = b["letter_idx"]  # index into LETTERS (0..3), not a vocab id
    labels = ids.clone()
    labels[~mask.bool()] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels,
            "answer_pos": pos, "answer_target": tgt}


@torch.no_grad()
def answer_accuracy(model, batch: Dict, n_options: int = 4) -> float:
    logits = model(batch["input_ids"], return_aux=False).logits
    # logits[t] predicts token t+1, so the answer at index p is scored at p-1
    gathered = logits[torch.arange(len(logits)), batch["answer_pos"] - 1]
    letter_ids = LETTER_IDS[:n_options]
    pred = gathered[:, letter_ids].argmax(-1)
    return (pred == batch["answer_target"]).float().mean().item()


LETTER_IDS: List[int] = []


def bind_letter_ids(tokenizer: Tokenizer) -> None:
    """Resolve the id of each answer letter under the trained vocabulary.

    A Whitespace pre-tokenizer yields the bare letter, but other pre-tokenizers
    may keep the leading space, so try every plausible surface form and fall
    back to the first sub-token of an encode rather than to the pad id.
    """
    LETTER_IDS.clear()
    for letter in LETTERS:
        tid = None
        for form in (letter, f" {letter}", f"{letter}", f"\n{letter}"):
            tid = tokenizer.tok.token_to_id(form)
            if tid is not None:
                break
        if tid is None:
            ids = tokenizer.tok.encode(f" {letter}", add_special_tokens=False).ids
            tid = ids[0] if ids else tokenizer.pad_id
        LETTER_IDS.append(tid)
