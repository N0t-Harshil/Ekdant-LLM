"""Medical MCQ data pipeline: MedMCQA then MedQA.

Both datasets are reformatted into one prompt template and trained as a
causal LM whose next token after ``Answer:`` is the correct letter.  The
answer-token index is tracked exactly, so evaluation reads the letter
logits at one position instead of scoring four continuations.

MedMCQA  openlifescienceai/medmcqa     182,822 train / 6,150 test
MedQA    GBaker/MedQA-USMLE-4-options   10,178 train /  1,273 test
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

LETTERS = ("A", "B", "C", "D")


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


def letter_from_answer(answer, options: Sequence[str]) -> str:
    if isinstance(answer, str) and answer.strip().upper() in LETTERS:
        return answer.strip().upper()
    if isinstance(answer, int):
        return LETTERS[answer]
    if isinstance(answer, str) and answer.strip().isdigit():
        return LETTERS[int(answer.strip())]
    text = str(answer).strip()
    for i, opt in enumerate(options):
        if text == str(opt).strip():
            return LETTERS[i]
    return "A"


@dataclass
class MCQExample:
    question: str
    options: List[str]
    answer: str


def _normalise_row(row: Dict, source: str) -> Optional[MCQExample]:
    try:
        opts = normalise_options(row["options"])
    except (KeyError, TypeError):
        return None
    if len(opts) != 4 or any(not o.strip() for o in opts):
        return None
    ans_raw = row.get("answer_idx", row.get("answer"))
    if ans_raw is None:
        return None
    letter = letter_from_answer(ans_raw, opts)
    q = str(row.get("question", "")).strip()
    if len(q) < 12:
        return None
    return MCQExample(question=q, options=opts, answer=letter)


def load_examples(name: str, split: str, limit: Optional[int] = None,
                  seed: int = 17) -> List[MCQExample]:
    from datasets import load_dataset

    if name == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa", split=split)
    elif name == "medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    else:
        raise ValueError(f"unknown dataset {name!r}")

    rows = [_normalise_row(r, name) for r in ds]
    out = [r for r in rows if r is not None]
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
        tgt[i] = b["answer_id"]
    labels = ids.clone()
    labels[~mask.bool()] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels,
            "answer_pos": pos, "answer_target": tgt}


@torch.no_grad()
def answer_accuracy(model, batch: Dict, n_options: int = 4) -> float:
    logits = model(batch["input_ids"], return_aux=False).logits
    gathered = logits[torch.arange(len(logits)), batch["answer_pos"]]
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
