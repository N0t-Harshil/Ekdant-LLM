"""Pretrained backbone loading for v2 (spec 07-Management/30 §3).

The backbone is a Hugging Face causal LM (default SmolLM2-135M).  Like the
datasets, weights are read from the local cache first; an empty cache goes
online for exactly one attempt (same pattern as ``data.load_examples``).
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar

from ekdant.data import _hub_offline

T = TypeVar("T")


def _offline_first(load: Callable[[], T]) -> T:
    try:
        return load()
    except Exception:
        _hub_offline(False)
        try:
            return load()
        finally:
            _hub_offline(True)


def load_complete_tokenizer(load: Callable[[], T], min_size: int) -> T:
    """Offline-first tokenizer load that refuses an incomplete tokenizer
    (T65).  With only ``config.json`` cached, transformers 5 builds an empty
    tokenizer offline (5 tokens, every word ``[UNK]``) instead of raising, so
    ``_offline_first`` never went online.  Fewer than ``min_size`` tokens:
    one online attempt; still too small: an error, never a silent result."""
    tok = _offline_first(load)
    if len(tok) >= min_size:
        return tok
    _hub_offline(False)
    try:
        tok = load()
    finally:
        _hub_offline(True)
    if len(tok) < min_size:
        raise ValueError(f"tokenizer has {len(tok)} tokens (< {min_size}): its "
                         f"files are missing from the cache and could not be downloaded")
    return tok


def apply_hf_token(path: str) -> str:
    """Make a Hugging Face token available to downloads without ever
    logging or storing it.  ``HF_TOKEN`` in the environment wins; otherwise
    the first line of ``path`` (a git-ignored file) is exported as
    ``HF_TOKEN``.  Returns where it came from: ``env``, ``file`` or ``none``
    (fine - SmolLM2, PubMedQA and MedQuAD are public)."""
    if os.environ.get("HF_TOKEN"):
        return "env"
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            token = f.readline().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return "file"
    return "none"


def load_backbone(model_id: str):
    """``AutoModelForCausalLM`` in float32 (training runs bf16 autocast).
    transformers 5 loads the stored dtype (SmolLM2 ships bf16), which would
    clash with the float32 add-on - so cast explicitly."""
    from transformers import AutoModelForCausalLM

    return _offline_first(
        lambda: AutoModelForCausalLM.from_pretrained(model_id)).float()


def load_tokenizer(model_id: str):
    """The backbone's own tokenizer; ``pad_token`` falls back to EOS."""
    from transformers import AutoTokenizer

    # SmolLM2 has 49,152 tokens; an incomplete cache gives a handful (T65)
    tok = load_complete_tokenizer(lambda: AutoTokenizer.from_pretrained(model_id), 1000)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_llama(vocab: int = 64, dim: int = 32, layers: int = 2):
    """Random tiny Llama with tied embeddings - a test stand-in for SmolLM2
    (same family, same interfaces), so tests never download weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=vocab, hidden_size=dim, intermediate_size=2 * dim,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=True, bos_token_id=1, eos_token_id=2,
                      pad_token_id=0)
    return LlamaForCausalLM(cfg)
