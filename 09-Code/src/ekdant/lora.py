"""LoRA for the v2 backbone (T41).

SmolLM2 is frozen; every attention and MLP projection gets a trainable
low-rank update ``W x + (alpha / r) * B A x`` (Hu et al., 2021).  ``B`` starts
at zero, so a fresh LoRA model is exactly the pretrained one.  Embeddings,
norms and the tied head stay frozen.  L2 is not touched: it stays fully
trainable.  Written here (no ``peft``) so it runs on the pod's torch 2.2.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.lora_a = nn.Parameter(base.weight.new_empty(r, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


def apply_lora(lm: nn.Module, r: int, alpha: float,
               targets: Iterable[str] = TARGETS) -> int:
    """Freeze ``lm`` and wrap every ``nn.Linear`` named in ``targets``;
    returns how many were wrapped."""
    for p in lm.parameters():
        p.requires_grad_(False)
    targets = set(targets)
    hits = [(parent, name, child) for parent in lm.modules()
            for name, child in parent.named_children()
            if name in targets and isinstance(child, nn.Linear)]
    for parent, name, child in hits:
        setattr(parent, name, LoRALinear(child, r, alpha))
    if not hits:
        raise ValueError(f"no LoRA target found among {sorted(targets)}")
    return len(hits)
