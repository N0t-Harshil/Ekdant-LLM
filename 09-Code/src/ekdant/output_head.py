"""Output head - tied vocab projection.

No separate parameters: the projection reuses the embedding matrix, so the
LM head adds zero parameters to the total.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHead(nn.Module):
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        if embed.weight.shape[1] != embed.embedding_dim:
            raise ValueError("embedding shape mismatch")
        self.weight = embed.weight
        self.out_features, self.in_features = embed.weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    @property
    def num_parameters(self) -> int:
        return 0
