"""Layer T - the dense windowed Transformer.

Pure-PyTorch windowed GQA + SwiGLU blocks.  No MoE, no flash-attn: the window
mask is hand-built and attention goes through torch.nn.functional
scaled_dot_product_attention.

Spec: 04-Model-DeepDive/26b_Dense_Transformer_for_T.md
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).type_as(x) * self.weight


def rope_cache(dim: int, length: int, base: float = 1e5,
               device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(length, device=device, dtype=dtype)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype).view(1, 1, x.shape[2], -1)
    sin = sin.to(dtype=x.dtype).view(1, 1, x.shape[2], -1)
    return x * cos + rotate_half(x) * sin


def window_mask(length: int, window: int, device=None) -> torch.Tensor:
    i = torch.arange(length, device=device)
    return ((i[None, :] <= i[:, None])
            & ((i[:, None] - i[None, :]) < window))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, nkv, T, hd = x.shape
    return (x[:, :, None]
            .expand(B, nkv, n_rep, T, hd)
            .reshape(B, nkv * n_rep, T, hd))


class WindowedGQA(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, window: int):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} not divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.window = window
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.n_heads // self.n_kv_heads)
        v = repeat_kv(v, self.n_heads // self.n_kv_heads)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=self.scale, dropout_p=0.0
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 window: int, ffn_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = WindowedGQA(dim, n_heads, n_kv_heads, window)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin, mask) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    """N identical windowed blocks.  KV is bounded by the window."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        window: int = 512,
        ffn_hidden: Optional[int] = None,
        rope_base: float = 1e5,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.window = window
        self.rope_base = rope_base
        ffn_hidden = ffn_hidden or 2 * dim
        if dim % n_heads:
            raise ValueError(f"dim {dim} not divisible by {n_heads} heads")
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, window, ffn_hidden)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(dim)
        self._reset_output_projections()

    def _reset_output_projections(self) -> None:
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.o_proj.weight)
            nn.init.zeros_(blk.ffn.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        cos, sin = rope_cache(
            self.head_dim, T, self.rope_base, device=x.device,
        )
        mask = window_mask(T, self.window, device=x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.norm_out(x)
