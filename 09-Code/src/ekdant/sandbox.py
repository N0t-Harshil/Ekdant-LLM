"""Layer 2 - the sandbox.

Pure-PyTorch implementation of the thesis sandbox: diverse initialisation,
cross-read debate rounds, anchored gated descent, consensus readout, and the
dimensionless spread floor.  No agent frameworks, no transcripts.

Spec: 04-Model-DeepDive/26c_Internal_Sandbox_Build.md
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekdant.transformer_stack import RMSNorm

RHO_MIN_DEFAULT = 0.05


class SandboxOutput(NamedTuple):
    consensus: torch.Tensor
    trajectories: torch.Tensor
    aux: Dict[str, torch.Tensor]
    # per-token agreement of the K trajectories, in (0, 1]: 1 = they all
    # reached the same state (T50); detached - a signal, not a loss target
    confidence: Optional[torch.Tensor] = None


def masked_mean_bt(x: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over the trailing [B,T] dims, counting only real tokens.

    x [..., B, T]; mask [B, T] (1 = real token, 0 = padding) or None.
    Returns shape x.shape[:-2].  With mask=None this is a plain mean, so
    unpadded inputs give exactly the old numbers.
    """
    if mask is None:
        return x.mean(dim=(-2, -1))
    w = mask.to(x.dtype)
    return (x * w).sum(dim=(-2, -1)) / w.sum().clamp_min(1.0)


def mutual_orthogonal(K: int, n: int, device=None, dtype=None) -> torch.Tensor:
    """K mutually Frobenius-orthogonal n-by-n matrices, each of unit norm."""
    mats = []
    for _ in range(K):
        m = torch.randn(n, n, device=device, dtype=dtype)
        for prev in mats:
            coeff = (m * prev).sum() / (prev * prev).sum()
            m = m - coeff * prev
        mats.append(m / m.norm().clamp_min(1e-12))
    return torch.stack(mats, dim=0)


def spread_gains(K: int, lo: float = 0.8, hi: float = 1.25,
                 device=None, dtype=None) -> torch.Tensor:
    if K == 1:
        return torch.ones(1, device=device, dtype=dtype)
    return torch.logspace(math.log10(lo), math.log10(hi), K,
                          device=device, dtype=dtype)


class DiverseInit(nn.Module):
    """K distinct projections of hT: orthogonal weights, spread gains."""

    def __init__(self, dim: int, K: int, gain_lo: float = 0.8,
                 gain_hi: float = 1.25):
        super().__init__()
        self.K = K
        self.dim = dim
        w = mutual_orthogonal(K, dim) * math.sqrt(dim)
        g = spread_gains(K, gain_lo, gain_hi)
        self.weight = nn.Parameter(w * g.view(K, 1, 1))
        self.bias = nn.Parameter(0.02 * torch.randn(K, dim))

    def forward(self, hT: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("kij,btj->kbti", self.weight, hT)
        return out + self.bias.view(self.K, 1, 1, self.dim)


class CrossRead(nn.Module):
    """Each trajectory attends the other K-1, never itself."""

    def __init__(self, dim: int):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        K = traj.shape[0]
        if K == 1:
            return torch.zeros_like(traj)
        q, k, v = self.wq(traj), self.wk(traj), self.wv(traj)
        scores = torch.einsum("kbtd,jbtd->kjbt", q, k) * self.scale
        eye = torch.eye(K, dtype=torch.bool, device=traj.device)
        scores = scores.masked_fill(eye.view(K, K, 1, 1), float("-inf"))
        attn = scores.softmax(dim=1)
        return torch.einsum("kjbt,jbtd->kbtd", attn, v)


class ContextRead(nn.Module):
    """Each trajectory reads the earlier tokens of the text (T50): causal
    multi-head attention over the sequence, the same weights for all K
    trajectories, so each hypothesis looks at the question and context in
    its own way.  The output projection starts at zero: an untrained
    ContextRead changes nothing.  ``attn_mask`` [B, T] (1 = real token)
    keeps padding out; every token always sees itself."""

    def __init__(self, dim: int, heads: int = 0):
        super().__init__()
        heads = heads or max(1, dim // 64)
        if dim % heads:
            raise ValueError(f"sandbox_heads {heads} must divide dim {dim}")
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, traj: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        K, B, T, D = traj.shape
        H = self.heads
        x = self.norm(traj).reshape(K * B, T, D)
        q, k, v = self.qkv(x).view(K * B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=traj.device).tril()
            m = causal & attn_mask.bool().unsqueeze(1)                    # [B,T,T]
            m = m | torch.eye(T, dtype=torch.bool, device=traj.device)
            m = m.repeat(K, 1, 1).unsqueeze(1)                           # [K*B,1,T,T]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        o = o.transpose(1, 2).reshape(K, B, T, D)
        return traj + self.out(o)


class DebateRound(nn.Module):
    """Repair gate lets a trajectory decline the crowd.  With ``context``
    (T50) every trajectory first reads the earlier tokens."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, gate_bias: float = -2.0,
                 norm_input: bool = False, context: bool = False,
                 heads: int = 0):
        super().__init__()
        self.context = ContextRead(dim, heads) if context else None
        # Pre-norm (v2, T33): the round reads a fixed-size input, so weight
        # growth cannot compound across rounds. Off for v1 (unchanged).
        self.norm = RMSNorm(dim) if norm_input else nn.Identity()
        self.read = CrossRead(dim)
        hidden = int(dim * mlp_ratio)
        self.gate = nn.Linear(2 * dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )
        nn.init.orthogonal_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, traj: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.context is not None:
            traj = self.context(traj, attn_mask)
        x = self.norm(traj)
        cross = self.read(x)
        gamma = torch.sigmoid(self.gate(torch.cat([x, cross], dim=-1)))
        return (1.0 - gamma) * traj + gamma * self.mlp(cross)


class AnchoredDescent(nn.Module):
    """Gate-scaled preconditioned descent toward an external anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.anchor = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.orthogonal_(self.anchor.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def _dev_from_consensus(traj: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = traj.mean(dim=0, keepdim=True)
        return masked_mean_bt((traj - c).pow(2).sum(-1).sqrt(), mask)

    def forward(self, traj, hT, steps, eta, mask=None):
        """Returns (traj, a_t, per-token energy [K,B,T], retention [K])."""
        a_t = self.anchor(hT)
        target = a_t.unsqueeze(0).expand_as(traj)
        dev0 = self._dev_from_consensus(traj, mask)
        for _ in range(steps):
            g = torch.sigmoid(self.gate(torch.cat([traj, target], dim=-1)))
            traj = traj - eta * g * (traj - target)
        dev1 = self._dev_from_consensus(traj, mask)
        retention = dev1 / dev0.clamp_min(1e-12)
        energy = 0.5 * (traj - target).pow(2).sum(dim=-1)
        return traj, a_t, energy, retention


class Sandbox(nn.Module):
    """K trajectories: init -> debate x R -> anchored descent x S -> consensus."""

    def __init__(
        self,
        dim: int,
        K: int = 4,
        rounds: int = 2,
        steps: int = 8,
        eta: Optional[float] = None,
        rho_min: float = RHO_MIN_DEFAULT,
        readout: str = "mean",
        gain_lo: float = 0.8,
        gain_hi: float = 1.25,
        mlp_ratio: float = 2.0,
        norm_inputs: bool = False,
        context: bool = False,
        heads: int = 0,
        confidence: bool = False,
    ):
        super().__init__()
        if readout not in ("mean", "lowest_energy"):
            raise ValueError(f"readout must be mean|lowest_energy, got {readout}")
        if eta is None:
            eta = 0.5 / steps if steps > 0 else 0.0
        self.dim = dim
        self.K = K
        self.rounds = rounds
        self.steps = steps
        # times the whole debate stack is applied (T47 thinking-time test;
        # training always uses 1)
        self.debate_passes = 1
        self.eta = eta
        self.rho_min = rho_min
        self.readout = readout
        self.init_proj = DiverseInit(dim, K, gain_lo, gain_hi)
        self.debate = nn.ModuleList(
            [DebateRound(dim, mlp_ratio=mlp_ratio, norm_input=norm_inputs,
                         context=context, heads=heads)
             for _ in range(rounds)])
        self.descent = AnchoredDescent(dim)
        # Confidence-weighted readout (T50): a verifier scores each
        # trajectory per token and the consensus trusts the better ones.
        # Zero-init = equal weights = the plain mean.
        self.verifier = None
        if confidence:
            self.verifier = nn.Sequential(RMSNorm(dim), nn.Linear(dim, 1))
            nn.init.zeros_(self.verifier[1].weight)
            nn.init.zeros_(self.verifier[1].bias)

    def forward(self, hT: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> SandboxOutput:
        """hT [B,T,H]; mask [B,T] (1 = token the loss trains) only affects
        the diagnostics / spread floor, never the consensus; attn_mask
        [B,T] (1 = real token) keeps padding out of the context read."""
        traj = self.init_proj(hT)
        for _ in range(self.debate_passes):
            for layer in self.debate:
                traj = layer(traj, attn_mask)
        traj, a_t, energy, retention = self.descent(
            traj, hT, self.steps, self.eta, mask
        )
        aux = self._aux(traj, hT, a_t, energy, retention, mask)
        if self.verifier is not None:
            w = self.verifier(traj).squeeze(-1).float().softmax(dim=0)   # [K,B,T]
            consensus = (w.unsqueeze(-1).type_as(traj) * traj).sum(0)
            conf = self._confidence(traj, consensus, w)
            aux["confidence"] = masked_mean_bt(conf, mask)
            aux["verifier_top_weight"] = masked_mean_bt(w.detach().max(dim=0).values, mask)
            return SandboxOutput(consensus, traj, aux, conf)
        if self.readout == "lowest_energy":
            # Per-position pick (thesis: argmin_k ||T_k - a_t||).  Choosing
            # one trajectory from batch/sequence-averaged energy let future
            # tokens and other batch rows decide position t's output.
            idx = energy.argmin(dim=0)                       # [B,T]
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(1, *traj.shape[1:])
            consensus = traj.gather(0, idx).squeeze(0)
        else:
            consensus = traj.mean(dim=0)
        conf = self._confidence(traj, consensus)
        aux["confidence"] = masked_mean_bt(conf, mask)
        return SandboxOutput(consensus, traj, aux, conf)

    @staticmethod
    def _confidence(traj: torch.Tensor, consensus: torch.Tensor,
                    w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-token agreement exp(-d), d = (weighted) mean squared distance
        of the trajectories from the consensus relative to its size: 1 when
        all K agree, towards 0 when they disagree.  Detached (T50)."""
        with torch.no_grad():
            d = (traj - consensus.unsqueeze(0)).float().pow(2).sum(-1)    # [K,B,T]
            d = (w * d).sum(0) if w is not None else d.mean(0)
            rel = d / consensus.float().pow(2).sum(-1).clamp_min(1e-6)
            return torch.exp(-rel)

    def _aux(self, traj, hT, a_t, energy, retention,
             mask=None) -> Dict[str, torch.Tensor]:
        # Statistics over real tokens only: padded positions carry states
        # the loss never sees and would dilute rho in mixed-length batches.
        var = masked_mean_bt(traj.var(dim=0, unbiased=False).mean(-1), mask)
        rho = var / masked_mean_bt(hT.pow(2).mean(-1), mask).clamp_min(1e-12)
        return {
            "anchor_energy": masked_mean_bt(energy, mask).mean(),
            "trajectory_variance": var,
            "spread_ratio": rho,
            "per_trajectory_retention": retention,
            "anchor_distance": masked_mean_bt(
                (traj - a_t.unsqueeze(0)).pow(2).sum(-1).sqrt(), mask),
        }

    def spread_floor(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.rho_min - rho).pow(2)

    @torch.no_grad()
    def post_round_state(self, hT: torch.Tensor,
                         attn_mask: Optional[torch.Tensor] = None
                         ) -> List[torch.Tensor]:
        states = []
        traj = self.init_proj(hT)
        states.append(traj)
        for layer in self.debate:
            traj = layer(traj, attn_mask)
            states.append(traj)
        return states
