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

RHO_MIN_DEFAULT = 0.05


class SandboxOutput(NamedTuple):
    consensus: torch.Tensor
    trajectories: torch.Tensor
    aux: Dict[str, torch.Tensor]


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


class DebateRound(nn.Module):
    """Repair gate lets a trajectory decline the crowd."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, gate_bias: float = -2.0):
        super().__init__()
        self.read = CrossRead(dim)
        hidden = int(dim * mlp_ratio)
        self.gate = nn.Linear(2 * dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim)
        )
        nn.init.orthogonal_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        cross = self.read(traj)
        gamma = torch.sigmoid(self.gate(torch.cat([traj, cross], dim=-1)))
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
    def _dev_from_consensus(traj: torch.Tensor) -> torch.Tensor:
        c = traj.mean(dim=0, keepdim=True)
        return (traj - c).pow(2).sum(-1).sqrt().mean(dim=(1, 2))

    def forward(self, traj, hT, steps, eta):
        a_t = self.anchor(hT)
        target = a_t.unsqueeze(0).expand_as(traj)
        dev0 = self._dev_from_consensus(traj)
        for _ in range(steps):
            g = torch.sigmoid(self.gate(torch.cat([traj, target], dim=-1)))
            traj = traj - eta * g * (traj - target)
        dev1 = self._dev_from_consensus(traj)
        retention = dev1 / dev0.clamp_min(1e-12)
        energy = 0.5 * (traj - target).pow(2).sum(dim=-1)
        return traj, a_t, energy.mean(dim=(1, 2)), retention


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
        self.eta = eta
        self.rho_min = rho_min
        self.readout = readout
        self.init_proj = DiverseInit(dim, K, gain_lo, gain_hi)
        self.debate = nn.ModuleList([DebateRound(dim) for _ in range(rounds)])
        self.descent = AnchoredDescent(dim)

    def forward(self, hT: torch.Tensor) -> SandboxOutput:
        traj = self.init_proj(hT)
        for layer in self.debate:
            traj = layer(traj)
        traj, a_t, energies, retention = self.descent(
            traj, hT, self.steps, self.eta
        )
        aux = self._aux(traj, hT, a_t, energies, retention)
        if self.readout == "lowest_energy":
            rows = torch.arange(traj.shape[1], device=traj.device)
            consensus = traj[energies.argmin(dim=0), rows]
        else:
            consensus = traj.mean(dim=0)
        return SandboxOutput(consensus, traj, aux)

    def _aux(self, traj, hT, a_t, energies, retention) -> Dict[str, torch.Tensor]:
        var = traj.var(dim=0, unbiased=False).mean()
        rho = var / hT.pow(2).mean().clamp_min(1e-12)
        return {
            "anchor_energy": energies.mean(),
            "trajectory_variance": var,
            "spread_ratio": rho,
            "per_trajectory_retention": retention,
            "anchor_distance": (traj - a_t.unsqueeze(0))
            .pow(2).sum(-1).sqrt().mean(dim=(1, 2)),
        }

    def spread_floor(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.rho_min - rho).pow(2)

    @torch.no_grad()
    def post_round_state(self, hT: torch.Tensor) -> List[torch.Tensor]:
        states = []
        traj = self.init_proj(hT)
        states.append(traj)
        for layer in self.debate:
            traj = layer(traj)
            states.append(traj)
        return states
