"""Layer 2 - the sandbox.

Pure-PyTorch implementation of the thesis sandbox: diverse initialisation,
cross-read debate rounds, anchored gated descent, consensus readout, and the
dimensionless spread floor.  No agent frameworks, no transcripts.

Spec: 04-Model-DeepDive/26c_Internal_Sandbox_Build.md
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, NamedTuple, Optional, Tuple

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
    # [B,T,3] with the verifier (T51): agreement, top verifier weight, vote
    # support - the detached inputs of the learned confidence
    features: Optional[torch.Tensor] = None


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
    """K distinct projections of hT: orthogonal weights, spread gains.

    ``aligned`` (T51): trajectory k = hT + delta_k(hT), delta_k starting as
    ``nudge`` x a random matrix, no bias - every trajectory starts in the
    model's own representation (same meaning, a different angle) instead of
    a random projection the model would have to fight.  Stored as the
    delta, so weight decay pulls towards the model's state, not towards 0.
    ``nudge`` 0.6 gives a spread ratio near the floor (0.05) on SmolLM2."""

    def __init__(self, dim: int, K: int, gain_lo: float = 0.8,
                 gain_hi: float = 1.25, aligned: bool = False,
                 nudge: float = 0.6):
        super().__init__()
        self.K = K
        self.dim = dim
        self.aligned = aligned
        w = mutual_orthogonal(K, dim) * math.sqrt(dim)
        if aligned:
            self.weight = nn.Parameter(nudge * w)
            self.bias = nn.Parameter(torch.zeros(K, dim))
        else:
            g = spread_gains(K, gain_lo, gain_hi)
            self.weight = nn.Parameter(w * g.view(K, 1, 1))
            self.bias = nn.Parameter(0.02 * torch.randn(K, dim))

    def forward(self, hT: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("kij,btj->kbti", self.weight, hT)
        if self.aligned:
            out = out + hT.unsqueeze(0)
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


class TextMemory(NamedTuple):
    """What L2 reads (T51): the model's own states of the tokens so far,
    computed once per forward and shared by every trajectory and round."""
    keys: torch.Tensor        # [B, H, Tk, dh]
    values: torch.Tensor      # [B, H, Tk, dh] = the model's states, unchanged
    allowed: torch.Tensor     # [B, 1, Tq, Tk] bool: causal + no padding
    history: torch.Tensor     # [B, Tq, D] mean of the model's states so far


def read_mask(Tq: int, Tk: int, attn_mask: Optional[torch.Tensor],
              B: int, device) -> torch.Tensor:
    """[B,1,Tq,Tk]: query i (= key position Tk-Tq+i) sees keys j <= it that
    are real tokens; it always sees itself (no empty row)."""
    off = Tk - Tq
    qi = torch.arange(Tq, device=device).view(Tq, 1) + off
    kj = torch.arange(Tk, device=device).view(1, Tk)
    m = (kj <= qi).unsqueeze(0).expand(B, Tq, Tk)
    if attn_mask is not None:
        m = m & attn_mask[:, -Tk:].bool().unsqueeze(1)
    m = m | (kj == qi).unsqueeze(0)
    return m.unsqueeze(1)


class ContextRead(nn.Module):
    """Each trajectory reads the text through the model's own states
    (T50/T51).  Keys and values come from ``TextMemory`` (one projection of
    the model's states per forward, values unchanged), so L2 reads in the
    model's own meaning and costs one query projection per round; the K
    trajectories are folded into the query length, so the text is never
    copied K times.  ``alpha`` (per channel) starts at 0: an untrained read
    changes nothing.

    ``diverse`` (T53): each trajectory adds its own learned query offset, so
    the K trajectories ask the text different questions - without it they
    start almost equal (aligned) and read the same thing, and the debate
    collapses into agreement (v3_end: spread at the floor, vote 0.69 -> 0.83)."""

    def __init__(self, dim: int, heads: int, K: int = 1, diverse: bool = False):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.q_offset = (nn.Parameter(0.5 * torch.randn(K, dim))
                         if diverse and K > 1 else None)

    def forward(self, traj: torch.Tensor, mem: TextMemory) -> torch.Tensor:
        K, B, T, D = traj.shape
        H = self.heads
        q = self.q(self.norm(traj))                                  # [K,B,T,D]
        if self.q_offset is not None:
            q = q + self.q_offset.view(K, 1, 1, D).type_as(q)
        q = q.view(K, B, T, H, D // H).permute(1, 3, 0, 2, 4).reshape(B, H, K * T, D // H)
        m = mem.allowed.repeat(1, 1, K, 1)                           # [B,1,K*T,Tk]
        o = F.scaled_dot_product_attention(q, mem.keys.type_as(q),
                                           mem.values.type_as(q), attn_mask=m)
        o = o.view(B, H, K, T, D // H).permute(2, 0, 3, 1, 4).reshape(K, B, T, D)
        return traj + self.alpha * o


class DebateRound(nn.Module):
    """Repair gate lets a trajectory decline the crowd.  With ``context``
    (T50) every trajectory first reads the text."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, gate_bias: float = -2.0,
                 norm_input: bool = False, context: bool = False,
                 heads: int = 1, K: int = 1, diverse: bool = False):
        super().__init__()
        self.context = ContextRead(dim, heads, K, diverse) if context else None
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
                mem: Optional[TextMemory] = None) -> torch.Tensor:
        if self.context is not None and mem is not None:
            traj = self.context(traj, mem)
        x = self.norm(traj)
        cross = self.read(x)
        gamma = torch.sigmoid(self.gate(torch.cat([x, cross], dim=-1)))
        return (1.0 - gamma) * traj + gamma * self.mlp(cross)


class AnchoredDescent(nn.Module):
    """Gate-scaled preconditioned descent toward an external anchor.
    ``aligned`` (T51): the anchor is the model's own state plus a learned
    correction that starts at 0, so the descent pulls every trajectory back
    towards what the model means; before, it was a random rotation of it."""

    def __init__(self, dim: int, aligned: bool = False):
        super().__init__()
        self.aligned = aligned
        self.anchor = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)
        if aligned:
            nn.init.zeros_(self.anchor.weight)
        else:
            nn.init.orthogonal_(self.anchor.weight)
        nn.init.zeros_(self.gate.bias)

    def anchor_of(self, hT: torch.Tensor) -> torch.Tensor:
        a = self.anchor(hT)
        return hT + a if self.aligned else a

    @staticmethod
    def _dev_from_consensus(traj: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = traj.mean(dim=0, keepdim=True)
        return masked_mean_bt((traj - c).pow(2).sum(-1).sqrt(), mask)

    def forward(self, traj, hT, steps, eta, mask=None):
        """Returns (traj, a_t, per-token energy [K,B,T], retention [K])."""
        a_t = self.anchor_of(hT)
        target = a_t.unsqueeze(0).expand_as(traj)
        dev0 = self._dev_from_consensus(traj, mask)
        for _ in range(steps):
            g = torch.sigmoid(self.gate(torch.cat([traj, target], dim=-1)))
            traj = traj - eta * g * (traj - target)
        dev1 = self._dev_from_consensus(traj, mask)
        retention = dev1 / dev0.clamp_min(1e-12)
        energy = 0.5 * (traj - target).pow(2).sum(dim=-1)
        return traj, a_t, energy, retention


class Verifier(nn.Module):
    """Scores each trajectory per token (T51):
    (1) its own quality, (2) how well it fits what was read / generated so far
    (the history of the model's states), (3) the vote - how many of the other
    trajectories agree with it (odd K gives a clear majority).
    (1) and (2) start at zero; the vote starts on (weight 1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.own = nn.Linear(dim, 1)
        self.fit = nn.Linear(dim, dim, bias=False)
        self.vote = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.own.weight)
        nn.init.zeros_(self.own.bias)
        nn.init.zeros_(self.fit.weight)

    @staticmethod
    def support(traj: torch.Tensor) -> torch.Tensor:
        """[K,B,T]: mean over the other trajectories of exp(-distance^2 /
        size^2) - 1 = all others agree with this one."""
        K = traj.shape[0]
        if K == 1:
            return torch.ones(traj.shape[:-1], device=traj.device, dtype=traj.dtype)
        t = traj.float()
        size = t.mean(0).pow(2).sum(-1).clamp_min(1e-6)               # [B,T]
        sq = t.pow(2).sum(-1)                                         # [K,B,T]
        gram = torch.einsum("kbtd,jbtd->kjbt", t, t)                  # [K,K,B,T]
        d = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * gram).clamp_min(0) / size
        s = torch.exp(-d).sum(1) - 1.0                                # drop self
        return s / (K - 1)

    def forward(self, traj: torch.Tensor,
                history: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(weights [K,B,T] summing to 1 over K, support [K,B,T])."""
        x = self.norm(traj)
        s = self.own(x).squeeze(-1).float()
        if history is not None:
            h = history.unsqueeze(0).type_as(x)
            s = s + (self.fit(x) * h).sum(-1).float() / math.sqrt(x.shape[-1])
        sup = self.support(traj)
        s = s + self.vote.float() * sup
        return s.softmax(dim=0), sup.detach()


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
        aligned: bool = False,
        diverse_read: bool = False,
        train_passes: int = 1,
    ):
        super().__init__()
        if readout not in ("mean", "lowest_energy"):
            raise ValueError(f"readout must be mean|lowest_energy, got {readout}")
        if eta is None:
            eta = 0.5 / steps if steps > 0 else 0.0
        heads = heads or max(1, dim // 64)
        if context and dim % heads:
            raise ValueError(f"sandbox_heads {heads} must divide dim {dim}")
        self.dim = dim
        self.K = K
        self.rounds = rounds
        self.steps = steps
        # times the whole debate stack is applied (T47 thinking-time test).
        # T54 train_passes: in training each forward draws 1..train_passes
        # passes, so a second pass learns to refine the first instead of
        # breaking it; the model then answers with train_passes passes.
        if train_passes < 1:
            raise ValueError(f"train_passes must be >= 1, got {train_passes}")
        self.train_passes = train_passes
        self.debate_passes = train_passes
        self._pass_rng = random.Random(0)
        self.eta = eta
        self.rho_min = rho_min
        self.readout = readout
        self.heads = heads
        self.init_proj = DiverseInit(dim, K, gain_lo, gain_hi, aligned=aligned)
        self.debate = nn.ModuleList(
            [DebateRound(dim, mlp_ratio=mlp_ratio, norm_input=norm_inputs,
                         context=context, heads=heads, K=K, diverse=diverse_read)
             for _ in range(rounds)])
        self.descent = AnchoredDescent(dim, aligned=aligned)
        # The text L2 reads (T51): the model's states, normalised; keys are
        # one projection of them per forward, values are the states as is.
        self.mem_norm = RMSNorm(dim) if context else None
        self.mem_key = nn.Linear(dim, dim, bias=False) if context else None
        # Confidence-weighted readout (T50/T51): see Verifier.
        self.verifier = Verifier(dim) if confidence else None

    def memory(self, ctx: torch.Tensor, Tq: int,
               attn_mask: Optional[torch.Tensor]) -> TextMemory:
        """ctx [B,Tk,D]: the model's states of every token so far (the last
        Tq are the ones being processed now)."""
        B, Tk, D = ctx.shape
        H = self.heads
        n = self.mem_norm(ctx)
        k = self.mem_key(n).view(B, Tk, H, D // H).transpose(1, 2)
        v = n.view(B, Tk, H, D // H).transpose(1, 2)
        allowed = read_mask(Tq, Tk, attn_mask, B, ctx.device)
        w = (attn_mask[:, -Tk:].to(n.dtype) if attn_mask is not None
             else torch.ones(B, Tk, dtype=n.dtype, device=n.device)).unsqueeze(-1)
        hist = (n * w).cumsum(1) / w.cumsum(1).clamp_min(1.0)
        return TextMemory(k, v, allowed, hist[:, Tk - Tq:])

    def draw_passes(self) -> Optional[int]:
        """The number of debate passes for one training forward (T54), or
        None = use ``debate_passes``."""
        if self.training and self.train_passes > 1:
            return self._pass_rng.randint(1, self.train_passes)
        return None

    def forward(self, hT: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ctx: Optional[torch.Tensor] = None,
                passes: Optional[int] = None) -> SandboxOutput:
        """hT [B,T,H]; mask [B,T] (1 = token the loss trains) only affects
        the diagnostics / spread floor, never the consensus; attn_mask
        [B,Tk] (1 = real token) keeps padding out of the text read; ctx
        [B,Tk,H] = the model's states of all tokens so far (default hT;
        longer during cached generation)."""
        mem = None
        if self.mem_key is not None:
            mem = self.memory(hT if ctx is None else ctx, hT.shape[1], attn_mask)
        traj = self.init_proj(hT)
        for _ in range(self.debate_passes if passes is None else passes):
            for layer in self.debate:
                traj = layer(traj, mem)
        traj, a_t, energy, retention = self.descent(
            traj, hT, self.steps, self.eta, mask
        )
        aux = self._aux(traj, hT, a_t, energy, retention, mask)
        if self.verifier is not None:
            w, sup = self.verifier(traj, mem.history if mem is not None else None)
            consensus = (w.unsqueeze(-1).type_as(traj) * traj).sum(0)
            conf = self._confidence(traj, consensus, w)
            top = w.detach().max(dim=0).values
            vote = (w.detach() * sup).sum(0)
            aux["confidence"] = masked_mean_bt(conf, mask)
            aux["verifier_top_weight"] = masked_mean_bt(top, mask)
            aux["vote_support"] = masked_mean_bt(vote, mask)
            feats = torch.stack([conf, top.float(), vote.float()], dim=-1)
            return SandboxOutput(consensus, traj, aux, conf, feats)
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
        mem = (self.memory(hT, hT.shape[1], attn_mask)
               if self.mem_key is not None else None)
        traj = self.init_proj(hT)
        states.append(traj)
        for layer in self.debate:
            traj = layer(traj, mem)
            states.append(traj)
        return states


# ---------------------------------------------------------------------------
# The sandbox, v0.1.0 (Ekdant v0.1.0, spec 04-Model-DeepDive/30)
# ---------------------------------------------------------------------------

# candidate 0 is the model's own state, kept unchanged; 1-4 are the roles
CANDIDATES = ("h", "R1_explore", "R2_consistency", "R3_uncertainty", "R4_supported")


def _generator(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    """A role's learnable transformation.  The last layer starts at zero, so
    an untrained role proposes no change (every arm = the backbone at step 0)."""
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))
    nn.init.zeros_(net[2].weight)
    nn.init.zeros_(net[2].bias)
    return net


class CandidateRound(nn.Module):
    """One communication / refinement round: every candidate reads the other
    four (``CrossRead``, never itself); each role then takes as much of what it
    read as its own repair gate allows (as ``DebateRound``), through its own
    output projection, so the roles stay distinct.  Candidate 0 (h) is read
    but never changed.  Output projections start at zero: an untrained round
    changes nothing."""

    def __init__(self, dim: int, roles: int = 4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.read = CrossRead(dim)
        self.gate = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(roles)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(roles)])
        for g, o in zip(self.gate, self.out):
            nn.init.constant_(g.bias, -2.0)
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

    def forward(self, x: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """x [B,T,D]; deltas [R,B,T,D] -> refined deltas."""
        cand = torch.cat([x.unsqueeze(0), x.unsqueeze(0) + deltas], dim=0)
        n = self.norm(cand)
        cross = self.read(n)
        new = []
        for i in range(deltas.shape[0]):
            gamma = torch.sigmoid(self.gate[i](torch.cat([n[i + 1], cross[i + 1]], dim=-1)))
            new.append(deltas[i] + gamma * self.out[i](cross[i + 1]))
        return torch.stack(new, dim=0)


class SandboxV01(nn.Module):
    """The sandbox v0.1.0: five candidate states per token - the model's own
    state h and four role proposals R_i = h + delta_i - one communication
    round, then a learned gate that weighs the five per token.

        R1 explore       delta orthogonal to h: an alternative reading, never
                         a copy or a rescaling of h
        R2 consistency   reads the earlier tokens (causal) with its own query
        R4 supported     reads the earlier tokens with its own query and the
                         proposals of R1 and R2: the best-supported alternative
        R3 uncertainty   reads how much R1, R2 and R4 disagree (per-channel
                         spread) next to h

    ``consensus`` is the selected change sum_i p_i delta_i (delta_0 = 0 for h)
    in the units of the normalised input; ``GatedAddOn`` sizes it to the
    stream.  ``force`` = i makes the gate pick candidate i (the candidate
    ablation)."""

    def __init__(self, dim: int, heads: int = 0, hidden_ratio: float = 4.0,
                 rounds: int = 1, gate_hidden: int = 256, r3_probe: bool = False):
        super().__init__()
        heads = heads or max(1, dim // 64)
        if dim % heads:
            raise ValueError(f"sandbox_heads {heads} must divide dim {dim}")
        if rounds < 0:
            raise ValueError(f"rounds must be >= 0, got {rounds}")
        self.dim, self.heads = dim, heads
        hidden = int(dim * hidden_ratio)
        # the text the roles read: the model's own states (as Sandbox, T51)
        self.mem_norm = RMSNorm(dim)
        self.mem_key = nn.Linear(dim, dim, bias=False)
        self.q2 = nn.Linear(dim, dim, bias=False)
        self.q4 = nn.Linear(dim, dim, bias=False)
        self.r1 = _generator(dim, hidden, dim)
        self.r2 = _generator(2 * dim, hidden, dim)
        self.r3 = _generator(2 * dim, hidden, dim)
        self.r4 = _generator(4 * dim, hidden, dim)
        self.rounds = nn.ModuleList([CandidateRound(dim) for _ in range(rounds)])
        self.cand_norm = RMSNorm(dim)
        self.gate = nn.Sequential(nn.Linear(3 * dim, gate_hidden), nn.SiLU(),
                                  nn.Linear(gate_hidden, 1))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        self.force: Optional[int] = None
        # Role objectives (Phase 5, spec 8.3), all off by default:
        # R3's probe predicts from R3's proposal whether the model's next
        # token is right (EkdantLM adds the BCE); explore_weight > 0 keeps the
        # gate's entropy (with gradient) for the early-exploration bonus.
        self.r3_probe = nn.Linear(dim, 1) if r3_probe else None
        self.explore_weight = 0.0
        self.last_r3_logit: Optional[torch.Tensor] = None
        self.last_entropy: Optional[torch.Tensor] = None
        # what EkdantLM.think / GatedAddOn expect of a sandbox
        self.debate_passes = 1
        self.steps = 0

    memory = Sandbox.memory

    def draw_passes(self) -> Optional[int]:
        return None

    def _read(self, q: nn.Linear, x: torch.Tensor, mem: TextMemory) -> torch.Tensor:
        B, T, D = x.shape
        H = self.heads
        qh = q(x).view(B, T, H, D // H).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, mem.keys.type_as(qh),
                                           mem.values.type_as(qh), attn_mask=mem.allowed)
        return o.transpose(1, 2).reshape(B, T, D)

    def role_parameters(self) -> Dict[str, List[nn.Parameter]]:
        return {"R1_explore": list(self.r1.parameters()),
                "R2_consistency": list(self.r2.parameters()) + list(self.q2.parameters()),
                "R3_uncertainty": list(self.r3.parameters()) + (
                    list(self.r3_probe.parameters()) if self.r3_probe is not None else []),
                "R4_supported": list(self.r4.parameters()) + list(self.q4.parameters()),
                "gate": list(self.gate.parameters()) + list(self.cand_norm.parameters()),
                "comm": (list(self.rounds.parameters()) + list(self.mem_norm.parameters())
                         + list(self.mem_key.parameters()))}

    def forward(self, hT: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ctx: Optional[torch.Tensor] = None,
                passes: Optional[int] = None) -> SandboxOutput:
        x = hT
        mem = self.memory(x if ctx is None else ctx, x.shape[1], attn_mask)
        u = self.r1(x)
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d1 = u - (u * xn).sum(-1, keepdim=True) * xn
        d2 = self.r2(torch.cat([x, self._read(self.q2, x, mem)], dim=-1))
        d4 = self.r4(torch.cat([x, self._read(self.q4, x, mem), d1, d2], dim=-1))
        # sqrt(var + eps): the spread of three equal proposals is 0 at init,
        # where a plain std would have an infinite gradient
        spread = (torch.stack([d1, d2, d4]).float().var(0, unbiased=False) + 1e-6).sqrt()
        d3 = self.r3(torch.cat([x, spread.type_as(x)], dim=-1))
        deltas = torch.stack([d1, d2, d3, d4], dim=0)                     # [4,B,T,D]
        for rnd in self.rounds:
            deltas = rnd(x, deltas)
        cand = torch.cat([x.unsqueeze(0), x.unsqueeze(0) + deltas], dim=0)  # [5,B,T,D]
        ctx_feat = torch.cat([x, mem.history.type_as(x)], dim=-1)
        s = self.gate(torch.cat([self.cand_norm(cand),
                                 ctx_feat.unsqueeze(0).expand(5, *ctx_feat.shape)],
                                dim=-1)).squeeze(-1).float()               # [5,B,T]
        if self.force is not None:
            p = F.one_hot(torch.full(s.shape[1:], int(self.force), dtype=torch.long,
                                     device=s.device), 5).float().permute(2, 0, 1)
        else:
            p = s.softmax(dim=0)
        consensus = (p[1:].unsqueeze(-1).type_as(deltas) * deltas).sum(0)
        self.last_r3_logit = (self.r3_probe(deltas[2]).squeeze(-1)
                              if self.r3_probe is not None else None)
        self.last_entropy = None
        if self.explore_weight > 0 and self.training and self.force is None:
            ent = -(p * p.clamp_min(1e-12).log()).sum(0)                    # [B,T], grad
            self.last_entropy = masked_mean_bt(ent, mask)
        aux = self._diagnostics(p, cand, deltas, mask)
        return SandboxOutput(consensus, cand, aux, None, p.permute(1, 2, 0).contiguous())

    @staticmethod
    @torch.no_grad()
    def _diagnostics(p: torch.Tensor, cand: torch.Tensor, deltas: torch.Tensor,
                     mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Selection and collapse diagnostics over real tokens (spec §8.2)."""
        p = p.float()
        top = F.one_hot(p.argmax(0), 5).float().permute(2, 0, 1)            # [5,B,T]
        ent = -(p * p.clamp_min(1e-12).log()).sum(0)                         # [B,T]
        c = cand.float()
        c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos = torch.einsum("ibtd,jbtd->ijbt", c, c)                          # [5,5,B,T]
        rms = deltas.float().pow(2).mean(-1).sqrt()                          # [4,B,T]
        return {"sel_p": masked_mean_bt(p, mask),
                "sel_top": masked_mean_bt(top, mask),
                "sel_entropy": masked_mean_bt(ent, mask),
                "cand_cos": masked_mean_bt(cos, mask),
                "delta_rms": masked_mean_bt(rms, mask)}


def build_sandbox(dim: int, version: str = "0.0.5", **kw) -> nn.Module:
    """The sandbox of a given version: 0.0.x = ``Sandbox`` (v1-v5),
    0.1.0 = ``SandboxV01``."""
    if version == "0.1.0":
        return SandboxV01(dim, **kw)
    return Sandbox(dim, **kw)
