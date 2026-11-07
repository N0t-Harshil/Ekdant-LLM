"""Ekdant model assembly.

    tokens -> embed -> T (windowed dense) -> L2 (diverse sandbox, optional)
           -> tied head -> logits

The A/B switch is a single boolean.  Both variants share the identical
backbone; only the sandbox layer differs, so any accuracy gap is
attributable to L2 alone.

Spec: 02-Architecture/03_Codebase_Map.md, 04-Model-DeepDive/26c
"""

from __future__ import annotations

import contextlib
import math
import functools
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from ekdant.output_head import OutputHead
from ekdant.sandbox import Sandbox, SandboxOutput, masked_mean_bt
from ekdant.transformer_stack import RMSNorm, TransformerStack

RHO_MIN_DEFAULT = 0.05


@dataclass
class EkdantConfig:
    vocab_size: int = 32768
    dim: int = 512
    n_t: int = 9
    n_heads: int = 8
    n_kv_heads: int = 2
    window: int = 512
    ffn_hidden: int = 1024
    max_seq_len: int = 1024
    pad_id: int = 0

    use_sandbox: bool = True
    n_sandbox: int = 4
    debate_rounds: int = 2
    descent_steps: int = 8
    eta: Optional[float] = None
    rho_min: float = RHO_MIN_DEFAULT
    lambda_sbx: float = 1.0
    readout: str = "mean"

    dropout: float = 0.0

    def validate(self) -> "EkdantConfig":
        if self.dim % self.n_heads:
            raise ValueError(f"dim {self.dim} not divisible by {self.n_heads}")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.readout not in ("mean", "lowest_energy"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.eta is None:
            self.eta = 0.5 / self.descent_steps
        return self

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "EkdantConfig":
        return cls(**d).validate()


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


class EkdantModel(nn.Module):
    def __init__(self, cfg: EkdantConfig):
        super().__init__()
        self.cfg = cfg.validate()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.dim, padding_idx=c.pad_id)
        self.t = TransformerStack(
            c.dim, c.n_t, c.n_heads, c.n_kv_heads, c.window, c.ffn_hidden,
        )
        self.sandbox: Optional[Sandbox]
        if c.use_sandbox:
            self.sandbox = Sandbox(
                c.dim, K=c.n_sandbox, rounds=c.debate_rounds,
                steps=c.descent_steps, eta=c.eta, rho_min=c.rho_min,
                readout=c.readout,
            )
        else:
            self.sandbox = None
        self.head = OutputHead(self.embed)
        self.drop = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()
        self.ln_f = RMSNorm(c.dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[c.pad_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        answer_pos: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        answer_weight: float = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """``attention_mask`` [B,T] (1 = real token) is optional and only
        restricts the sandbox diagnostics / spread floor to real tokens.
        The forward is causal and padding is on the right, so logits at
        real positions never depend on it."""
        c = self.cfg
        x = self.drop(self.embed(input_ids))

        aux: Dict[str, torch.Tensor] = {}
        hT = self.t(x)

        if self.sandbox is not None:
            out: SandboxOutput = self.sandbox(hT, mask=attention_mask)
            hT = out.consensus
            if return_aux:
                aux.update({k: v for k, v in out.aux.items()
                            if torch.is_tensor(v)})

        hT = self.ln_f(hT)
        logits = self.head(hT)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, c.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            if answer_weight > 0.0 and answer_pos is not None \
                    and answer_ids is not None:
                # The answer letter is 1 token out of ~34 (MedMCQA) or ~173
                # (MedQA), so plain LM CE spends ~97-99% of its gradient on
                # predicting question text - accuracy plateaus while train CE
                # collapses (40-epoch pilot). This term re-weights the letter
                # decision (same position/metric as eval) without dropping
                # the LM loss the sandbox needs for generation.
                b_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                ans_logits = logits[b_idx, answer_pos - 1]
                loss = loss + answer_weight * F.cross_entropy(
                    ans_logits, answer_ids)
            if self.sandbox is not None and return_aux:
                loss = loss + self._floor_penalty(aux)

        return ModelOutput(logits=logits, loss=loss, aux=aux)

    def _floor_penalty(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.cfg
        rho = aux.get("spread_ratio")
        if rho is None:
            return torch.zeros((), device=next(self.parameters()).device)
        deficit = F.relu(c.rho_min - rho)
        return c.lambda_sbx * deficit.pow(2).mean()

    @torch.no_grad()
    def score_options(
        self,
        input_ids: torch.Tensor,
        option_mask: torch.Tensor,
        n_options: int = 4,
    ) -> torch.Tensor:
        """Mean log-probability of each option's tokens.

        input_ids  [B*n_options, L]   question+option concatenated
        option_mask[B*n_options, L]   1 where the token belongs to the option
        returns    [B, n_options]
        """
        logits = self.forward(input_ids, return_aux=False).logits
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = input_ids[:, 1:]
        mask = option_mask[:, 1:].float()
        seq = (logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
        return (seq / mask.sum(-1).clamp(min=1)).view(-1, n_options)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return count_parameters(self, trainable_only)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def estimate_model_size(cfg: EkdantConfig) -> Dict[str, int]:
    cfg = cfg.validate()
    m = EkdantModel(cfg)
    emb = cfg.vocab_size * cfg.dim
    total = count_parameters(m)
    return {
        "total": total,
        "embedding_tied": emb,
        "t": sum(p.numel() for p in m.t.parameters()),
        "sandbox": sum(p.numel() for p in m.sandbox.parameters())
        if m.sandbox is not None else 0,
        "ln_f": sum(p.numel() for p in m.ln_f.parameters()),
    }


def baseline_config(cfg: EkdantConfig) -> EkdantConfig:
    d = cfg.to_dict()
    d["use_sandbox"] = False
    return EkdantConfig.from_dict(d)


# ---------------------------------------------------------------------------
# v2: L2 on a pretrained backbone (spec 07-Management/30 §3).
# ---------------------------------------------------------------------------

class GatedAddOn(nn.Module):
    """L2 joined to the backbone through a per-channel gate ``g`` that is
    initialised to 0, so at step 0 the output is exactly ``h`` and the
    sandbox arm starts as the backbone.

    ``mode="final"`` (v2.0): after the final norm, ``h + g*(norm(c(h)) - h)``.
    ``h`` goes straight to the head, so the consensus is RMS-normalised
    first (T32).
    ``mode="mid"`` (v2.1): inside the residual stream after a decoder layer,
    ``h + g*norm(c(norm(h)))`` - the standard pre-norm residual form. The
    mid-network stream has rms ~100 with outliers ~2e4, so L2 reads a
    normalised input, and later layers see its output (spec §13)."""

    def __init__(self, dim: int, sandbox_kwargs: Dict, mode: str = "final",
                 sandbox: Optional[Sandbox] = None, gate_cap: float = 0.0,
                 checkpoint: bool = False, force: bool = False,
                 token_gate: bool = False, aligned: bool = False,
                 token_gate_min: float = 0.0, stream_scale: bool = False,
                 n_read: int = 0):
        super().__init__()
        if mode not in ("final", "mid"):
            raise ValueError(f"mode must be final|mid, got {mode!r}")
        self.mode = mode
        # ``aligned`` (T51): L2's output is brought to the size of the
        # model's own state at that token, so it speaks at the model's scale
        self.aligned = aligned
        # Learned confidence + per-token gate (T50/T51).  The confidence
        # head reads L2's agreement signals (detached) and is trained to
        # predict whether the model's next token is right (``_conf_loss``);
        # the per-token gate scales the gate by 2*sigmoid(u . L2_output +
        # v * confidence), so L2 counts more where it helps and where it is
        # sure.  Both zero-init: confidence 0.5, gate factor 1.
        self.token_gate = None
        self.conf_head = None
        # T53: the factor lies in (token_gate_min, 2) and is 1 at init; with
        # 0 it is 2*sigmoid (v3_end: it fell to 0.03 during PubMedQA, i.e.
        # the per-token gate switched L2 off for the reasoning phase)
        self.token_gate_min = token_gate_min
        p = (1.0 - token_gate_min) / (2.0 - token_gate_min)
        self._tg_shift = math.log(p / (1.0 - p))
        if token_gate:
            self.conf_head = nn.Linear(dim + 3, 1)
            self.token_gate = nn.Linear(dim + 1, 1)
            for lin in (self.conf_head, self.token_gate):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        # set each forward: the confidence logits [B,T] (for the confidence
        # loss) and the states L2 read (the generation cache, T51)
        self.last_conf_logit: Optional[torch.Tensor] = None
        self.last_ctx: Optional[torch.Tensor] = None
        # ``gate`` is the raw parameter; with ``gate_cap`` > 0 the effective
        # gate is cap * tanh(gate / cap): 0 at init, slope 1 there, and never
        # beyond the cap however hard training pushes (T37).
        self.gate_cap = gate_cap
        self.checkpoint = checkpoint
        # Forced L2 (T42): the floor becomes a hard minimum, L2's output
        # scale is frozen (it cannot be shrunk to zero behind the gate) and
        # a mid site's output is sized to the residual stream.
        self.force = force
        # False = L2 switched off at this site (the L2-off test, T44)
        self.enabled = True
        # Scheduled minimum opening (T39), set by the trainer each step:
        # g = floor + (1 - floor) * capped_gate, so floor 0 = the plain gate.
        self.gate_floor = 0.0
        self.gate = nn.Parameter(torch.zeros(dim))
        # T56: a mid site's output sized to the residual stream (as forced)
        self.stream_scale = stream_scale
        # T56: per-channel weights of the earlier depths L2's memory reads
        # (zero-init: an untrained read is the plain v3 memory)
        self.read_w = nn.Parameter(torch.zeros(n_read, dim)) if n_read else None
        # ``sandbox`` given = one L2 shared by several sites (v2.2, spec §14)
        self.sandbox = sandbox if sandbox is not None else Sandbox(dim, **sandbox_kwargs)
        self.out_norm = RMSNorm(dim)
        if force:
            self.out_norm.weight.requires_grad_(False)
        if mode == "mid":
            self.in_norm = RMSNorm(dim)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                past_ctx: Optional[torch.Tensor] = None,
                deep: Optional[list] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.last_conf_logit = None
        if not self.enabled:
            self.last_ctx = None
            return h, {}
        x = self.in_norm(h) if self.mode == "mid" else h
        now = x
        if self.read_w is not None and deep:
            # T56: what the model knew about each word at earlier depths,
            # brought to the size of x and mixed in per channel
            size = x.float().pow(2).mean(-1, keepdim=True).sqrt()
            for w, d in zip(self.read_w, deep):
                r = (size / d.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
                now = now + w.type_as(x) * (d.float() * r.detach()).type_as(x)
        # what L2 reads: the model's states of every token so far
        ctx = now if past_ctx is None else torch.cat([past_ctx.type_as(x), now], dim=1)
        self.last_ctx = ctx.detach()
        g = self.effective_gate()
        # drawn here, outside the checkpoint, so the recompute in the
        # backward pass runs the same number of passes (T54)
        passes = self.sandbox.draw_passes()
        if self.checkpoint and torch.is_grad_enabled():
            # recompute L2 in the backward pass instead of storing its
            # activations: identical numbers, far less memory (T40)
            out = torch.utils.checkpoint.checkpoint(
                self.sandbox, x, mask, attn_mask, ctx, passes, use_reentrant=False)
        else:
            out = self.sandbox(x, mask=mask, attn_mask=attn_mask, ctx=ctx, passes=passes)
        aux = dict(out.aux)
        # L2's internal size, which the output norm hides from the loss;
        # logged so drift is visible during training (T34)
        aux["consensus_rms"] = masked_mean_bt(
            out.consensus.detach().float().pow(2).mean(-1), mask).sqrt()
        y = self.out_norm(out.consensus)
        if self.aligned:
            # the model's scale at this token (end: the state itself; mid:
            # the normalised input L2 read)
            ref = h if self.mode == "final" else x
            y = y * ref.float().pow(2).mean(-1, keepdim=True).sqrt().detach().type_as(y)
        if self.token_gate is not None:
            feats = out.features if out.features is not None else \
                out.confidence.unsqueeze(-1).expand(*out.confidence.shape, 3)
            logit = self.conf_head(torch.cat([y.detach(), feats.type_as(y)], dim=-1)).squeeze(-1)
            self.last_conf_logit = logit
            conf = torch.sigmoid(logit).detach().unsqueeze(-1)       # [B,T,1]
            aux["learned_confidence"] = masked_mean_bt(conf.float().squeeze(-1), mask)
            lo = self.token_gate_min
            f = lo + (2.0 - lo) * torch.sigmoid(
                self.token_gate(torch.cat([y, conf], dim=-1)) + self._tg_shift)
            g = g * f                                                 # [B,T,dim]
            if self.gate_cap > 0:
                # the per-token factor may not push the gate past its cap
                # (at the end, g > 1 would overshoot past L2's output)
                g = self.gate_cap * torch.tanh(g / self.gate_cap)
            if self.force and self.gate_floor > 0:
                # forced L2: the per-token gate never goes below the floor
                g = g + (g.clamp(min=self.gate_floor) - g).detach()
            aux["token_gate"] = masked_mean_bt(f.detach().float().squeeze(-1), mask)
        if self.mode == "mid":
            add = y
            if self.force or self.stream_scale:
                # sized to the stream: g = 0.5 adds half the stream's rms
                add = add * h.float().pow(2).mean(-1, keepdim=True).sqrt().detach().type_as(h)
            return h + g * add, aux
        return h + g * (y - h), aux

    def effective_gate(self) -> torch.Tensor:
        g = (self.gate_cap * torch.tanh(self.gate / self.gate_cap)
             if self.gate_cap > 0 else self.gate)
        if self.gate_floor > 0:
            g = self.gate_floor + (1.0 - self.gate_floor) * g
            if self.force:
                # hard minimum; straight-through, so the raw gate keeps its
                # gradient and the gate can still open further
                g = g + (g.clamp(min=self.gate_floor) - g).detach()
        return g

    def set_gate_floor(self, floor: float) -> None:
        self.gate_floor = float(floor)

    def gate_magnitude(self) -> float:
        return float(self.effective_gate().detach().abs().mean())

    def gates(self) -> list:
        return [self.gate]

    def site_gate_magnitudes(self) -> list:
        return [self.gate_magnitude()]


class MultiAddOn(nn.Module):
    """The same L2 at several depths (v2.2, spec §14).  With ``shared`` one
    Sandbox serves every site; each site keeps its own zero-init gate and
    its own input/output norms, because the residual stream's scale differs
    a lot between depths (rms ~6 at layer 10, ~100 at 20, ~2 at the end).
    ``shared=False`` gives every site its own Sandbox (the fallback)."""

    mode = "mid"

    def __init__(self, dim: int, sandbox_kwargs: Dict, n_sites: int,
                 shared: bool = True, gate_cap: float = 0.0,
                 final: bool = False, checkpoint: bool = False,
                 force: bool = False, token_gate: bool = False,
                 aligned: bool = False, token_gate_min: float = 0.0,
                 stream_scale: bool = False, n_reads: Optional[list] = None):
        super().__init__()
        one = Sandbox(dim, **sandbox_kwargs) if shared else None
        # T56: n_reads = how many earlier depths each site reads (mid sites,
        # then the end)
        n_reads = list(n_reads) if n_reads else [0] * (n_sites + int(final))
        self.sites = nn.ModuleList(
            [GatedAddOn(dim, sandbox_kwargs, mode="mid", sandbox=one,
                        gate_cap=gate_cap, checkpoint=checkpoint, force=force,
                        token_gate=token_gate, aligned=aligned,
                        token_gate_min=token_gate_min, stream_scale=stream_scale,
                        n_read=n_reads[i])
             for i in range(n_sites)])
        # ``final`` (the "mix", T40): the same L2 once more after the final
        # norm, i.e. the inner sites plus the end placement
        self.final_site = (GatedAddOn(dim, sandbox_kwargs, mode="final",
                                      sandbox=one, gate_cap=gate_cap,
                                      checkpoint=checkpoint, force=force,
                                      token_gate=token_gate, aligned=aligned,
                                      token_gate_min=token_gate_min,
                                      n_read=n_reads[-1])
                           if final else None)

    def _all(self) -> list:
        return list(self.sites) + ([self.final_site] if self.final_site else [])

    def gates(self) -> list:
        return [s.gate for s in self._all()]

    def set_gate_floor(self, floor: float) -> None:
        for s in self._all():
            s.set_gate_floor(floor)

    def site_gate_magnitudes(self) -> list:
        return [s.gate_magnitude() for s in self._all()]

    def gate_magnitude(self) -> float:
        g = self.site_gate_magnitudes()
        return sum(g) / len(g)


@dataclass
class V2Config:
    """v2: pretrained backbone, with (``sandbox``) or without (``none``) L2.
    The sandbox defaults scale L2 to the v1 share of the model (spec §3)."""
    backbone: str = "HuggingFaceTB/SmolLM2-135M"
    addon: str = "none"
    n_sandbox: int = 4
    debate_rounds: int = 4
    sandbox_mlp_ratio: float = 4.0
    # Pre-norm inside every debate round (T33); v1 builds Sandbox without it.
    sandbox_prenorm: bool = True
    descent_steps: int = 8
    eta: float = 0.0625
    rho_min: float = RHO_MIN_DEFAULT
    lambda_sbx: float = 1.0
    readout: str = "mean"
    max_seq_len: int = 1024
    max_answer: int = 512
    # Path of a git-ignored file holding a Hugging Face token (optional:
    # the backbone and datasets are public).  Never the token itself -
    # this config is saved into last.pt and results.json.
    hf_token_file: str = ""
    # Where L2 joins the backbone: -1 = after the final norm (v2.0, a dead
    # end: nothing reads it), k >= 1 = after decoder layer k, so layers
    # k+1.. and, through attention, later tokens use its output (v2.1).
    addon_layer: int = -1
    # Several depths (v2.2): if non-empty, L2 is applied after each of these
    # decoder layers and ``addon_layer`` is ignored.  ``addon_shared`` = one
    # set of L2 weights for all sites (else one copy per site).
    addon_layers: list = field(default_factory=list)
    addon_shared: bool = True
    # The "mix" (T40): with addon_layers, also apply the same L2 after the
    # final norm.  sandbox_checkpoint recomputes L2 in the backward pass.
    addon_final: bool = False
    sandbox_checkpoint: bool = False
    # Soft cap on the effective gate, cap * tanh(gate / cap) (T37);
    # 0 = uncapped (v2.0-v2.2 runs).
    gate_cap: float = 0.0
    # Forced L2 (T42): set by --gate-force together with --gate-floor; the
    # floor is then a hard minimum and L2 cannot be switched off.
    gate_force: bool = False
    # L2 v3 (T50).  sandbox_context: in every debate round each trajectory
    # first reads the earlier tokens (causal attention; before, L2 saw one
    # token at a time).  sandbox_heads: its heads (0 = dim // 64).
    # sandbox_confidence: a verifier weights the trajectories in the
    # consensus, their agreement is a per-token confidence, and a per-token
    # gate uses it.  All off = the v2 L2, so older runs load unchanged.
    sandbox_context: bool = False
    sandbox_heads: int = 0
    sandbox_confidence: bool = False
    # T51: sandbox_aligned - the trajectories start at the model's own state
    # (plus a small different nudge), the descent anchor is the model's own
    # state, and L2's output has the model's scale: L2 thinks in the model's
    # meaning instead of a random projection of it.  lambda_conf - weight of
    # the confidence loss (the confidence head learns to predict whether the
    # next token is right; its inputs are detached, so it never changes
    # what the language model learns).
    sandbox_aligned: bool = False
    lambda_conf: float = 0.1
    # T53: sandbox_diverse_read - each trajectory reads the text with its own
    # query offset (they ask different questions, so the debate does not
    # collapse); token_gate_min - the per-token gate factor never goes below
    # this, so L2 cannot be switched off token by token (0 = the T50 gate).
    sandbox_diverse_read: bool = False
    token_gate_min: float = 0.0
    # T54: sandbox_train_passes - each training forward runs the debate
    # stack 1..N times (drawn at random), so thinking twice is trained, not
    # just tried at test time; the model answers with N passes.  1 = as before.
    sandbox_train_passes: int = 1
    # T55: sandbox_free_views - v3 with v22's thinking space: the trajectories
    # start as K independent views (v22's orthogonal projections) and the
    # descent pulls them to a learned anchor, instead of starting at and being
    # pulled back to the model's own state.  Everything else of v3 stays (text
    # read, verifier, confidence, per-token gate, output at the model's scale).
    # Why: v22's trajectories really differ (spread ratio 0.22-0.46) and its
    # L2 is robust to more thinking; v3's sit at the spread floor (0.05), so
    # its 5 trajectories act almost as one.  Needs sandbox_aligned.
    sandbox_free_views: bool = False
    # T56: L2 without its two limits.  sandbox_stream_scale - at a site inside
    # the network L2's output is sized to the residual stream (as forced L2,
    # T42), so a gate of 0.05 changes 5 % of what the later layers read; before,
    # it was sized to the normalised input (rms ~1 against a stream of rms
    # ~100) and could hardly change anything (v2.2: gate 0.005).
    # sandbox_read_layers - L2's text memory also holds what the model knew
    # about every word at these earlier depths (each read with the norm the
    # model itself uses there), information the last layers have compressed
    # away; mixed in by learned per-channel weights that start at 0.  Each
    # site reads the listed layers below it.
    sandbox_stream_scale: bool = False
    sandbox_read_layers: list = field(default_factory=list)
    # T58: the end site's output norm starts as a copy of the backbone's own
    # final norm, so L2's output has the model's per-channel scale.  Before,
    # it started at 1 while SmolLM2's final state has rms ~2.1 (final-norm
    # weights ~1.8): h + g*(y - h) then mostly shrank the model's own state
    # (cos(y - h, -h) ~0.9) instead of adding what L2 computed.
    sandbox_out_from_backbone: bool = False

    def validate(self) -> "V2Config":
        if self.sandbox_out_from_backbone and self.sandbox_aligned:
            raise ValueError("sandbox_out_from_backbone sizes L2's output; sandbox_aligned "
                             "already sizes it to the model's state")
        if self.sandbox_out_from_backbone and self.sites() and not self.addon_final:
            raise ValueError("sandbox_out_from_backbone sets the end site: needs L2 at "
                             "the end (addon_layer -1 or addon_final)")
        reads = list(self.sandbox_read_layers)
        if reads != sorted(set(reads)) or any(int(l) < 1 for l in reads):
            raise ValueError(f"sandbox_read_layers must be increasing, unique and >= 1, got {reads}")
        if reads and not self.sandbox_context:
            raise ValueError("sandbox_read_layers extends L2's text read: needs sandbox_context")
        if set(reads) & set(self.sites()):
            raise ValueError(f"sandbox_read_layers {reads} must not be L2 sites {self.sites()}")
        if self.sandbox_stream_scale and not self.sites():
            raise ValueError("sandbox_stream_scale sizes L2 inside the network: needs "
                             "addon_layer or addon_layers")
        if self.sandbox_free_views and not self.sandbox_aligned:
            raise ValueError("sandbox_free_views changes the v3 (sandbox_aligned) L2; "
                             "without sandbox_aligned the views are already free")
        if self.addon not in ("none", "sandbox"):
            raise ValueError(f"addon must be none|sandbox, got {self.addon!r}")
        if self.readout not in ("mean", "lowest_energy"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.addon_final and not self.addon_layers:
            raise ValueError("addon_final needs addon_layers (use addon_layer=-1 "
                             "for the end placement alone)")
        if self.sandbox_train_passes < 1:
            raise ValueError(f"sandbox_train_passes must be >= 1, got {self.sandbox_train_passes}")
        if not 0.0 <= self.token_gate_min < 1.0:
            raise ValueError(f"token_gate_min must be in [0, 1), got {self.token_gate_min}")
        if self.sandbox_confidence and self.n_sandbox % 2 == 0:
            raise ValueError("sandbox_confidence votes between trajectories: "
                             f"n_sandbox must be odd (no ties), got {self.n_sandbox}")
        if list(self.addon_layers) != sorted(set(self.addon_layers)):
            raise ValueError(f"addon_layers must be increasing and unique, "
                             f"got {self.addon_layers}")
        return self

    def sites(self) -> list:
        """Decoder layers after which L2 runs; [] = after the final norm."""
        if self.addon_layers:
            return list(self.addon_layers)
        return [self.addon_layer] if self.addon_layer > 0 else []

    def sandbox_kwargs(self) -> Dict:
        return dict(K=self.n_sandbox, rounds=self.debate_rounds,
                    steps=self.descent_steps, eta=self.eta,
                    rho_min=self.rho_min, readout=self.readout,
                    mlp_ratio=self.sandbox_mlp_ratio,
                    norm_inputs=self.sandbox_prenorm,
                    context=self.sandbox_context, heads=self.sandbox_heads,
                    confidence=self.sandbox_confidence,
                    aligned=self.sandbox_aligned and not self.sandbox_free_views,
                    diverse_read=self.sandbox_diverse_read,
                    train_passes=self.sandbox_train_passes)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "V2Config":
        return cls(**d).validate()


class L2Cache:
    """KV cache of a model whose L2 reads the text (T51): the backbone's own
    cache plus, per L2 site, the model's states of the tokens so far."""

    def __init__(self, backbone, ctx: list):
        self.backbone = backbone
        self.ctx = ctx


class EkdantLM(nn.Module):
    """backbone -> last hidden state -> gated add-on -> backbone lm_head.

    The v2 add-on mixes the K trajectories within one token position only,
    so incremental decoding with the backbone's KV cache is exact; with
    ``sandbox_context`` (T50/T51) L2 also reads the model's states of the
    earlier tokens, which are cached alongside (``L2Cache``), so cached
    decoding stays exact."""

    def __init__(self, cfg: V2Config, lm: nn.Module):
        super().__init__()
        self.cfg = cfg.validate()
        self.lm = lm
        dim = lm.config.hidden_size
        self.addon: Optional[GatedAddOn] = None
        n_layers = len(lm.model.layers)
        if cfg.addon_layer != -1 and not 1 <= cfg.addon_layer <= n_layers:
            raise ValueError(f"addon_layer must be -1 or 1..{n_layers}, "
                             f"got {cfg.addon_layer}")
        sites = cfg.sites()
        if any(not 1 <= k <= n_layers for k in sites):
            raise ValueError(f"addon_layers must lie in 1..{n_layers}, got {sites}")
        self._sites: list = []
        reads = [int(l) for l in cfg.sandbox_read_layers]
        if any(l >= n_layers for l in reads):
            raise ValueError(f"sandbox_read_layers must lie in 1..{n_layers - 1}, got {reads}")
        # T56: the earlier depths each site reads (mid sites in order, then
        # the end, which reads every listed layer)
        ends = list(sites) + ([n_layers + 1] if (cfg.addon_final or not sites) else [])
        self._site_reads = [[l for l in reads if l < k] for k in ends]
        if cfg.addon == "sandbox":
            n_reads = [len(r) for r in self._site_reads]
            if cfg.addon_layers:
                self.addon = MultiAddOn(dim, cfg.sandbox_kwargs(), len(sites),
                                        shared=cfg.addon_shared,
                                        gate_cap=cfg.gate_cap,
                                        final=cfg.addon_final,
                                        checkpoint=cfg.sandbox_checkpoint,
                                        force=cfg.gate_force,
                                        token_gate=cfg.sandbox_confidence,
                                        aligned=cfg.sandbox_aligned,
                                        token_gate_min=cfg.token_gate_min,
                                        stream_scale=cfg.sandbox_stream_scale,
                                        n_reads=n_reads)
                self._sites = list(self.addon.sites)
            else:
                self.addon = GatedAddOn(dim, cfg.sandbox_kwargs(),
                                        mode="mid" if sites else "final",
                                        gate_cap=cfg.gate_cap,
                                        force=cfg.gate_force,
                                        token_gate=cfg.sandbox_confidence,
                                        aligned=cfg.sandbox_aligned,
                                        token_gate_min=cfg.token_gate_min,
                                        stream_scale=cfg.sandbox_stream_scale,
                                        n_read=n_reads[0])
                self._sites = [self.addon] if sites else []
            if cfg.sandbox_out_from_backbone:
                # T58: the end site speaks at the model's per-channel scale
                with torch.no_grad():
                    for s in self._all_sites():
                        if s.mode == "final":
                            s.out_norm.weight.copy_(lm.model.norm.weight)
            # capture hooks first: a read layer is never a site (validate)
            for l in reads:
                lm.model.layers[l - 1].register_forward_hook(
                    functools.partial(self._read_hook, l))
            for i, k in enumerate(sites):
                lm.model.layers[k - 1].register_forward_hook(
                    functools.partial(self._site_hook, i))
        self._ctx: Dict = {"mask": None, "attn": None, "aux": [], "l2_past": None,
                           "deep": {}}
        # T51: L2 reads the model's own states, which are cached with the
        # backbone's (``L2Cache``), so generation keeps the KV cache
        self.needs_full_context = False

    def _read_hook(self, l, module, args, output):
        """T56: keep layer l's output, read with the norm the model itself
        applies before layer l+1, for the L2 sites above it."""
        h = output[0] if isinstance(output, tuple) else output
        self._ctx["deep"][l] = self.lm.model.layers[l].input_layernorm(h)

    def _deep(self, i: int) -> Dict:
        """Keyword ``deep`` for site i (empty when it reads no earlier depth,
        so a site without T56 is called exactly as before)."""
        reads = self._site_reads[i] if i < len(self._site_reads) else []
        return {"deep": [self._ctx["deep"][l] for l in reads]} if reads else {}

    def _site_hook(self, i, module, args, output):
        """Replace decoder layer k's output by site i's gated L2 refinement."""
        h = output[0] if isinstance(output, tuple) else output
        past = self._ctx["l2_past"]
        new, aux = self._sites[i](h, self._ctx["mask"], self._ctx["attn"],
                                  past[i] if past is not None else None,
                                  **self._deep(i))
        self._ctx["aux"].append(aux)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values=None, use_cache: bool = False,
                position_ids: Optional[torch.Tensor] = None) -> ModelOutput:
        # logits[t] predicts labels[t+1]: diagnostics / spread floor cover
        # exactly the positions the loss trains.  Incremental decoding
        # steps carry no mask (diagnostics are unused there).
        if labels is not None:
            dmask = F.pad((labels[:, 1:] != -100).long(), (0, 1))
        else:
            dmask = attention_mask if past_key_values is None else None
        l2_past = None
        if isinstance(past_key_values, L2Cache):
            l2_past, past_key_values = past_key_values.ctx, past_key_values.backbone
        self._ctx = {"mask": dmask, "attn": attention_mask, "aux": [],
                     "l2_past": l2_past, "deep": {}}
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=use_cache, position_ids=position_ids)
        h = out.last_hidden_state
        aux: Dict = {}
        per_site: list = []
        if self.addon is not None and self.addon.mode == "final":
            h, aux = self.addon(h, dmask, attention_mask,
                                l2_past[0] if l2_past is not None else None,
                                **self._deep(0))
            per_site = [aux]
        elif self.addon is not None:
            per_site = list(self._ctx["aux"])
            if getattr(self.addon, "final_site", None) is not None:
                h, aux_end = self.addon.final_site(
                    h, dmask, attention_mask,
                    l2_past[-1] if l2_past is not None else None,
                    **self._deep(len(self._sites)))
                per_site.append(aux_end)
            # one entry per site: diagnostics averaged over the sites
            keys = set.intersection(*(set(a) for a in per_site)) if per_site else set()
            aux = {k: torch.stack([a[k].float() for a in per_site]).mean(0)
                   for k in keys if torch.is_tensor(per_site[0][k])}
        logits = self.lm.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            rhos = [a["spread_ratio"] for a in per_site if "spread_ratio" in a]
            if rhos:
                # the spread floor applies at every site
                loss = loss + self.cfg.lambda_sbx * torch.stack(
                    [F.relu(self.cfg.rho_min - r).pow(2).mean() for r in rhos]).mean()
            if self.cfg.sandbox_confidence and self.cfg.lambda_conf > 0:
                loss = loss + self.cfg.lambda_conf * self._conf_loss(logits, labels)
        if use_cache:
            aux["past_key_values"] = out.past_key_values
            if self.cfg.addon == "sandbox" and self.cfg.sandbox_context:
                aux["past_key_values"] = L2Cache(
                    out.past_key_values, [s.last_ctx for s in self._all_sites()])
        return ModelOutput(logits=logits, loss=loss, aux=aux)

    def _all_sites(self) -> list:
        """Every GatedAddOn in forward order (mid sites, then the end)."""
        if self.addon is None:
            return []
        if isinstance(self.addon, MultiAddOn):
            return self.addon._all()
        return [self.addon]

    def _conf_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """T51: binary cross-entropy of the confidence head against "is the
        model's next token right" (argmax == label) on the trained positions,
        averaged over the L2 sites.  The head's inputs are detached, so this
        loss trains only the head."""
        right = (logits[:, :-1].argmax(-1) == labels[:, 1:]).float().detach()
        keep = (labels[:, 1:] != -100).float()
        losses = []
        for s in self._all_sites():
            if s.last_conf_logit is None:
                continue
            z = s.last_conf_logit[:, :-1].float()
            bce = F.binary_cross_entropy_with_logits(z, right, reduction="none")
            losses.append((bce * keep).sum() / keep.sum().clamp_min(1.0))
        if not losses:
            return logits.new_zeros(()).float()
        return torch.stack(losses).mean()

    def addon_params(self) -> int:
        return count_parameters(self.addon) if self.addon is not None else 0

    def sandboxes(self) -> list:
        """Every distinct L2 Sandbox (one when shared across sites)."""
        if self.addon is None:
            return []
        sites = (self.addon._all() if isinstance(self.addon, MultiAddOn)
                 else [self.addon])
        out = []
        for s in sites:
            if all(s.sandbox is not o for o in out):
                out.append(s.sandbox)
        return out

    @contextlib.contextmanager
    def think(self, passes: Optional[int] = None, steps: Optional[int] = None):
        """More (or less) L2 thinking at test time, no retraining (T47):
        ``passes`` = times the debate stack runs (trained: 1, v3c: 2), ``steps`` =
        anchored-descent steps (trained: S, same step size eta).  Restored
        afterwards."""
        boxes = self.sandboxes()
        saved = [(b.debate_passes, b.steps) for b in boxes]
        for b in boxes:
            if passes is not None:
                b.debate_passes = passes
            if steps is not None:
                b.steps = steps
        try:
            yield self
        finally:
            for b, (p, s) in zip(boxes, saved):
                b.debate_passes, b.steps = p, s

    @contextlib.contextmanager
    def l2_off(self):
        """Every L2 site switched off: the model is exactly its backbone
        (the L2-off test, T44); switched back on afterwards."""
        sites = ([] if self.addon is None else
                 self.addon._all() if isinstance(self.addon, MultiAddOn)
                 else [self.addon])
        for s in sites:
            s.enabled = False
        try:
            yield self
        finally:
            for s in sites:
                s.enabled = True

    def param_groups(self, backbone_lr: float, addon_lr: float,
                     wd: float = 0.1, gate_lr: Optional[float] = None) -> list:
        """AdamW groups: {backbone, add-on} x {decay, no decay}, plus the
        gate alone at ``gate_lr`` when given (T35: the 576-entry gate does
        not drift like L2's matrices, so it may move faster). Decay only on
        >=2-D weights; the gate and norms/biases get none."""
        addon_ids = ({id(p) for p in self.addon.parameters()}
                     if self.addon is not None else set())
        gates = (self.addon.gates() if (self.addon is not None
                                        and gate_lr is not None) else [])
        gate_ids = {id(g) for g in gates}
        buckets: Dict[Tuple[bool, bool], list] = {}
        for p in self.parameters():
            if not p.requires_grad or id(p) in gate_ids:
                continue
            key = (id(p) in addon_ids, p.dim() >= 2)
            buckets.setdefault(key, []).append(p)
        groups = [{"params": ps, "lr": addon_lr if is_addon else backbone_lr,
                   "base_lr": addon_lr if is_addon else backbone_lr,
                   "weight_decay": wd if decay else 0.0,
                   "kind": "addon" if is_addon else "backbone"}
                  for (is_addon, decay), ps in buckets.items()]
        if gates:
            groups.append({"params": gates, "lr": gate_lr, "base_lr": gate_lr,
                           "weight_decay": 0.0, "kind": "gate"})
        return groups
