"""Ekdant model assembly.

    tokens -> embed -> T (windowed dense) -> L2 (diverse sandbox, optional)
           -> tied head -> logits

The A/B switch is a single boolean.  Both variants share the identical
backbone; only the sandbox layer differs, so any accuracy gap is
attributable to L2 alone.

Spec: 02-Architecture/03_Codebase_Map.md, 04-Model-DeepDive/26c
"""

from __future__ import annotations

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
                 checkpoint: bool = False):
        super().__init__()
        if mode not in ("final", "mid"):
            raise ValueError(f"mode must be final|mid, got {mode!r}")
        self.mode = mode
        # ``gate`` is the raw parameter; with ``gate_cap`` > 0 the effective
        # gate is cap * tanh(gate / cap): 0 at init, slope 1 there, and never
        # beyond the cap however hard training pushes (T37).
        self.gate_cap = gate_cap
        self.checkpoint = checkpoint
        # Scheduled minimum opening (T39), set by the trainer each step:
        # g = floor + (1 - floor) * capped_gate, so floor 0 = the plain gate.
        self.gate_floor = 0.0
        self.gate = nn.Parameter(torch.zeros(dim))
        # ``sandbox`` given = one L2 shared by several sites (v2.2, spec §14)
        self.sandbox = sandbox if sandbox is not None else Sandbox(dim, **sandbox_kwargs)
        self.out_norm = RMSNorm(dim)
        if mode == "mid":
            self.in_norm = RMSNorm(dim)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.in_norm(h) if self.mode == "mid" else h
        g = self.effective_gate()
        if self.checkpoint and torch.is_grad_enabled():
            # recompute L2 in the backward pass instead of storing its
            # activations: identical numbers, far less memory (T40)
            out = torch.utils.checkpoint.checkpoint(
                self.sandbox, x, mask, use_reentrant=False)
        else:
            out = self.sandbox(x, mask=mask)
        aux = dict(out.aux)
        # L2's internal size, which the output norm hides from the loss;
        # logged so drift is visible during training (T34)
        aux["consensus_rms"] = masked_mean_bt(
            out.consensus.detach().float().pow(2).mean(-1), mask).sqrt()
        if self.mode == "mid":
            return h + g * self.out_norm(out.consensus), aux
        return h + g * (self.out_norm(out.consensus) - h), aux

    def effective_gate(self) -> torch.Tensor:
        g = (self.gate_cap * torch.tanh(self.gate / self.gate_cap)
             if self.gate_cap > 0 else self.gate)
        if self.gate_floor > 0:
            g = self.gate_floor + (1.0 - self.gate_floor) * g
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
                 final: bool = False, checkpoint: bool = False):
        super().__init__()
        one = Sandbox(dim, **sandbox_kwargs) if shared else None
        self.sites = nn.ModuleList(
            [GatedAddOn(dim, sandbox_kwargs, mode="mid", sandbox=one,
                        gate_cap=gate_cap, checkpoint=checkpoint)
             for _ in range(n_sites)])
        # ``final`` (the "mix", T40): the same L2 once more after the final
        # norm, i.e. the inner sites plus the end placement
        self.final_site = (GatedAddOn(dim, sandbox_kwargs, mode="final",
                                      sandbox=one, gate_cap=gate_cap,
                                      checkpoint=checkpoint)
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
    # LoRA on the backbone (T41): 0 = full fine-tuning (v2.0-v2.4 runs);
    # r > 0 = SmolLM2 frozen, rank-r adapters on every projection learn.
    lora_r: int = 0
    lora_alpha: float = 32.0

    def validate(self) -> "V2Config":
        if self.addon not in ("none", "sandbox"):
            raise ValueError(f"addon must be none|sandbox, got {self.addon!r}")
        if self.readout not in ("mean", "lowest_energy"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.lora_r < 0:
            raise ValueError(f"lora_r must be >= 0, got {self.lora_r}")
        if self.addon_final and not self.addon_layers:
            raise ValueError("addon_final needs addon_layers (use addon_layer=-1 "
                             "for the end placement alone)")
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
                    norm_inputs=self.sandbox_prenorm)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "V2Config":
        return cls(**d).validate()


class EkdantLM(nn.Module):
    """backbone -> last hidden state -> gated add-on -> backbone lm_head.

    The add-on mixes the K trajectories within one token position only,
    so incremental decoding with the backbone's KV cache is exact."""

    def __init__(self, cfg: V2Config, lm: nn.Module):
        super().__init__()
        self.cfg = cfg.validate()
        if cfg.lora_r > 0:
            from ekdant.lora import apply_lora
            apply_lora(lm, cfg.lora_r, cfg.lora_alpha)
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
        if cfg.addon == "sandbox":
            if cfg.addon_layers:
                self.addon = MultiAddOn(dim, cfg.sandbox_kwargs(), len(sites),
                                        shared=cfg.addon_shared,
                                        gate_cap=cfg.gate_cap,
                                        final=cfg.addon_final,
                                        checkpoint=cfg.sandbox_checkpoint)
                self._sites = list(self.addon.sites)
            else:
                self.addon = GatedAddOn(dim, cfg.sandbox_kwargs(),
                                        mode="mid" if sites else "final",
                                        gate_cap=cfg.gate_cap)
                self._sites = [self.addon] if sites else []
            for i, k in enumerate(sites):
                lm.model.layers[k - 1].register_forward_hook(
                    functools.partial(self._site_hook, i))
        self._ctx: Dict = {"mask": None, "aux": []}

    def _site_hook(self, i, module, args, output):
        """Replace decoder layer k's output by site i's gated L2 refinement."""
        h = output[0] if isinstance(output, tuple) else output
        new, aux = self._sites[i](h, self._ctx["mask"])
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
        self._ctx = {"mask": dmask, "aux": []}
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=use_cache, position_ids=position_ids)
        h = out.last_hidden_state
        aux: Dict = {}
        per_site: list = []
        if self.addon is not None and self.addon.mode == "final":
            h, aux = self.addon(h, dmask)
            per_site = [aux]
        elif self.addon is not None:
            per_site = list(self._ctx["aux"])
            if getattr(self.addon, "final_site", None) is not None:
                h, aux_end = self.addon.final_site(h, dmask)
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
        if use_cache:
            aux["past_key_values"] = out.past_key_values
        return ModelOutput(logits=logits, loss=loss, aux=aux)

    def addon_params(self) -> int:
        return count_parameters(self.addon) if self.addon is not None else 0

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
