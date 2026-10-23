"""Ekdant model assembly.

    tokens -> embed -> T (windowed dense) -> L2 (diverse sandbox, optional)
           -> tied head -> logits

The A/B switch is a single boolean.  Both variants share the identical
backbone; only the sandbox layer differs, so any accuracy gap is
attributable to L2 alone.

Spec: 02-Architecture/03_Codebase_Map.md, 04-Model-DeepDive/26c
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekdant.output_head import OutputHead
from ekdant.sandbox import Sandbox, SandboxOutput
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
    """``h + g * (norm(consensus(h)) - h)`` with ``g`` a per-channel gate
    initialised to 0, where ``consensus`` is the L2 sandbox.

    At step 0 the output is exactly ``h``, so the sandbox arm starts as the
    backbone; any later difference comes from training L2.  ``h`` is already
    past SmolLM2's final norm and goes straight to the head, so the
    consensus is RMS-normalised first (v1 had ``ln_f`` after L2): an
    unbounded L2 output otherwise grew the logits without limit (T32)."""

    def __init__(self, dim: int, sandbox_kwargs: Dict):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
        self.sandbox = Sandbox(dim, **sandbox_kwargs)
        self.out_norm = RMSNorm(dim)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.sandbox(h, mask=mask)
        return h + self.gate * (self.out_norm(out.consensus) - h), out.aux

    def gate_magnitude(self) -> float:
        return float(self.gate.detach().abs().mean())


@dataclass
class V2Config:
    """v2: pretrained backbone, with (``sandbox``) or without (``none``) L2.
    The sandbox defaults scale L2 to the v1 share of the model (spec §3)."""
    backbone: str = "HuggingFaceTB/SmolLM2-135M"
    addon: str = "none"
    n_sandbox: int = 4
    debate_rounds: int = 4
    sandbox_mlp_ratio: float = 4.0
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

    def validate(self) -> "V2Config":
        if self.addon not in ("none", "sandbox"):
            raise ValueError(f"addon must be none|sandbox, got {self.addon!r}")
        if self.readout not in ("mean", "lowest_energy"):
            raise ValueError(f"unknown readout {self.readout!r}")
        return self

    def sandbox_kwargs(self) -> Dict:
        return dict(K=self.n_sandbox, rounds=self.debate_rounds,
                    steps=self.descent_steps, eta=self.eta,
                    rho_min=self.rho_min, readout=self.readout,
                    mlp_ratio=self.sandbox_mlp_ratio)

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
        self.lm = lm
        dim = lm.config.hidden_size
        self.addon: Optional[GatedAddOn] = None
        if cfg.addon == "sandbox":
            self.addon = GatedAddOn(dim, cfg.sandbox_kwargs())

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values=None, use_cache: bool = False,
                position_ids: Optional[torch.Tensor] = None) -> ModelOutput:
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=use_cache, position_ids=position_ids)
        h = out.last_hidden_state
        aux: Dict = {}
        if self.addon is not None:
            # logits[t] predicts labels[t+1]: diagnostics / spread floor
            # cover exactly the positions the loss trains.  Incremental
            # decoding steps carry no mask (diagnostics are unused there).
            if labels is not None:
                dmask = F.pad((labels[:, 1:] != -100).long(), (0, 1))
            else:
                dmask = attention_mask if past_key_values is None else None
            h, aux = self.addon(h, dmask)
        logits = self.lm.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            rho = aux.get("spread_ratio")
            if rho is not None:
                loss = loss + self.cfg.lambda_sbx * F.relu(
                    self.cfg.rho_min - rho).pow(2).mean()
        if use_cache:
            aux["past_key_values"] = out.past_key_values
        return ModelOutput(logits=logits, loss=loss, aux=aux)

    def addon_params(self) -> int:
        return count_parameters(self.addon) if self.addon is not None else 0

    def param_groups(self, backbone_lr: float, addon_lr: float,
                     wd: float = 0.1) -> list:
        """Four AdamW groups: {backbone, add-on} x {decay, no decay}.
        Decay only on >=2-D weights; the gate and norms/biases get none."""
        addon_ids = ({id(p) for p in self.addon.parameters()}
                     if self.addon is not None else set())
        buckets: Dict[Tuple[bool, bool], list] = {}
        for p in self.parameters():
            if not p.requires_grad:
                continue
            key = (id(p) in addon_ids, p.dim() >= 2)
            buckets.setdefault(key, []).append(p)
        return [{"params": ps, "lr": addon_lr if is_addon else backbone_lr,
                 "base_lr": addon_lr if is_addon else backbone_lr,
                 "weight_decay": wd if decay else 0.0}
                for (is_addon, decay), ps in buckets.items()]
