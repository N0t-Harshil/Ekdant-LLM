# Glossary — symbols, sizes, names (read once, link often)

> Map: [README](../README.md) · Canonical: [02](../02-Architecture/02_System_Architecture.md)

## v2 names

| Name | Meaning |
|---|---|
| backbone | pretrained SmolLM2-135M (H=576, 30 layers, vocab 49,152) |
| add-on | L2 on the backbone's last hidden state (`sandbox` arm); the `baseline` arm has none |
| gate `g` | per-channel vector in `h + g ⊙ (f(h) − h)`, initialised to 0; `|g|` = its mean absolute value |
| reference-answer perplexity | `exp` of the mean NLL over answer tokens of the reference answer — the v2 primary metric (lower is better) |
| paired bootstrap | 95 % CI of the per-answer NLL difference between two arms, resampling whole questions |
| benchmark half | the MedQuAD questions never trained on or used for selection, scored once |

## Tensors & dims

| Symbol | Meaning | XS / S / M |
|---|---|---|
| B / T / V | batch / sequence length / vocab | — / 2048 / 32768 |
| H | model width | 256 / 320 / 512 |
| d / N_ssm | SSM state dim / L1 blocks | 96×2 / 128×4 / 256×4 |
| N_T / W | Transformer blocks / local window | 3 / 5 / 6 · 512 all |
| K / R / S | sandbox paths / debate rounds / energy steps | 4 / 2 / 8 all |
| h / hT / y | L1 out `[B,T,H]` / T out `[B,T,H]` / consensus `[B,T,H]` | — |
| `compressed` | pooled L1 state `[B,d]` for the O(1) proof | — |

## Names (never confuse these)

| Name | What it is |
|---|---|
| L1 | SSM Storage (`ssm_storage.py`) — the memory layer |
| T | Dense Transformer stack (`transformer_stack.py`) — the retrieval layer (NOT tiny-size, NOT transpose) |
| L2 | Sandbox (`sandbox.py` + `output_head.py`) — the reasoning layer + head |
| Name | The model is called **Ekdant** everywhere — prose, code, configs (`EkdantModel`, `EkdantConfig`, `{sandbox,baseline}_50m.json`) |
| XS / S / M | Design-time size labels ≈12M / 21M / 46M for the full three-stage model (NOT experts, NOT steps); the built model is the 50M A/B pair (sandbox 50.09M / baseline 43.53M) |
| K | Sandbox trajectories (NOT top-k, NOT KV) |
| W | Attention window 512 (NOT width) |
| T (size) | Never used — sizes are XS/S/M precisely to avoid clashing with layer T and length T |

## Losses & knobs

- `lm_ce` (w `lm_weight`=1.0) + spread floor `max(0, ρ_min − ρ)²` (design weight `sandbox_weight`=0.1; shipped `lambda_sbx`=1.0; `ρ_min`=0.05) — the two signals, plus the optional answer-letter CE (`--answer-weight`, both arms). `ρ = trajectory_variance / mean(hT²)` is the dimensionless trajectory spread; monitors `anchor_energy`, `trajectory_variance`, `spread_ratio` are reported, never minimized.
- Every `12_Configuration_Guide.md` knob maps to an ablation in `11_Future_Work.md`.
