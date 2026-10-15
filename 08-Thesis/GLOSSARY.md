# Glossary — symbols, sizes, names (read once, link often)

> Map: [README](../README.md) · Canonical: [02](../02-Architecture/02_System_Architecture.md)

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
| Name | The model is called **Ekdant** everywhere — prose, code, configs (`model_type = "ekdant"`, `EkdantConfig`, `config_ekdant_xs/s/m.yaml`). No code existed yet, so all slugs were renamed cleanly |
| XS / S / M | Model sizes ≈12M / 21M / 46M (NOT experts, NOT steps) |
| K | Sandbox trajectories (NOT top-k, NOT KV) |
| W | Attention window 512 (NOT width) |
| T (size) | Never used — sizes are XS/S/M precisely to avoid clashing with layer T and length T |

## Losses & knobs

- `lm_ce` (w `lm_weight`=1.0) + spread floor `max(0, ρ_min − ρ)²` (w `sandbox_weight`=0.1, `ρ_min`=0.05) — the only two losses. `ρ = trajectory_variance / mean(hT²)` is the dimensionless trajectory spread; monitors `anchor_energy`, `trajectory_variance`, `spread_ratio` are reported, never minimized.
- Every `12_Configuration_Guide.md` knob maps to an ablation in `11_Future_Work.md`.
