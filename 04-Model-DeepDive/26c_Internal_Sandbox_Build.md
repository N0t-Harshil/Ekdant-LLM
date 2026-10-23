# 26c — Build Manual: internal sandbox (L2)

> Prev: [26b](26b_Dense_Transformer_for_T.md) · Next: [07](../06-Training-Eval/07_Training_Pipeline.md) · Map: [README](../README.md) · Spec: [02](../02-Architecture/02_System_Architecture.md) §5

> [!important] Debate + energy + consensus are ==plain tensor ops== in `sandbox.py`. No agent frameworks, no transcripts, no orchestrator — K explicit trajectories, cost-matched by construction.

> [!important] **v2 build (current).** Same modules, two knobs and one wrapper: `Sandbox(576, K=4, rounds=4, steps=8, eta=0.0625, mlp_ratio=4.0)` = 19,591,488 params; wrapped by `GatedAddOn` (`model.py`), which adds the per-channel gate `g` (576 params, zeros) and an output RMSNorm (576 params) and returns `h + g ⊙ (RMSNorm(consensus) − h)` (T32). Tests: `tests/test_addon.py`, `tests/test_backbone.py`. Spec: [30](../07-Management/30_Design_v2_Pretrained_Backbone.md).

## 1. Diverse init (K=4 default)

- K distinct `Linear(H→H)` projections with ==orthogonal init + spread gains== (copies collapse — diversity is load-bearing, C19).
- Input: backbone-enhanced `hT [B,T,H]` → `trajectories [K,B,T,H]`, plus per-position M3 context `m3_t` (causal sandbox-channel states).

## 2. Debate round (×R=2, pure torch)

1. Each trajectory cross-attends the other K−1 (K ≤ 8, so full attention is trivially cheap).
2. Repair gate: `g = sigmoid(Linear([own, cross]))`; `x ← (1−g)·own + g·MLP(cross)`.
3. No text exchange anywhere — "debate" is propose→cross-read→revise on tensors (C3, in-graph).

## 3. Anchored gated descent (×S=8)

- **Anchor from the backbone, not from the peers:** `a_t = W_a·hT_t` (one `[H,H]`, `W_a`).
- Energy `E_k = ½‖T_k − a_t‖²` — reported as `anchor_energy`, never optimized directly.
- Gate `g = sigmoid(Linear([own, anchor]))` per trajectory **and** per channel; step `x ← x − η·g⊙(x − anchor)`, η ≈ 0.5/S.
- **Do not implement a pull toward the running mean.** It is mean-preserving (the target matrix is doubly stochastic), so `logits(S=0) == logits(S=8)` exactly — proved in [29](29_Prior_Art_and_Novelty_Check.md) §3.1. Narrow form: a *doubly stochastic* peer mix is inert; an arbitrary linear peer target is not. The gate is what breaks exchange symmetry here, and the anchor is what gives the step a target that is not a function of the peers.
- **What this step is, precisely:** `T_k → T_k − η g_k ⊙ (T_k − a_t)` is a **gate-scaled (preconditioned) descent** on `E_k`, not literal `−η∇E_k` unless `g ≡ 1`. Monotonicity still holds: `E_k' − E_k = (η²g_k² − 2ηg_k)‖T_k − a_t‖² ≤ 0` for `0 ≤ η g_k ≤ 2`. Say "preconditioned" in the thesis; do not write "exact energy gradient descent".
- Readout = mean (default) or per-position `argmin_k ‖T_k − a_t‖` (never batch- or sequence-averaged — T24); always report all three monitors.

## 4. Collapse guards (honesty first)

- `spread_ratio ρ → 0` is ==failure, not success==: log it per step.
- Loss uses the **spread floor** `max(0, ρ_min − ρ)²` (dimensionless; `ρ_min = 0.05`), never a bare agreement penalty — a bare `0.1·MSE(traj, consensus)` term is minimized by exactly the collapse this section forbids ([29](29_Prior_Art_and_Novelty_Check.md) §3.2).
- If collapse is still observed (not preemptively): raise `sandbox_rho_min` or lower `sandbox_weight`; never hide it, and never fix it by re-adding an agreement loss.
- Aux dict must carry three separately-named numbers: `anchor_energy`, `trajectory_variance`, `spread_ratio`. Do not alias two of them to one tensor.

## 5. Internalization path (Ch.7, not thesis)

- IMAD recipe (C10): SFT on K-path traces → RL that shortens/clips explicit rounds → single path decides (93%-fewer-tokens precedent).
- Thesis keeps explicit K with cost-matched evaluation (C8 rule).

## 6. Params + tests

- XS ≈ 0.8M · S ≈ 1.3M · M ≈ 2.8M (init + debate nets + anchor `W_a` and gate); verify with `estimate_model_size`.
- Tests: K=1 vs 4 shapes; grads to all K paths; **readout-moves assert** (`logits(S=0) != logits(S=8)`, and `‖c^(8) − c^(0)‖ > 0`); **regression guard assert** (per-trajectory deviation retention differs across k, i.e. gates are not identical); spread-floor assert (`ρ < ρ_min` ⇒ penalty > 0, healthy ρ ⇒ penalty == 0); overturn table per round; agreement-vs-accuracy calibration (RQ3).

## Connections
- Why: [20](../03-Background/20_Thesis_Ch2_Sandbox_Papers.md) (C1 control, C2 baseline, C3/C11 blueprints, C8 honesty) · How: [22](22_Thesis_Ch3_Layer2_Sandbox.md) · Train: [07](../06-Training-Eval/07_Training_Pipeline.md) · Prev build: [26b](26b_Dense_Transformer_for_T.md)
