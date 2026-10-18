# 28 — Worked Examples: both layers solved by hand (no code)

> Prev: [26c](26c_Internal_Sandbox_Build.md) · Map: [README](../README.md) · Math: [21](21_Thesis_Ch3_Layer1_SSM_Storage.md) §6 · [22](22_Thesis_Ch3_Layer2_Sandbox.md) §6
> Every number below was computed and checked — follow with a calculator, not a GPU.

## Example 1 — L1 selective scan, 3 steps by hand

Setup (single store, scalar state): A = −1.0, x₀ = 0; per-step (Δ, B, u, C) = (0.5, 1.0, 1.0, 1.0), (1.0, 0.5, 2.0, 1.0), (0.2, 2.0, 0.5, 1.0). `Ā = exp(ΔA) = exp(−Δ)`. `B̄` has **two** conventions — see `21` §6 — so both are computed:

- **Exact ZOH:** `B̄ = ((e^{ΔA} − 1)/A)·B`; with `A = −1` this is `(1 − e^{−Δ})·B`. Check against the series `Δ − Δ²/2 + Δ³/6 − Δ⁴/24`: at Δ=0.5, 0.393229 vs 0.393469 ✓.
- **First-order (what we implement, Mamba-1):** `B̄ = ΔB`.

| t | Ā = exp(−Δ) | B̄ exact | xₜ exact = Ā·xₜ₋₁ + B̄·uₜ | yₜ = C·xₜ | B̄ 1st-order | xₜ 1st-order | 1st-order overstates by |
|---|---|---|---|---|---|---|---|
| 1 | 0.6065 | 0.3935 | 0.6065·0 + 0.3935·1.0 = **0.3935** | **0.3935** | 0.5000 | 0.6065·0 + 0.5·1.0 = 0.5000 | +27% |
| 2 | 0.3679 | 0.3161 | 0.3679·0.3935 + 0.3161·2.0 = 0.1448 + 0.6321 = **0.7769** | **0.7769** | 0.5000 | 0.3679·0.5 + 0.5·2.0 = 1.1839 | +52% |
| 3 | 0.8187 | 0.3625 | 0.8187·0.7769 + 0.3625·0.5 = 0.6360 + 0.1813 = **0.8173** | **0.8173** | 0.4000 | 0.8187·1.1839 + 0.4·0.5 = 1.1693 | **+43%** |

Read it: at t=2 the big Δ (1.0) forgets most of the past (retention 0.3679) but admits the big input; at t=3 the small Δ (0.2) retains (0.8187). That selective forgetting-vs-keeping *is* the Mamba result in miniature — static SSMs cannot do this. **This holds under either `B̄` convention**: only the magnitudes move, not the retention factors `Ā`, which are exact in both.

**The honest caveat, now measured rather than asserted.** The two conventions are *not* interchangeable here: the first-order form inflates `x₃` by **43%** because Δ reaches 1.0, far from the small-Δ regime where `B̄ = ΔB(1 + ΔA/2 + …) ≈ ΔB`. The error is `ΔA/2` relative, so at our initialization range `Δ ∈ [0.001, 0.1]` (`21` §6) it is `≤ 0.5%` and harmless — but the scan arithmetic must be quoted with the convention it was computed under. `21` §6 records the decision.

## Example 2 — L2 sandbox, K=3 trajectories by hand (H=2, one position)

Init (orthogonal-distinct): T₁=[1.0, 0.0], T₂=[0.2, 1.0], T₃=[−0.5, 0.5]. Q=K=V=identity (transparency choice — real weights are learned).

**Debate round** (scores ÷√2, softmax, read, gate g=σ(0.5·own₁+0.5·read₁), revise):

| k | scores vs others | weights | read | g | revised T_k |
|---|---|---|---|---|---|
| 1 | [0.1414, −0.3536] | [0.6213, 0.3787] | [−0.0651, 0.8106] | 0.6148 | **[0.3452, 0.4984]** |
| 2 | [0.1414, 0.2828] | [0.4647, 0.5353] | [0.1971, 0.2676] | 0.5495 | **[0.1984, 0.5976]** |
| 3 | [−0.3536, 0.2828] | [0.3461, 0.6539] | [0.4768, 0.6539] | 0.4971 | **[−0.0144, 0.5765]** |

Note trajectory 1 listened mostly to 2 (0.62) — cross-examination working; with K=2 the weights would be trivially [1.0], which is why K ≥ 3 matters.

**Energy + consensus:** mean c = [0.1764, 0.5575]; mean squared distance to the consensus V = mean_k ‖T_k − c‖² = 0.023615 (the peer-mean energy of the discarded variant, without the ½ used for the anchor energy below).
**The correction that fixed this (proved in [29](29_Prior_Art_and_Novelty_Check.md) §3.1):** the original step `T ← T − η(T − c)` *cannot* change c, because `Σ_k (T_k − η(T_k − c)) = Kc` exactly — so `y = c` was bit-identical whether S = 0 or S = 8, while the squared energy shrank `(1−η)^{2S}` = 0.356, a free 64% cut. The vault now descends toward the **token-conditioned anchor** `a_t = W_a·hT_t` through a **per-trajectory gate** instead.

**Anchored gated descent, H=6, S=8, η = 0.0625** (seeded random trajectories, anchor and gates, full check in [29](29_Prior_Art_and_Novelty_Check.md)). *Note: the retention row lists four values, so this numerical check was run with K = 4 random trajectories, not the three hand-worked ones above; the seed was not recorded, so these figures cannot be re-derived by hand — they illustrate the qualitative claim, which the unit tests `test_readout_moves_descent_changes_consensus` and `test_peer_mean_target_is_a_noop` check exactly.*

| Quantity | Value |
|---|---|
| anchor `a` | [−0.3962, 0.0867, 0.2321, 0.0644, 0.6623, −0.7392] |
| consensus before descent | [−0.308273, −0.059091, …] |
| consensus after 8 AGD steps | [−0.336123, −0.042106, …] |
| how far the consensus moved | **0.2295** (L2) — nonzero, so the readout really changed |
| of which, closed toward the anchor | **0.2199** — it moved *toward* `a`, not randomly |
| per-trajectory deviation retained | **[0.827, 0.822, 0.744, 0.787]** — differs per k, i.e. refinement without collapse |

**Spread floor (the only L2 training term):** `ρ = V_pre / mean(hT²)`, dimensionless. Healthy run: ρ 1.95 → 1.22, penalty `max(0, 0.05 − ρ)²` = **0**. Collapsed run (trajectories × 0.001): ρ = 1e-6, penalty = **0.0025**. Zero cost while healthy, positive only on collapse.
**Diagnostics:** agreement â = 1/(1+V) = **0.9769** for the table above — note how little headroom a mean-squared diagnostic has at this scale, which is why ρ (normalized) is the collapse monitor and â is reserved for the RQ3 calibration plot.

## Example 3 — RoPE rotation by hand (why positions compose)

2-D pair, position m=1: angle = 1 rad. q=[1,0] → q'=[cos1, sin1]=[0.5403, 0.8415]; k=[1,0] at n=2 → k'=[cos2, sin2]=[−0.4161, 0.9093]. Score q'·k' = **0.5403 = cos(2−1)** — the score depends only on relative distance m−n, which is the whole point of RoPE (T2).

## Connections
- L1 math: [21](21_Thesis_Ch3_Layer1_SSM_Storage.md) §6 · L2 math: [22](22_Thesis_Ch3_Layer2_Sandbox.md) §6 · Code recipes: [26a](26a_Internal_SSM_Build.md) · [26c](26c_Internal_Sandbox_Build.md)
