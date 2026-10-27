# 21 — Layer 1: SSM Storage — Problem, Design, and How It Works

> Thesis Ch. 3 §1. Background: [`19_Thesis_Ch2_SSM_Storage_Papers.md`](../03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md) · PDFs: [`papers/SSM_Storage/`](../papers/SSM_Storage/) · Index: [`papers/00_Papers_Index.md`](../papers/00_Papers_Index.md)
> Code: `src/ekdant/ssm_storage.py` · Config: [`12_Configuration_Guide.md`](../02-Architecture/12_Configuration_Guide.md) · Results: [`14_Performance_Analysis.md`](../06-Training-Eval/14_Performance_Analysis.md)

## 1. Problem: memory that grows with the sequence

Self-attention compares every position with every other (quadratic time) and stores a key-value cache that grows linearly with context length. A model meant to reason over long inputs — codebases, records, papers with references — pays the most exactly where it should be strongest. L1 exists to remove that scaling wall: whatever the sequence length, training stays linear-time and inference carries a fixed-size state.

## 2. Why a selective state-space design

A classical state-space model maps an input signal through a latent state (`x' = Ax + Bu`, `y = Cx`) — inherently recurrent and constant-memory, but historically weak at long-range recall. Three results turn it into a working neural memory, and L1 follows them in order:

1. **Structured parameterization + principled initialization remember far back.** HiPPO theory derives the optimal online projection of history onto polynomial bases, and S4 builds a deep sequence model on it that handles 16K-length tasks (PDFs `papers/SSM_Storage/B13_Gu_2020_HiPPO.pdf`, `B1_Gu_2021_S4.pdf`).
2. **Diagonal and simplified variants keep the power with less machinery.** Diagonal state matrices (S4D, DSS) and the multi-input multi-output S5 match S4 with simpler implementations and parallel scans (PDFs `B2_Gu_2022_S4D.pdf`, `B3_Gupta_2022_DSS.pdf`, `B4_Smith_2022_S5.pdf`).
3. **Input-dependent selection learns what to keep.** Making the dynamics (Δ, B, C) functions of the input lets the state forget selectively — the step that closes copying and induction-head gaps and extrapolates far beyond training length. This is L1's core mechanism (PDF `papers/SSM_Storage/B5_Gu_Dao_2023_Mamba.pdf`); the duality framework unifying it with attention math is in `B6_Dao_Gu_2024_Mamba2_SSD.pdf`.

The alternatives were considered and positioned, not ignored: long convolutions (Hyena, `B7`), gated SSMs (H3, `B8`), retention (RetNet, `B9`), modern RNNs (RWKV `B10`, Griffin `B17`, xLSTM `B19`), byte-level selectivity (MambaByte, `B18`), multidimensional states (S4ND, `B15`), and liquid time-constants (Liquid-S4, `B16`). The efficiency bar is exact IO-aware attention (FlashAttention, `B20`), and the benchmark framing is the Long Range Arena (`B14`). The acknowledged price — weaker exact associative recall than attention, where hybrids win — bounds the claims section, with the survey (`B12a_Somvanshi_2025_S4_to_Mamba_Survey.pdf`) as the family-tree source.

## 3. How L1 works, end to end

Shapes use three sizes — XS (H=256, V=32768, d=96, N=2), S (H=320, V=32768, d=128, N=4) and M (H=512, V=32768, d=256, N=4) — with batch B, length T = 2048:

1. **Embed:** `x = Embed(input_ids) [B,T,H]`, weights tied with the output head (saves V×H: 10.5M S / 16.8M M).
2. **Position:** rotary injection (RoPE, base 100K scaled for small head dims, window 4096, YaRN-extendable). No learned absolute positions, no cache.
3. **aSSM triple store (×N) — how the three memories work:** one shared input trunk `U = DWConv(SiLU(Linear(h)))` is scanned by three selective channels in parallel. **M_content** (model stream 1) keeps *what was said* with a slow Δ (long memory). **M_control** (model stream 2) keeps *what to do* with a fast Δ (instruction/task features update quickly). **M_task** (formerly `M_sandbox`) tracks the running task state with a medium Δ: it scans distilled task features from the input only. L1 does not read L2 — the stages are independent (T38). Each channel has its own diagonal A/Δ/B/C selected from the shared trunk, its own recurrence `x^c_t = Ā^c_t·x^c_{t−1} + B̄^c_t·u_t (+ F·f_{t−1} only for M_sandbox)`, readout, gate, and norm. Training scans all channels in parallel (linear work); generation carries three fixed states — still O(1) in T.
4. **Output:** fused `h [B,T,H] = RMSNorm(Σ_c g_c·y^c + h_in)` with learned weights `g = softmax(Linear_fuse(h_in))`, handed to T — plus per-channel pooled `compressed = [c1; c2; c3] [B,3·d_state]` for logging and the bounded-memory proof (shape still independent of T).

## 4. Configuration (what each knob does)

| Knob (thesis default) | Role |
|-----------------------|------|
| `d_state = 96 (XS) / 128 (S) / 256 (M)` | Per-store dim (×3 stores: content/control/sandbox); ablate within size |
| `n_ssm_layers = 2 (XS) / 4 (S/M)` | Stack depth; ablate 2–6 |
| `hidden_size = 256 (XS) / 320 (S) / 512 (M)` | Model width |
| `vocab_size = 32768` | Must match the tokenizer (both sizes) |
| `rope_theta = 100K`, `max_position_embeddings = 4096` | Scaled for small head dims; window both sizes |
| `tie_word_embeddings = true` | Embedding/head weight sharing |

## 5. How L1 is tested (RQ1)

1. **O(1) proof:** forward passes at T = 512 / 2048 / 4096 must show identical `compressed [B,3·d_state]` triple shapes and flat peak memory (±5%), with tokens/second reported against the FlashAttention baseline at each length.
2. **Mechanism checks:** selective-copying accuracy and induction-head extrapolation, where selective dynamics must clearly beat static gating.
3. **Backbone ablation:** SSM vs. tiny-Transformer at equal parameters and tokens (perplexity, throughput, memory-vs-length).
4. **Recall probe:** one exact-recall task where attention is expected to win — reported as the bounded limitation, not hidden.

## 6. Math reference (implement L1 from this)

- **Continuous SSM:** `x'(t) = A·x(t) + B·u(t)`, `y(t) = C·x(t)` (skip term D = 0).
- **Discretize (ZOH, step Δ), exact:** `Ā = exp(ΔA)`, `B̄ = (ΔA)⁻¹·(exp(ΔA) − I)·ΔB`. For scalar/one diagonal entry `A`: `B̄ = ((e^{ΔA} − 1)/A)·B`.
- **Discretize, first-order (what we actually implement):** `B̄ = ΔB`. This is the **Euler/trapezoid-free approximation** of the line above, and it is the Mamba-1 convention. Expansion: `((e^{ΔA}−1)/A)B = ΔB·(1 + ΔA/2 + (ΔA)²/6 + …)`, so `B̄ = ΔB + O(Δ²)`. Sanity check by hand: with `A = −1`, `B̄ = (1 − e^{−Δ})B`, and `1 − e^{−Δ} = Δ − Δ²/2 + Δ³/6 − Δ⁴/24 + …` reproduces the closed form to 4 decimals at `Δ ∈ {0.2, 0.5, 1.0}`.
- **Convention decision (do not leave this ambiguous).** `Ā = exp(ΔA)` is exact in both; **only `B̄` differs**. We implement the first-order `B̄ = ΔB` and say so, because that is what the Mamba reference recurrence uses and it keeps `B̄` a product of already-computed tensors (no per-channel division by `A`). **The approximation is not cosmetic: on the 3-step hand check in [28](28_Worked_Examples.md) §Example 1 it inflates `x₃` by 43%** (1.1693 vs. the exact 0.8173), because `Δ` there reaches 1.0. Two honest options, and the thesis must pick one and report it: (a) keep `ΔB` and state that our `Δ` range is `≤ 0.1` at initialization, where the error is `≈ Δ²/2 < 0.5%`; or (b) use the exact `B̄` and pay one extra division per channel. **Do not quote exact-ZOH numbers from a `ΔB` implementation, or vice versa.**
- **Selection (per position t):** `Δ_t = softplus(Linear_Δ(u_t) + b_Δ)`, `B_t = Linear_B(u_t)`, `C_t = Linear_C(u_t)` — dynamics become input-dependent, so the state forgets selectively (the Mamba result).
- **Recurrence (diagonal A → elementwise):** `x_t = Ā_t·x_{t−1} + B̄_t·u_t`, readout `y_t = C_t·x_t`.
- **Gate + residual:** `y'_t = y_t · SiLU(Linear_gate(x_t))`, `h_out = RMSNorm(y' + h_in)`.
- **Init (in-house):** `A = −softplus(A_log)`, `A_log = log(arange(1, d+1))` (S4D-Lin style); Δ-bias set so `dt ∈ [0.001, 0.1]` at start; B/C Xavier; gates zero-init (start open).
- **Two modes, one math:** train = parallel associative/cumsum scan (linear work, autograd flows); generate = carry fixed `x` (`[B,d]`) step to step — O(1) inference is the same recurrence unrolled.
- **Numerics:** BF16 model, scan state + Δ accumulation in fp32.
- Worked numbers: [28_Worked_Examples.md](28_Worked_Examples.md) §1 (3-step selective scan by hand).
- **Triple math (aSSM):** shared trunk `U`; per-channel `Δ^c_t = softplus(Linear_Δc(U_t) + b^c_Δ)` (slow bias for content, fast bias for control/sandbox), `B^c_t, C^c_t` from shared projectors (Mamba-2 multi-head precedent, B6); `x^c_t`, `y^c_t` as in §3; fusion `h_out = RMSNorm(Σ_c softmax(Linear_fuse(h_in))_c · y^c + h_in)`.
- **No feedback from L2 (T38):** the task store is driven by the input only; the former write-back `F·f_{t−1}` into the third store was removed — L1 and L2 are independent stages.
- **Causality:** every store is a causal scan; the pooled triple is logging-only and never feeds forward.
