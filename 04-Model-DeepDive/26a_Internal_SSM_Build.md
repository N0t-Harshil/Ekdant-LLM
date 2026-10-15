# 26a — Build Manual: internal selective SSM (L1)

> Prev: [02](../02-Architecture/02_System_Architecture.md) · Next: [26b](26b_Dense_Transformer_for_T.md) · Map: [README](../README.md) · Spec: [02](../02-Architecture/02_System_Architecture.md) §3

> [!important] Everything below is buildable with ==PyTorch core only== — no `mamba-ssm`, no `causal-conv1d`, no Triton. Reference behavior: Mamba Algorithm 2 (B5 §3); readable models: `mamba-minimal`, `mamba-tiny` (cumsum scan).

## 1. Block math (XS: H=256, d=96, N=2 · S: H=320, d=128, N=4 · M: H=512, d=256, N=4; d per store ×3 stores)

Per position, with input-dependent selection:

1. `u = DWConv1d_k4(SiLU(Linear_{H→2H}(x)))` — causal depthwise conv, kernel 4, implemented manually (cumsum-shift, not a package).
2. `Δ,B,C = split(Linear(u))` — Δ via softplus (slow-forget bias init below), B/C plain projections.
3. Discretize: `Ā = exp(Δ·A)`, `B̄ = Δ·B`, diagonal A (S4D-style, per vault 19 decision — no full HiPPO NPLR).
4. **Scan (train):** cumulative-sum / logcumsumexp parallel scan over T — linear work, autograd flows (the `mamba-tiny` trick).
5. **Recur (generate):** carry fixed state `[B,d_state]` step to step — the O(1) mode.
6. Readout `y = C·x`, SiLU gate, residual, RMSNorm.

## 2. Init recipe (in-house)

- `A_log = log(arange(1, d+1))`, used as `A = -softplus(A_log)` (S4D-Lin style long memory).
- Δ projection bias ∼ U such that `dt ∈ [0.001, 0.1]` at start (forget slowly, then learn selection).
- B/C/Xavier; conv depthwise Xavier; gate projections zeros (start open); norms ones.

## 3. Numerics

- BF16 model, but keep the ==scan state and Δ accumulation in fp32== (selection ratios overflow in low precision).
- Grad clip 1.0; smoke NaNs → check Δ bias and fp32 state first ([13](../06-Training-Eval/13_Debugging_Guide.md) §6).

## 4. Params (estimate, verify with `estimate_model_size`)

- Per block ≈ shared trunk + 3 selective heads (≈1.5× single-store): XS ~1.5M · S ~4.8M · M ~13M; embeddings tied (XS 8.4M · S 10.5M · M 16.8M, shared with head).
- Envelope check: counted XS ≈ 12M · S ≈ 21M · M ≈ 46M — count, don't guess.

## 5. Tests that prove it (design target `test_ekdant.py`)

> L1 is not in the current `09-Code` package (L2-only build); these are design-target tests for when L1 is integrated. Shipped tests cover T + L2 only.

Shapes `[B,T,H]` + `compressed [B,3·d_state]` triple; grads reach all 3 channels' Δ/B/C; fusion weights move; feedback port carries `f`; selective-copying unit check (B5 Table 1 style); O(1) curve 512/2048/4096; 10-step loss decrease.

## 6. Triple-store build notes (aSSM)

- Code shape: one shared trunk (`Linear H→2H` + causal DWConv) → three lightweight selective heads (per-channel Δ/B/C linears + diagonal A). Mamba-2 multi-head precedent (B6).
- Per-channel Δ regimes: slow bias (content, dt≈0.01–0.1) vs. fast bias (control/sandbox, dt≈0.1–0.5) — set via projection bias, learned after.
- M3 feedback wiring: `forward(..., prev_feedback)` adds `F·f_{t−1}` to M3 only; train with detached previous-micro-batch `f`, generate with exact carry. Test the port with a nonzero `f` asserting M3 moves and M1/M2 don't.
- Causality: sandbox reads per-position `m3_t`; pooled triple is logging-only.

## Connections
- Why: [19](../03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md) (B5 centerpiece) · How: [21](21_Thesis_Ch3_Layer1_SSM_Storage.md) · Next build: [26b](26b_Dense_Transformer_for_T.md)
