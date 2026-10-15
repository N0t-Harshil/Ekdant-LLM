# Performance Analysis (Ekdant) — Ch. 6

> Map: [README](../README.md) · Train: [07](07_Training_Pipeline.md) · Build: [26b](../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) · [26c](../04-Model-DeepDive/26c_Internal_Sandbox_Build.md) · Prior-art constraints: [29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) §3

All numbers below are from `estimate_model_size` / direct param counts on the shipped A/B configs (`09-Code/configs/{sandbox,baseline}_50m.json`, H=512, V=32768, N_T=12, W=512, K=4, R=2, S=8). Inference-state and compute figures are ==analytic projections==, not measurements — no training run exists yet.

## 1. Parameter table (measured)

| component | formula | params |
|---|---|---|
| Embedding (tied with head) | V×H = 32768×512 | **16,777,216 (16.777M)** |
| `ln_f` | H | 512 |
| T block (attn 2.5H² + FFN 3HI + norms 2H) | 655,360 + 1,572,864 + 1,024 | **2,229,248 (2.229M)** |
| T stack × N_T=12 | 12 × 2,229,248 | **26,750,976 (26.75M)** |
| L2 · DiverseInit | K·H² + K·H = 4×512² + 2048 | 1,050,624 ≈ **1.05M** |
| L2 · DebateRound ×R=2 | per round: CrossRead 3H² (0.79M) + gate 2H² (0.52M) + MLP H→2H→H (1.05M) ≈ 2.36M | ×2 ≈ **4.72M** |
| L2 · AnchoredDescent | anchor H² + gate 2H² | 786,944 ≈ **0.79M** |
| **L2 total** | 1.05 + 4.72 + 0.79 | **6,560,256 (6.56M)** |
| **sandbox variant** | 16.777 + 26.75 + 6.56 + ln_f | **≈ 50.09M** |
| **baseline variant** | identical, `use_sandbox=false` | **≈ 43.53M** |

The A/B is ==architecture-matched only== — the 6.56M delta is exactly L2. Open item: L2 is **2.3× the thesis estimate (6.56M vs 2.8M)** because the build uses full H×H q/k/v cross-projections and untied rounds; decide slim-vs-restate before publishing any results ([09-Code/README](../09-Code/README.md), [11_Future_Work](../07-Management/11_Future_Work.md) — a parameter-matched depth control is also owed per C24).

## 2. Inference-state analysis (O(1) claim — projection, not measurement)

**Thesis claim (with L1, designed not shipped):** state is flat in T — full-attention KV grows linearly (6.29 MB @ T=2048 → 201.33 MB @ T=65536, ×32) while Ekdant stays **flat ≈1.58 MB** (L1 state `compressed [B,3·d_state]` per store + windowed KV). ==These are analytic projections; L1 is not in the current code.==

**Shipped L2-only stack:** T uses a sliding window W=512, so the KV cache is bounded per layer: 2 (K+V) × W × (n_kv_heads·head_dim) × 2 B (bf16) = 2×512×128×2 = 0.25 MB/layer → **≈3.0 MB flat for any T ≥ W** (12 layers), versus O(T) for a full-attention model of the same width. Weights are constant anyway (~100 MB bf16 @ 50M). The L1 integration would shrink the *activation* side further (O(1) state), but the windowed-KV bound above is what the current code actually delivers.

## 3. Sandbox compute cost model (per token, K=4 / R=2 / S=8)

Sandbox ops run on `[K,B,T,H]` — every refinement op touches all K trajectories:

| stage | per-token MACs (dominant terms) | ≈ |
|---|---|---|
| DiverseInit | K·H² | 1.05M |
| Debate round ×2 | per round: CrossRead q/k/v K·3H² + gate K·2H² + MLP K·4H² ≈ 9.4M | 18.9M |
| Anchored descent ×S=8 | anchor H² (once) + gate K·2H² × 8 steps | 17.1M |
| **L2 total** | | **≈ 37M MACs/token** |
| T stack (reference) | ≈ params = 26.75M MACs/token | 26.75M |

**Rough multiplier:** the sandbox ≈ **1.4× the T stack's own per-token compute**, i.e. ≈ **2.4× the baseline forward** per token (plus the `[K,B,T,H]` scratch activation, ~4× the backbone activations at the same B,T). Backward roughly ×2 again. This is a hand count of dense layers — ==measure before publishing==; the dominant uncertainty is SDPA/compile efficiency on the `[K,B,T,H]` layout. For limited GPU: K=2, R=1, S=4 ablation must be reported. Bottleneck notes: T attention is O(T·W) but memory-bound SDPA; L2 is pure GEMM-bound; head is one dense [H,V] projection; data/tokenization CPU-side ([07](07_Training_Pipeline.md)).

**Reporting rule (C8):** the sandbox path re-reads the *same* K=4 times — always publish ==tokens×K== (effective trajectory-tokens) and FLOPs beside accuracy; never compare a K=4 accuracy against a K=1 baseline at nominal token count.

## 4. What must be published with any result

| item | source | status |
|---|---|---|
| Params, embed/T/L2 split (§1) | `estimate_model_size` | ✅ measured |
| tokens×K, FLOPs, wall-clock, tok/s | trainer timers + C8 rule | ⚠️ runner logs tok/s only |
| ρ trace (`spread_ratio` per step/epoch) | `Sandbox._aux` | ❌ **not logged** — `train.py` drops `aux`; add before any run |
| `anchor_energy`, `trajectory_variance` | same aux dict (three separately-named numbers, never aliased) | ❌ not logged |
| K=1 vs K=4 cost-vs-accuracy; depth/window ablations | config sweeps | ❌ open |
| Baseline vs sandbox `results.json` (letter accuracy both test sets) | runner output | ready to produce |

Ch.6 result tables stay `[PENDING]` until the LR bug ([07](07_Training_Pipeline.md) §4.1) and the ρ logging gap (§4.2 above) are fixed.

## 5. RQ2/RQ3 evaluation plan (what is runnable at 12–46M)

GSM8K/HumanEval cannot decide the sandbox claims at this scale (smallest reported 80% GSM8K model is 34B, arXiv:2312.09241) — they are *scaling-held-out*, reported only, never the gate.

| Role | Tasks | Used as |
|---|---|---|
| **Runnable now (XS/S/M)** | val+test perplexity; ARC-easy; HellaSwag; LAMBADA; synthetic known-answer multi-step items | the RQ2/RQ3 **decision gate** |
| **Scaling-held-out** | GSM8K; HumanEval; MMLU | reported only |

RQ2 = cost-matched sandbox vs. greedy vs. self-consistency-majority at equal FLOPs/tokens. RQ3 = agreement-vs-accuracy calibration plus the `spread_ratio ρ` learning curve (falling ρ = failure signal, [29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) §3.2–3.3). Addition ladder required: backbone-only vs +L1 vs +L1+L2 (matched params) on recall probe + reasoning samples.

## Connections
- Train/losses: [07](07_Training_Pipeline.md) · Validation harness: [09](09_Validation_System.md) · Debug: [13](13_Debugging_Guide.md) · Issues: [10_Known_Issues](../07-Management/10_Known_Issues.md) · Future work: [11_Future_Work](../07-Management/11_Future_Work.md)
