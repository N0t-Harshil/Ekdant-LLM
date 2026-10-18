# 29 — Prior-Art and Novelty Check for Layer 2 (Sandbox)

> Question this file answers: *has L2 been implemented before?* Checked 2026-10-04 against the literature, not from memory. PDFs: [`papers/Sandbox_Reasoning/`](../papers/Sandbox_Reasoning/) · Index: [`papers/00_Papers_Index.md`](../papers/00_Papers_Index.md) · Spec under review: [22](22_Thesis_Ch3_Layer2_Sandbox.md) · Build: [26c](26c_Internal_Sandbox_Build.md)

## 1. Verdict

**Every individual ingredient has been implemented before. The exact combination has not.** So the thesis cannot claim "nobody has done multi-trajectory sandbox reasoning." It *can* claim a combination novelty, which must be stated precisely and with the closest competitors named.

| # | L2 ingredient | Already implemented? | Closest prior work | What is left for Ekdant |
|---|---|---|---|---|
| 1 | K parallel latent hypotheses in **one** model | **Yes** | Multi-Stream Transformers (Burtsev & Rumshisky 2021, arXiv:2107.10342) — split the encoder into streams, merge hypotheses at the end; reports gains | Ekdant's streams are token-aligned, cross-read each other, and live after a dense backbone rather than replacing it |
| 2 | Streams that **read and revise** each other | **Yes** | Talking-Heads Attention (Shazeer et al. 2020, arXiv:2003.02436) — projections across the head dimension let heads depend on all other heads | Ekdant makes each trajectory a full-width [T,H] stream with its own gate + MLP, not a head slice |
| 3 | Function-separated stores (content / control / sandbox) | **Yes, at our scale** | Dual-Stream Transformer (arXiv:2603.07461) — residual stream split into two functional components with tunable mixing, evaluated at 29M params | Three stores inside one SSM, plus M3 driven by L2 |
| 4 | Iterative **latent** refinement instead of CoT | **Yes** | Coconut (continuous thoughts); TRM / Tiny Autoregressive Recursive Models (Rauba, Fanconi & van der Schaar, arXiv:2603.08082; OpenReview `ikNCYypbT9`) — explicitly compares single-stream, two-stream and nested compute placement with weighted readout `∑_k w_k H_k`, and **finds the full Autoregressive TRM gives no reliable gain** (§4 below) | Weight sharing + integration with a selective SSM front-end |
| 5 | **Energy descent as the layer's computation** | **Yes** | Energy Transformer (NeurIPS 2023); NRGPT (arXiv:2512.16762, gradient steps on a parameterized energy, fixed step count); Hyper-SET (arXiv:2502.11646); Causal Energy Minimization (Microsoft 2026) | Our first energy was *internally defined* (distance to own consensus) and turned out provably inert — §3.1. Now anchored to the backbone state with a per-trajectory gate; still the minimal version of the idea |
| 6 | Debate **internalized** into one model | **Yes, but by distillation** | Latent Agents / IMAD (ACL 2026, arXiv:2604.24881, up to 93% fewer tokens); AgentArk (arXiv:2602.03955) | Ekdant internalizes architecturally (parallel streams in one graph), so no teacher trajectories and no distillation stage — a genuine difference in method |
| 7 | Layered agents reading the previous layer's outputs | **Yes** | Mixture-of-Agents (ICLR 2025) — layered proposal/aggregation over full LLM responses | Ekdant has no text transcripts at all |
| 8 | Agreement as a **training signal** | **Yes, and it is known to backfire** | Diversity-collapse line: reverse-KL is mode-seeking (arXiv:2510.20817); entropy collapse in RLVR (ACL 2026 Findings `2026.findings-acl.1266`, `2026.findings-acl.879`) | Our own bare agreement term had the same defect — replaced by a dimensionless spread *floor*, §3.2 |

## 2. How the novelty claim should be worded

- **Do not claim:** "no prior work runs multiple reasoning paths inside one model."
- **Can claim:** *Ekdant* is, to the best of this search, the first architecture that (a) puts multi-hypothesis tensor-level debate **after** a dense backbone and (b) closes a loop from the consensus head back into a selective-SSM store, at 12–50M parameters trained from scratch. The closest single-paper ancestor is Multi-Stream Transformers; the closest in spirit are TRM/Coconut (latent iteration) and the Energy Transformer family (descent). **However, note the strongest counter-evidence found in this audit:** C24 (arXiv:2603.08082) runs a compute-matched ladder from a plain Transformer to a two-stream recursive model and reports that the *full* Autoregressive TRM yields **no reliable performance gain** over simpler variants, with untied depth and a flat two-stream recurrent baseline generalizing best. That is direct evidence against the *prior* that recursive latent refinement helps small models. Ekdant must therefore (a) report the K=1 and no-refinement controls as the primary comparison, and (b) treat the L2 gain as **an empirical question the thesis must answer, not an assumption** — see §4.
- **Required in the thesis:** cite items 1–8 above in the same section as the contribution, and add the search method (databases, date range, queries) as an appendix so the claim is auditable.

## 3. Defects found while checking (all three fixed; evidence kept as thesis material)

### Defect 1 — the energy-descent loop could not change the prediction → **fixed by anchored gated descent**

Spec as originally written: `T_k ← T_k − η·(T_k − c)` with `c = (1/K)Σ_j T_j`, then read out `y = c`.

Because `Σ_k (T_k − η(T_k − c)) = Kc − η(Kc − Kc) = Kc`, **the mean is exactly invariant**. Therefore `c^(S) = c^(0)` bit-for-bit and `y` is unchanged by S. Verified numerically: `output identical? True`.

Closed form: `T_k^(S) = c + (1−η)^S (T_k^(0) − c)`, so with η = 0.5/S = 0.0625, S = 8: deviations retain `(1−η)^S = 0.5967`, and since `C` is a *squared* distance, `C_S/C_0 = (1−η)^{2S} = 0.356` — a **64% reduction of the loss term for free**. (An earlier vault note said "~40%"; that figure is the standard-deviation ratio, not the variance ratio the loss actually uses. Corrected in [28](28_Worked_Examples.md).)

**Fix adopted — anchored gated descent (AGD):**

| Piece | Form |
|---|---|
| anchor | `a_t = W_a·hT_t` — external, token-conditioned, one `[H,H]` |
| energy | `E_k = ½‖T_k − a_t‖²` |
| gate | `g_k = σ(W_g[T_k ; a_t] + b_g) ∈ (0,1)^H` — per trajectory **and** channel |
| step | `T_k ← T_k − η·g_k ⊙ (T_k − a_t)`, `η ≈ 0.5/S` |

Two independent changes break the invariance: the target is external rather than peer-derived, and `g_k` differs across k so `Σ_k g_k ⊙ (T_k − a) ≠ 0`. Verified: consensus moved L2 = 0.2295, closing 0.2199 of its distance to the anchor, with per-trajectory deviation retention `[0.827, 0.822, 0.744, 0.787]` — refinement, not collapse.

**General result worth stating in the thesis (stated precisely, since the loose version is false).** Write the step as `T^(k) → T^(k) + η(y^(k) − T^(k))` and average over `k`: `mean' + η( mean(y) − mean' )`. So the consensus moves **unless** the targets themselves average to the current mean. That happens exactly when the target matrix is **doubly stochastic**: if `y^(k) = ∑_j A_kj T^(j)` with rows summing to 1 (each trajectory is a convex mix of the others) **and** columns summing to 1 (no trajectory is over-represented), then `mean(y) = mean'` and the step is mean-preserving. Leave-one-out mixing `(J−I)/(K−1)` satisfies this, which is why every peer-mean variant collapses onto the same inert fixed point. An *arbitrary* linear function of the trajectories does **not**: a non-doubly-stochastic `A` leaks weight onto some trajectories, `mean(y) ≠ mean'`, and the readout does move. **Design rule: the descent target must be either external to the trajectory set (our anchor `a_t = W_a·hT_t`) or a non-linear function of it (the gate) — a normalized linear peer mix cannot work.**

### Defect 2 — the loss term was minimized by the failure it declares → **fixed by a dimensionless spread floor**

Spec: `loss += 0.1·C` (training term) and "`V → 0` is failure, never success" (monitor). But `C` *is* the variance (the file defined `C = E^(S)` and `V = C` — the same number under two names, while claiming "don't confuse them"). So gradient descent on `0.1·C` drives exactly the quantity the monitor calls failure, and Defect 1 showed 64% of it could be removed with no accuracy change. Prior art is unambiguous that agreement-style objectives collapse diversity (reverse-KL mode-seeking, entropy collapse in RLVR).

**Fix adopted — spread floor.** Define the dimensionless ratio `ρ = V_pre / mean_{t,d} hT²` (fraction of backbone activation energy the K hypotheses still span) and penalize only below a floor:

`L += sandbox_weight · max(0, ρ_min − ρ)²`, with `ρ_min = 0.05` (design weight 0.1; shipped `lambda_sbx = 1.0`).

This is non-degenerate by construction: exactly zero while the hypotheses keep ≥5% of the backbone spread, positive only once they collapse, and comparable across `H`, `T`, and model size (which the raw variance was not). Verified: healthy run ρ 1.95 → 1.22 pays **0**; collapsed run (×0.001, ρ = 1e-6) pays **0.0025**. `ρ` is a monitor, never a term to be minimized directly.

Rejected alternatives, for the record: dropping the term entirely (loses the collapse guard); a diversity-preserving f-divergence (heavier, and unnecessary for a *floor* rather than a reward).

### Defect 3 — the eval plan was not measurable at 12–50M parameters → **fixed by splitting held-out from runnable**

RQ2/RQ3 were specified on GSM8K and HumanEval. TinyGSM (arXiv:2312.09241) states the smallest model that broke 80% on GSM8K was 34B, and needed 1.3B + a verifier to hit 81.5%. A 12–50M model trained on ~1.6B tokens will sit at the floor on both, so "sandbox beats self-consistency" cannot be demonstrated there.

**Fix adopted:** GSM8K/HumanEval are marked *scaling-held-out* (reported, never used as the RQ2/RQ3 decision gate). XS/S run RQ2/RQ3 on tasks with signal at that scale — perplexity, agreement-vs-accuracy on LAMBADA/HellaSwag/ARC-easy, and synthetic multi-step items with answers known by construction, which is exactly the regime where consensus-as-confidence is measurable. Budget/eval table in [14_Performance_Analysis.md](../06-Training-Eval/14_Performance_Analysis.md).

## 4. Related work fetched for the vault

Added to [`papers/Sandbox_Reasoning/`](../papers/Sandbox_Reasoning/) and indexed in [`papers/00_Papers_Index.md`](../papers/00_Papers_Index.md) (all downloaded and byte-verified):

| Tag | Paper | File |
|---|---|---|
| C21 | Burtsev & Rumshisky 2021, Multi-Stream Transformers, arXiv:2107.10342 | `C21_Burtsev_2021_Multi_Stream_Transformers.pdf` |
| C22 | Shazeer et al. 2020, Talking-Heads Attention, arXiv:2003.02436 | `C22_Shazeer_2020_Talking_Heads_Attention.pdf` |
| C23 | Kerce & Fox 2026, Dual-Stream Transformer, arXiv:2603.07461 | `C23_Kerce_2026_Dual_Stream_Transformer.pdf` |
| C24 | Rauba, Fanconi & van der Schaar 2026, Tiny Autoregressive Recursive Models, arXiv:2603.08082 (OpenReview `ikNCYypbT9`) | `C24_Rauba_2026_Tiny_Autoregressive_Recursive_Models.pdf` — **fetched via arXiv after OpenReview returned 403** |
| C29 | Jolicoeur-Martineau 2025, Less is More: Recursive Reasoning with Tiny Networks, arXiv:2510.04871 | `C29_JolicoeurMartineau_2025_Less_is_More_TRM.pdf` — the original TRM that C24 builds on | |
| C25 | Energy Transformer, NeurIPS 2023 | `C25_Energy_Transformer_NeurIPS2023.pdf` |
| C26 | NRGPT, arXiv:2512.16762 | `C26_NRGPT_Energy_Alternative_for_GPT.pdf` |
| C27 | Wang et al. 2025, Mixture-of-Agents, ICLR 2025 | `C27_Wang_2025_Mixture_of_Agents_ICLR.pdf` |
| C28 | Li et al. 2026, arXiv:2510.20817 + ACL 2026 Findings 1266 (entropy collapse) | `C28a_…pdf`, `C28b_…pdf` |

## 4. The strongest counter-evidence: C24's negative result (read this before writing RQ2)

C24 (*Tiny Autoregressive Recursive Models*, arXiv:2603.08082) is the most important prior art in this audit **and
it cuts against us**. Holding block design, token stream, objective and the block-pass budget fixed, it compares:

| Variant | What it is | Outcome (character-level algorithmic tasks, compute-matched) |
|---|---|---|
| Deep Transformer | 12 untied layers | **strongest generalization per block evaluation** |
| Universal Transformer | 1 shared block unrolled 12x (tied params) | flat |
| Single-stream recurrent + step embeddings | tied + iteration index | flat, no breakthrough |
| **Dual / two-stream UT** | split into `(Y, Z)`, weighted readout `∑_k w_k Y_k` | **strong, higher plateau, accelerates earlier** |
| Dual nested UT | `Z` refined `L` inner steps per `Y` update | low |
| Autoregressive TRM | two-stream + Q-halt, nested | **no consistent benefit** |

Three consequences for Ekdant, stated plainly:

1. **Our L2 cannot be justified by analogy to TRM.** C24's own advice is "caution against investing in the
   autoregressive TRM-specific model as a fruitful research direction." Any sentence in the thesis that leans on
   TRM as evidence that latent refinement helps would be using the paper against its own conclusion.
2. **The one encouraging signal is structural, not mechanistic.** What did work was a *flat two-stream split with a
   weighted readout* — which is closer to Ekdant's L2 shape (a second stream feeding a consensus head) than
   to TRM's nested halting. Two-stream is also C23's finding at 29M. Two independent sources therefore point at
   *stream separation*, and that is the part of L2 we should defend; the *recursive-refinement* part is the part
   under test.
3. **This raises the bar for our ablation, and it is a design constraint we should have had earlier.** The
   informative baseline is **not** "sandbox vs. greedy" — it is **"depth-matched": a plain Transformer with
   the same parameter count and the same block-pass budget but no L2**. Otherwise we cannot tell a gain from L2
   against a gain from spending the same compute on more layers, which is exactly what C24 found winning. Add
   `depth-matched-no-sandbox` as a control, and report K=1 as the other.

**Net effect on the novelty claim.** The novelty claim in " + SEC1 + " is unchanged (the M3 write-back and the
after-the-backbone debate placement are still unclaimed), but the *expected result* is now weaker: the honest prior
from the best-matched study in our own library is **"no reliable gain"**. RQ2 should therefore be written as a
falsifiable question with a genuine chance of a negative answer, and the thesis must pre-commit to reporting a tie.

Also already on disk and directly relevant: Coconut / continuous-thought latent reasoning, Latent Agents (IMAD, ACL 2026), AgentArk (arXiv:2602.03955), TinyGSM (arXiv:2312.09241, used in §3 defect 3).

## Connections
- Spec: [22](22_Thesis_Ch3_Layer2_Sandbox.md) · Build: [26c](26c_Internal_Sandbox_Build.md) · Worked numbers: [28](28_Worked_Examples.md) · Open defects: [10_Known_Issues](../07-Management/10_Known_Issues.md) T12–T14 · Papers: [20_Thesis_Ch2_Sandbox_Papers.md](../03-Background/20_Thesis_Ch2_Sandbox_Papers.md)