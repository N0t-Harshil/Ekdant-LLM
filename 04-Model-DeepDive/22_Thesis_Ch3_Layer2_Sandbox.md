# 22 — Layer 2: Sandbox — Problem, Design, and How It Works

> Thesis Ch. 3 §2. Background: [`20_Thesis_Ch2_Sandbox_Papers.md`](../03-Background/20_Thesis_Ch2_Sandbox_Papers.md) · PDFs: [`papers/Sandbox_Reasoning/`](../papers/Sandbox_Reasoning/) · Index: [`papers/00_Papers_Index.md`](../papers/00_Papers_Index.md)
> Code: `src/ekdant/sandbox.py` + `src/ekdant/output_head.py` · Config: [`12_Configuration_Guide.md`](../02-Architecture/12_Configuration_Guide.md) · Eval: [`09_Validation_System.md`](../06-Training-Eval/09_Validation_System.md)

> [!important] **Current status (2026-10-27).** L1 and L2 are **independent** stages (T38). The best result so far is L2 as the **final stage after SmolLM2**, with gate control (v2.3): benchmark perplexity 3.783 → 3.776, ROUGE-L / F1 +0.006 (`run.md` A7d). Placing L2 inside the backbone (after layer 20, or at layers 10/20/30) helped less.

## 0. How the sandbox works — a plain-language walkthrough (as built)

This section explains L2 the way the code runs it in the current experiment
(`configs/v2_sandbox_final.json`: L2 after SmolLM2's last layer). Every number
is the real setting; the code is `09-Code/src/ekdant/sandbox.py` (L2 itself)
and `model.py` (`GatedAddOn`, the way L2 is attached).

**What goes in.** For every token, SmolLM2 produces a vector `h` of 576 numbers —
its final "understanding" of that position, just before it predicts the next
word. L2 works on **each token separately**: it never looks at other tokens
(attention across the text is SmolLM2's job). Its question is: *can a small
team of hypotheses improve this one vector before the prediction is made?*

**Step 1 — make 4 different hypotheses (`DiverseInit`, 1.33M parameters).**
`h` is multiplied by **K = 4 different matrices**, giving 4 versions of the
token's state, called *trajectories* `T_1 … T_4`. The matrices are built to be
mutually orthogonal and slightly different in scale (gains 0.8–1.25), so the
four start out genuinely different. If they started identical, the debate
below would have nothing to argue about.

**Step 2 — debate, 4 rounds (`DebateRound` ×4, 4.32M parameters each).** In
each round, every trajectory:
1. is normalised (pre-norm, T33), so no round can blow up the next;
2. **reads the other three** with a small attention (query/key/value, 1.0M
   parameters): "what do the others think?" — it never attends to itself;
3. passes what it read through an MLP (576 → 2,304 → 576, 2.66M parameters)
   to form a proposal;
4. decides **how much to accept** with a gate `γ` between 0 and 1, computed
   from its own state and what it read (starts at about 0.12: cautious):
   `T_k ← (1 − γ)·T_k + γ·MLP(read_k)`.

After 4 rounds each hypothesis has been revised by what the others said, but
each one could also refuse (γ ≈ 0) — that is what lets a correct minority
survive.

**Step 3 — an external reference point (the anchor, 0.33M parameters).**
`a = W_a · h`: one more learned projection of the **original** backbone state.
It is deliberately *not* the average of the hypotheses (pulling towards their
own average provably changes nothing — Defect 1, note 29).

**Step 4 — careful descent towards the anchor, 8 steps (`AnchoredDescent`).**
Each step moves every trajectory a little towards the anchor:
`T_k ← T_k − 0.0625 · g_k ⊙ (T_k − a)`, where `g_k` is a per-trajectory,
per-number gate between 0 and 1. Because each trajectory has its own gate,
they move by different amounts — so their average genuinely changes, and the
disagreement shrinks without forcing them into one point.

**Step 5 — consensus.** The final answer of the team is the **average of the
4 trajectories** (`readout: mean`; `lowest_energy` instead picks, per token,
the trajectory closest to the anchor).

**Step 6 — health checks (no effect on the answer).** Three numbers are
logged: `anchor_energy` (how far the team still is from the anchor),
`trajectory_variance` (how different the hypotheses still are after the
debate) and `spread_ratio` ρ (that variance relative to the backbone's own
size). If ρ falls below 0.05 the hypotheses have **collapsed** into one, and
a small penalty `(0.05 − ρ)²` is added to the loss — the only loss term L2 has
of its own.

**Step 7 — attaching L2 to SmolLM2 (`GatedAddOn`).** The consensus is
normalised (RMSNorm, T32) and mixed into the backbone's state through a gate:
`h' = h + g ⊙ (RMSNorm(consensus) − h)`, then SmolLM2's own head predicts the
next word from `h'`.
- `g` has one value per number (576) and **starts at exactly 0**, so at the
  start the model *is* SmolLM2 — any later difference comes from training L2.
- `g = tanh(θ)`: it can never exceed 1 (T37).
- **Gate control (T37):** the gate learns 10× faster during the first 1,000
  steps (boost); if L2's gradient spikes (> 3× its average) or its internal
  size `l2` exceeds 50, the brake closes the gates a little (×0.9) and halves
  L2's learning rates, recovering slowly.

**Cost.** L2 adds 19.59M parameters (12.7 % of the model) and roughly
`K × (R debate + S descent)` extra work per token — the end placement took
14 % longer to train than the baseline.

### What the experiments taught us about how L2 behaves

| observation | evidence | meaning |
|---|---|---|
| L2 helps most as the **final** stage | v2.2: end −0.0017 > three depths −0.0009 > middle −0.0003 (per-token NLL vs baseline) | it works as a refinement close to the output |
| the gate opens only as far as L2 helps | gates 0.03–0.13; after the boost faded the network closed them partly, then reopened them on MedQuAD | L2 helps more on MedQuAD than on PubMedQA |
| without normalisation inside, L2's size compounds and explodes | T32–T34 | the pre-norm, output norm, low L2 LR and brake are load-bearing — keep them |
| training longer hurts both models | v2.1: 2 PubMedQA epochs → 3.914 vs 3.782 | more epochs is not a lever |
| the gate controller roughly tripled L2's effect | end placement: −0.0006 without, −0.0017 with | how fast L2 *learns* matters more than how big it is |

### Levers you can change for a better result

Each lever is a single change; compare it against the existing baseline run
(`runs/v22_baseline`, seed 17, 1 epoch) — only the sandbox run is needed, about
1 h 30 min each. Change one thing at a time.

| # | lever | how | why it might help | risk |
|---|---|---|---|---|
| 1 | **L2 learns faster** | `--addon-lr 3e-4` (now 1e-4) | L2's matrices still learn slowly; the effect grew when the *gate* learned faster. The brake and norms now guard against the old drift | drift returns — watch `l2=` and `brake=` |
| 2 | **more hypotheses** | `"n_sandbox": 8` in a copy of `v2_sandbox_final.json` | more diverse candidates per token; only +1.33M params (20.92M) but ~2× L2 compute | slower; may add little |
| 3 | **pick instead of average** | `"readout": "lowest_energy"` | the closest-to-anchor hypothesis may be sharper than the average | can be noisier |
| 4 | **more refinement** | `"descent_steps": 16` (η = 0.0625 kept, or `"eta": 0.03125`) | more careful convergence towards the anchor | over-smoothing |
| 5 | **gate boost again in phase 2** | needs a small code change | the gate reopened during MedQuAD, where L2 helps most | — |
| 6 | **confirm the result first** | baseline + end with `--seed 23` | the current gain is one seed; a second seed shows it is real | none — recommended before any lever |

## 1. Problem: one pass that hides its reasoning and misstates its confidence

A single autoregressive pass produces one answer with one probability distribution. When it fails, nothing records which step went wrong; when it succeeds, its softmax score still does not mean "I am sure." L2 exists to make reasoning inspectable and confidence observable: keep several live hypotheses, let them confront each other, and read agreement across them as the confidence signal.

## 2. Why parallel trajectories with debate and consensus

The lineage, in the order the thesis uses it:

1. **Reasoning benefits from intermediate steps** — chain-of-thought is the K = 1 control every later method must beat (PDF `papers/Sandbox_Reasoning/C1_Wei_2022_Chain_of_Thought.pdf`).
2. **Several paths plus a vote beat one greedy path** — self-consistency sampling with majority vote is the matched-budget baseline for every sandbox claim (PDF `C2_Wang_2022_Self_Consistency.pdf`).
3. **Letting the paths read and revise each other helps further** — multi-agent debate (propose → debate → revise over rounds) improves reasoning and factuality, with both agent count and rounds mattering (PDF `C3_Du_2023_Multiagent_Debate.pdf`, the primary debate blueprint). Diversity of starting points drives the gains; ordering and visibility tweaks do not (PDFs `C19_Liang_2023_Divergent_Debate.pdf`, `C9_Wu_2025_Can_Agents_Really_Debate.pdf`).
4. **Search structure matters** — tree and graph formulations of thought show that keeping live alternatives with aggregation beats resampling only at the end (PDFs `C4_Yao_2023_Tree_of_Thoughts.pdf`, `C13_Besta_2024_Graph_of_Thoughts.pdf`); decomposing hard problems into easier subproblems works without any planner module (PDF `C17_Zhou_2022_Least_to_Most.pdf`).
5. **Reflection compresses into refinement** — verbal reflect-and-retry loops work, but the thesis folds that loop into latent refine steps rather than a separate module (PDFs `C6_Shinn_2023_Reflexion.pdf`, `C7_Madaan_2023_Self_Refine.pdf`).
6. **Refinement has an energy view** — calibrating latent thoughts toward low-energy, high-consistency regions lets one refined path rival a many-path ensemble (PDF `C11_Energy_Calibration_Implicit_CoT_2025.pdf`, the primary energy blueprint).
7. **Supervision should be per-step, not just per-answer** — process feedback beats outcome-only feedback for reranking math reasoning (PDFs `C12_Lightman_2023_Lets_Verify_Step_by_Step.pdf`, `C16_Uesato_2022_Process_Feedback.pdf`), and high-consistency trajectories can become training targets themselves (PDFs `C14_Huang_2022_Self_Improve.pdf`, `C15_Zelikman_2022_STaR.pdf`).
8. **Cost honesty is load-bearing** — debate systems often tie voting ensembles at equal budgets and are tuning-sensitive, so every sandbox result is reported cost-matched with tokens and FLOPs, never against greedy alone (PDF `C8_Smit_2024_Should_We_Be_Going_MAD.pdf`).
9. **Evaluation and efficiency close the loop** — debate among judges improves evaluation quality (PDF `C18_Chan_2023_ChatEval.pdf`); rationale-aware early stopping cuts sampling cost (PDF `C20_Wan_2025_RASC.pdf`); distilling explicit debate into latent subspaces is the stated future work (PDF `C10_Yi_2026_Latent_Agents_ACL.pdf`). Acting with tools is out of scope by design (PDF `C5_Yao_2022_ReAct.pdf` is cited to draw that boundary).
10. **Closest architectural priors — cite these beside the contribution, not in a distant related-work section** — parallel hypothesis streams inside one model (Multi-Stream Transformers, Burtsev & Rumshisky 2021); streams that read and revise each other (Talking-Heads Attention, Shazeer et al. 2020); function-separated streams at small scale (Dual-Stream Transformer 2026); iterative latent refinement with weighted readout (TRM, arXiv:2510.04871; Coconut); energy descent as the computation (Energy Transformer NeurIPS 2023, NRGPT, Hyper-SET); layered proposal/aggregation (Mixture-of-Agents, ICLR 2025). The closest prior is also the strongest evidence against us: Tiny Autoregressive Recursive Models (`C24_Rauba_2026_Tiny_Autoregressive_Recursive_Models.pdf`, arXiv:2603.08082) compares compute placement against a weighted readout `∑_k w_k H_k` and reports **no reliable gain** for the full Autoregressive TRM — see [29_Prior_Art_and_Novelty_Check.md](29_Prior_Art_and_Novelty_Check.md) §4. Full table with what each leaves open for Ekdant: [29_Prior_Art_and_Novelty_Check.md](29_Prior_Art_and_Novelty_Check.md) §1. Every ingredient therefore has prior art; the novelty claimed is the *combination* plus the M3 write-back loop.

## 3. How L2 works, end to end

Shapes with thesis defaults (K = 4 paths, R = 2 debate rounds, S = 8 energy steps):

1. **Diverse initialization:** `trajectories = init_K(hT) [K,B,T,H]` — K different projections of the backbone-enhanced states, deliberately divergent (uniform copies would collapse; diversity is load-bearing).
2. **Debate (×R):** each trajectory cross-attends to the others and revises itself. Ablation: rounds 0 / 2 / 3.
3. **Anchored gated descent (×S):** each trajectory descends the energy `½‖T_k − a_t‖²` toward the token-conditioned anchor, through a learned per-trajectory gate. Moves the readout (unlike a peer-mean pull) and shrinks disagreement without collapse. Ablation: steps 0 / 4 / 8.
4. **Consensus:** mean of trajectories (default) or per-position lowest-anchor-energy pick → `y [B,T,H]`. Three quantities ship with it and are **not** interchangeable: `anchor_energy` E (monitor; the descent target), `trajectory_variance` `V_pre` (monitor; did diversity survive the debate rounds), `spread_ratio` ρ = `V_pre / mean(hT²)` (monitor; dimensionless, comparable across H/T/scale). The **only** L2 training term is the floor `max(0, ρ_min − ρ)²`.
5. **Head:** `logits = vocab_head(y)` — one dense `[H,V]` projection per micro-batch, trained with causal cross-entropy.

## 4. Configuration (what each knob does)

| Knob (thesis default) | Role |
|-----------------------|------|
| `n_sandbox_paths = 4` | K trajectories; ablate 1 / 4 / 8 (K = 1 is the CoT control) |
| `n_debate_rounds = 2` | Cross-reading depth; ablate 0 / 2 / 3 |
| `sandbox_steps = 8` | Anchored gated descent iterations. Moves the readout (verified); ablate 0 / 4 / 8 |
| `sandbox_weight = 0.1` | Weight on the **spread floor** `max(0, ρ_min − ρ)²` — not on raw agreement (tune 0.05–0.3). Design value; the shipped configs use `lambda_sbx = 1.0` |
| `sandbox_rho_min = 0.05` | Minimum fraction of backbone activation energy the K hypotheses must keep spanning; collapse threshold |
| `lm_weight = 1.0` | Next-token loss weight (fixed) |

## 5. How L2 is tested (RQ2 + RQ3)

1. **Cost-matched comparisons only:** sandbox vs. greedy vs. self-consistency-majority at equal FLOPs/tokens — never sandbox-vs-greedy alone. **Scale caveat:** at 12–50M params GSM8K/HumanEval sit at the floor (the smallest model reported to pass 80% on GSM8K is 34B, arXiv:2312.09241), so XS/S must additionally run RQ2/RQ3 on signal-bearing tasks — perplexity, ARC-easy/HellaSwag/LAMBADA, and synthetic multi-step items with known answers; GSM8K/HumanEval are scaling-held-out ([29](29_Prior_Art_and_Novelty_Check.md) §3 defect 3).
2. **Ablations:** K, rounds, steps, diverse-vs-copy init, mean-vs-lowest-energy consensus.
3. **Debate dynamics:** per-round overturn table (wrong→right vs. right→wrong) to catch majority pressure suppressing corrections.
4. **Calibration (RQ3):** trajectory agreement vs. accuracy — the plot that decides whether consensus works as a confidence signal.
5. **Reranking:** step-energy rerank compared against final-vote; rationale-aware early stopping for the efficiency table.

## 6. Math reference (implement L2 from this — the full how-it-works)

Shapes: `hT [B,T,H]` in; K=4, R=2, S=8; all ops plain torch (build recipe → [26c](26c_Internal_Sandbox_Build.md)).

- **Init (diverse by construction):** `T_k^(0) = W_k·hT + b_k`, k = 1..K, with `W_k` orthogonal-distinct and spread gains. Uniform copies collapse — diversity is load-bearing (C19).
- **Debate round r = 1..R (per trajectory k):** queries read the others —
  `Q_k = T_k·W_Q`, `K_j = T_j·W_K`, `V_j = T_j·W_V` for j≠k;
  `read_k = softmax(Q_k·K_{−k}ᵀ / √d_h)·V_{−k}`;
  repair gate `g_k = σ(W_g·[T_k ; read_k] + b_g)`;
  revise `T_k ← (1 − g_k)·T_k + g_k·MLP(read_k)`.
  K ≤ 8, so this cross-attention is trivially cheap — no kernels, no frameworks.
- **Anchored gated descent (AGD), s = 1..S — the replacement for the broken peer-mean step:**
  token-conditioned anchor `a_t = W_a · hT_t` (learned, per position);
  energy `E_k = ½‖T_k − a_t‖²`;
  per-trajectory, per-channel gate `g_k = σ(W_g[T_k ; a_t] + b_g) ∈ (0,1)^H`;
  step `T_k ← T_k − η · g_k ⊙ (T_k − a_t)`, with `η ≈ 0.5/S`.
  > [!important] **Why this one is not inert (unlike the peer-mean version it replaces).** Descent toward the *running mean of the trajectories* is provably mean-preserving (`Σ_k(T_k − η(T_k − c)) = Kc`), so the readout never moved — proved and quantified in [29](29_Prior_Art_and_Novelty_Check.md) §3.1. AGD fixes both causes at once: the target is **external** (the backbone state at that position, not the peers), and the gate **differs across k**, so `Σ_k g_k ⊙ (T_k − a) ≠ 0` in general and the consensus `c^(S)` genuinely moves. Verified numerically: consensus moved L2 = 0.2295 and closed 0.2199 of its distance to the anchor, while per-trajectory deviation retention *differed* across k (0.744–0.827) — i.e. refinement without forced collapse.
  > **Precisely which pulls are inert** (the loose version of this claim is false). For `T_k → T_k + η(y_k − T_k)`, averaging over k gives `mean' = mean + η(mean(y) − mean)`, so the consensus moves **unless** the targets themselves average to the current mean. That holds exactly when the target matrix is **doubly stochastic**: `y_k = ∑_j A_kj T_j` with rows summing to 1 (each trajectory a convex mix of the peers) *and* columns summing to 1 (no trajectory over-weighted). Leave-one-out mixing `(J−I)/(K−1)` qualifies, which is why every peer-mean variant lands on the same inert fixed point. An arbitrary linear function of the trajectories does **not** qualify: a non-doubly-stochastic `A` gives `mean(y) ≠ mean` and the readout does move. **Design rule: the target must be external to the trajectory set (our `a_t`) or a non-linear function of it (the gate) — a normalized linear peer mix cannot work** (precedents for external/energy targets: Energy Transformer, NRGPT — C25/C26).
- **Consensus:** `y = c^(S)` (mean, thesis default) or lowest-anchor-energy pick `argmin_k ‖T_k − a_t‖`. Then `logits = vocab_head(y)`, one dense `[H,V]` projection.
- **The three reported quantities (distinct on purpose):**
  - `anchor_energy` `E = mean_{k,t} E_k` — monitor, the quantity AGD descends. Not a loss.
  - `trajectory_variance` `V_pre` — mean squared distance to consensus, measured **after the debate rounds and before descent**. This is the honest measure of whether diversity survived cross-examination.
  - `spread_ratio` `ρ = V_pre / mean_{t,d} hT²` — **dimensionless**: the fraction of backbone activation energy the K hypotheses still span. Collapse is `ρ → 0`, and it is comparable across H, T, and scale.
- **Total loss (revised):** `L = lm_weight · CE(shift(logits), shift(labels)) + sandbox_weight · max(0, ρ_min − ρ)²`. As shipped: `lambda_sbx = 1.0`, plus the optional answer-letter CE `answer_weight · CE(logits[answer_pos−1], letter)` applied to both arms (T22).
  The second term is a **floor**, not an agreement penalty: it is exactly zero while the hypotheses keep at least `ρ_min` of the backbone's spread, and it grows only once they collapse. That makes it non-degenerate by construction — a bare `0.1·C` agreement term was minimized *by* the collapse the design forbids ([29](29_Prior_Art_and_Novelty_Check.md) §3.2). Checked numerically: a healthy run (ρ 1.95 → 1.22) pays 0; a collapsed run (ρ = 1e-6) pays `(0.05 − 1e-6)² = 0.0025`.
- **RQ3 calibration:** agreement `â = 1/(1+V)` plotted against per-example accuracy — the plot that decides whether consensus works as confidence.
- **Overturn accounting (per round):** count wrong→right vs. right→wrong transitions; majority pressure suppressing corrections shows up here first (C9).
- **Cost:** `O(R·K²·T·H + S·K·T·H)` — quadratic only in tiny K, never in T; always report FLOPs/tokens beside accuracy (C8 rule).
- Worked numbers: [28_Worked_Examples.md](28_Worked_Examples.md) §2–3 (K=3 debate + RoPE by hand).
- **Independent of L1 (T38):** L2 reads only the backbone's states and writes only to the head (or, in the v2.1/v2.2 extensions, back into the residual stream of the same backbone). Nothing flows into L1; the earlier M3 write-back was removed from the design.
