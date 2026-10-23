# Ekdant — Project Explanation

**Harshil Gorasiya** · MSc Thesis, Hochschule Heilbronn
Document: [thesis.pdf](08-Thesis/thesis.pdf) · Source: [thesis.tex](08-Thesis/thesis.tex) · Outline: [thesis.md](08-Thesis/thesis.md)

> **What is built vs. designed.** The evaluated model is **T + L2** (embed → windowed dense
> Transformer → parallel sandbox → head), ~50M parameters, trained from scratch on MedMCQA then
> MedQA, as an architecture-matched A/B against the identical backbone without L2. **L1 (SSM
> storage) and the M3 write-back are fully specified but not implemented** — the memory
> numbers below are projections for the full design, and RQ1 is future work.
>
> This file explains the whole project end to end — why the two layers exist, how
> they work, why they are separate, how they help, and how all the pieces connect.

---

> [!important] **Status 2026-10-22 — v2.** The thesis now tests L2 on a pretrained SmolLM2-135M: two arms (`baseline` 134.52M, `sandbox` +19.59M L2), both starting identical (zero-init gate), trained on PubMedQA then half of MedQuAD, benchmarked on the other half by reference-answer perplexity with a 95 % confidence interval. Built and tested (141 tests); GPU runs pending. Why: the v1 from-scratch 50M model learned only the answer-letter prior (T27, T28). Sections below describe the design and the v1 pilot. Spec: [30](07-Management/30_Design_v2_Pretrained_Backbone.md).

## 1. The pitch

Transformers must hold a KV cache that grows linearly with every token you add, so
the model is most expensive exactly where reasoning is needed — long codebases,
long records, long papers. Ekdant has **two layers**. **L1** compresses any sequence
into a fixed-size selective SSM state. **L2** is a parallel sandbox that runs several
hypotheses against each other and refines them with a provable descent step. A small
standard Transformer between them only bounds the attention window. L1 answers
*what was said*, L2 answers *what to do*, and they are wired together by a single
write-back edge. For the full design (with L1), at 12M / 21M / 46M parameters the inference state is projected to stay **flat at
1.58 MB** regardless of sequence length, and every design claim is backed by either a
proof, a locally stored paper, or a number.

---

## 2. Why these layers: three problems, two layers

We work on **two layers, and only two**:

- **L1 — Selective SSM storage** — the *memory* answer
- **L2 — Parallel sandbox** — the *reasoning* answer

Two supporting components exist but are **not** layers of their own: the embedding /
output tables are tied to the head, and **T** is a plain windowed dense Transformer
that sits between them. T is standard, not a contribution — its only job is to bound
the KV cache (window W = 512) and give local exact retrieval. It is never the subject
of a claim.

The thesis states three failure modes separately, **because each has a different,
non-interchangeable fix**:

| | Problem | Answered by |
| --- | --- | --- |
| **P1** | **Memory grows with the sequence.** Attention is quadratic in training; the KV cache grows linearly at inference. | **Layer 1 — L1, the selective SSM.** Compresses any sequence into a fixed-size recurrent state. |
| **P3** | **Agreement is not confidence.** Sampling several paths and trusting their agreement has a degenerate optimum: make them agree and you are confidently wrong. | **Layer 2 — L2, the sandbox + spread floor.** Makes the mechanism that *produces* agreement incapable of benefiting from collapse. |
| **P2** | **A monolithic stack cannot be audited.** You cannot tell whether a gain came from added capacity or from an added mechanism, so the claim is unfalsifiable. | **The separation between layer 1 and layer 2.** Because they are two distinct layers, either can be removed, replaced, or ablated independently at matched parameters and matched compute. |

**So: P1 → layer 1, P3 → layer 2, P2 → the fact that they are two layers and not one.**
P2 is not a third layer — it is the reason the two layers must not be merged. L1 and
L2 are not two attempts at the same job: they solve different, incompatible problems,
which is precisely why they are two layers rather than one.

Two supporting decisions fall out of the same reasoning:

- **Why a dense windowed T between them, and not more attention** — T's job is only to
  bound the KV cache (window W = 512) and give local exact retrieval. It is not asked
  to reason or to remember long-range.
- **Why no MoE** — at 12–50M parameters, MoE adds routing machinery without the
  capacity to exploit it. L2 spends the same parameters on *hypothesis diversity*
  instead. The sandbox is the mixture mechanism.
---

## 3. The two layers in plain words

This is the version to say out loud. No equations.

### Layer 1 - SSM storage: the memory

**The problem.** A normal transformer remembers by keeping a list - one entry for
every word it has read. Read 10,000 words, hold 10,000 entries. The list only ever
grows, and the model gets more expensive exactly when you give it more to read.

**What L1 does instead.** It keeps a small, fixed notebook - a handful of numbers -
and rewrites that notebook as it reads. However long the text, the notebook stays the
same size.

**How it decides what to keep.** For every word it reads, a small network answers
three questions: *how strongly should I write this in?* (the step size), *what exactly
do I write?* (the input term), and *how do I read it back?* (the readout). Those
answers depend on the word itself, so it highlights what matters and skips the filler.
That is what **"selective"** means.

**Why three notebooks, not one.**

- **content** - written slowly; keeps **what was said**, over the whole document
- **control** - written fast; keeps **what to do** (instructions must update quickly)
- **sandbox** - written at medium speed; this is the one the reasoning layer writes
  its conclusion back into

**The honest part.** This memory is lossy: you get the gist, not a photocopy. That is
a known property of this whole family of models, and the thesis states the bound
rather than hiding it.

**The number that matters.** At our largest size the entire L1 memory is **6 KB**, and
it does not grow with the length of the text.

### Layer 2 - the sandbox: the reasoning

**The problem.** A normal model emits one answer, with no record of which step went
wrong. A common fix is to sample several answers and trust them only when they agree -
but push them to agree and the model becomes confidently wrong.

**What L2 does instead.** It keeps **four hypotheses at once** (K = 4) and runs them
against each other.

**Step 1 - start different.** The four hypotheses come from four *deliberately
different* projections of the backbone. If they started identical, the debate would be
theatre.

**Step 2 - debate, two rounds.** Each hypothesis reads the other three - never itself -
and may revise. A gate lets one **decline the crowd**, so the group can actually
disagree instead of merely averaging.

**Step 3 - refine, eight steps.** Each hypothesis is pulled toward a common target, the
**anchor**. The anchor is computed from the backbone, i.e. from **outside** the group
of four.

**Step 4 - combine.** Average the four, and one shared head produces the answer. Four
hypotheses, one answer head - that is why it stays cheap.

**Why the anchor is the whole point.** The obvious version pulls each hypothesis toward
the *average of the four*. We proved that this changes nothing at all: the average is
preserved exactly, so the output after 8 steps is bit-identical to the output before
any refinement. It is a refinement loop that provably cannot refine. Pulling toward a
target that lives outside the group is what fixes it.

**Why a floor and not a bonus.** If we *reward* agreement, the model wins by making
all four agree - and then you are confidently wrong. So we do not reward agreement. We
only flag collapse: if the four have collapsed onto each other, a floor term switches
on. In a healthy run it is exactly zero and contributes no gradient at all.

> **One sentence to remember:** Layer 1 remembers the document in a fixed 6 KB.
> Layer 2 thinks about it four different ways, argues with itself, and refines toward
> a target that is not the group's own opinion.

> **Thirty seconds, spoken:** "L1 replaces the ever-growing memory of a transformer
> with a fixed 6 KB notebook that only writes what matters. L2 replaces a single
> answer with four competing hypotheses that debate each other and then refine
> toward an external target - because refining toward their own average is provably
> a no-op. The two are separate layers so each can be ablated, and they are joined
> by a single write-back edge."

---

## 4. How L1 (SSM storage) works — the memory answer

**Input → output.** `input_ids [B,T]` → tied embedding `[B,T,H]` + rotary positions
→ **N** aSSM blocks → `h [B,T,H]`, plus a pooled `compressed [3·d_state]` handed to T.

**The selection mechanism.** A shared trunk
`u_t = DWConv(SiLU(Linear(h_t)))` feeds a network that emits **per-channel step
sizes**. Everything else follows from those:

```
Δᵗᶜ = softplus(W_Δ,c uₜ + b_Δ,c) + ε          ← selected per channel and step
B̄ᵗᶜ = Δᵗᶜ Bᶜ(uₜ)                              ← input-dependent injection
Āᵗᶜ = exp(Δᵗᶜ A_c),   A_c = −softplus(A_log,c) ← stable, contractive state
xᵗᶜ = Āᵗᶜ ⊙ xᵗᶜ₋₁ + B̄ᵗᶜ ⊙ uₜ,   yᵗᶜ = Cᶜ(uₜ) ⊙ xᵗᶜ
```

Because `Δ` and `B` depend on the *current input*, the layer selectively writes
some inputs to memory and skips others — that is what "selective" means, and it is
the property that Mamba-style SSMs introduced over the fixed-rate recurrence.

**Three stores, three rates.** Instead of one state, each block keeps three, each
with its own `A, Δ, B, C`, recurrence, readout, gate and norm:

| Store | Step regime | Function |
|---|---|---|
| `S^content` | slow Δ | retains **what was said**; long memory over discourse |
| `S^control` | fast Δ | retains **what to do**; instruction/task features that update quickly |
| `S^sandbox` | medium Δ | **tracks stage L2**; scans distilled task features plus the L2 write-back `f` |

> **Naming caution — two layers share a word.** Stage **L2 is the sandbox** (the
> debate layer). `S^sandbox` is one of the three state stores **inside L1** — a
> fixed-size SSM memory, not a layer. They are separate stages; their only coupling
> is the M3 write-back. Elsewhere in the thesis, "sandbox" alone always means the
> L2 layer.

The three outputs are fused by a learned gate and a residual:
`h_t = RMSNorm(Σ_c gᶜₜ ⊙ yᶜₜ + h^in_t)`.

**Why it helps.** Training is O(T) per channel via the associative parallel scan
(same algorithm as S5). Inference holds a fixed state — at configuration M,
**6 KB total** (`3 stores × 256 × 2 bytes × N = 4 layers`) — regardless of how many
tokens were processed. No KV cache is required from this stage at all.

**Honest bound.** Pure SSMs trail attention on exact recall (this is the Jamba
result, cited in the thesis). The thesis stays pure-SSM *for falsifiability* and
reports the bound rather than hiding it.

---

## 5. How L2 (sandbox) works — the reasoning answer

**Input.** The backbone state `hT` — four independent entry points, in order:

**1. Diverse initialization** — `T⁰ₖ = Wₖ · hT + bₖ`, k = 1…K (K = 4), with the
`Wₖ` mutually orthogonal so the hypotheses start apart instead of agreeing.

**2. Debate rounds** (R = 2). Each trajectory reads the *others* (not itself):
`readₖ = softmax(Qₖ K₋ₖᵀ / √d) V₋ₖ`, then a repair gate lets it decline the crowd:
`Tᵏ ← (1 − γₖ) ⊙ Tᵏ + γₖ ⊙ MLP(readₖ)`. Cross-attention here is over K−1 ≤ 7 keys,
so it is negligible cost.

**3. Anchored gated descent (AGD)** — S = 8 steps. This is the core contribution:

```
aₜ = Wₐ · hTₜ                    ← anchor: computed OUTSIDE the trajectory set
Eₖ = ½‖Tᵏ − aₜ‖²                  ← energy
gₖ = σ(W_g [Tᵏ; aₜ] + b_g)        ← per-trajectory, per-channel gate, ∈ (0,1)^H
Tᵏ ← Tᵏ − η gₖ ⊙ (Tᵏ − aₜ),  η ≈ 0.5/S = 0.0625
```

**4. Consensus → readout.** `y = T̄⁽ˢ⁾`, then `logits = y W_vocab` from a **single**
dense head shared by all trajectories.

**Why the anchor, not the average?** The obvious design descends each trajectory
toward the running mean of the trajectories. We prove (Proposition 1) that this is
*readout-invariant*: the mean is preserved exactly, so `logits(S=0)` and
`logits(S=8)` are bit-identical — gradient descent that provably cannot change the
output. The fix is a design rule: **the descent target must be external to the
trajectory set, or a non-linear function of it.** AGD satisfies both — `aₜ` is
external, and `gₖ` is non-linear and differs across k.

**Monotone descent is proven.** With `0 ≤ ηgₖ ≤ 2`, `Eₖ` is non-increasing, strict
whenever `ηgₖ ∈ (0,2)` and `Tᵏ ≠ aₜ`. Verified numerically: 0 violations in 8 steps.

**How it helps.**
- Makes reasoning *visible* — you can log `anchor_energy`, `trajectory_variance`,
  `spread_ratio` at every step.
- `ρ` is dimensionless (`V_pre / mean(hT²)`), so it is comparable across width,
  length and training scale.
- The only L2 training term is a **floor**, `max(0, 0.05 − ρ)²`, weight 0.1 — exactly
  zero while hypotheses remain diverse (so it injects no gradient in healthy runs),
  bounded at 0.0025 under total collapse (so it can *detect* collapse without ever
  being able to *cause* it).

---

## 6. How they connect — and why they must stay separate

**They are two layers, not one combined component.** The forward pass is a single
chain with no external calls:

```
input_ids → L1 (SSM storage) → T (dense local attention) → L2 (sandbox) → head → logits
```

L1 = memory (what was said). L2 = reasoning (what to do). Different jobs,
different failure modes, each independently ablatable at matched parameters — that
independence is contribution 1 and it is what makes RQ2 falsifiable.

**The only coupling is the M3 write-back** (the dashed red edge in Fig. 2):

```
fₜ = Linear_fb([yₜ ; Vₜ])
x³ₜ = Ā³ₜ ⊙ x³ₜ₋₁ + B̄³ₜ ⊙ uₜ + F ⊙ fₜ₋₁
```

L2's consensus is written into L1's third store `S^sandbox`, so reasoning feeds
memory. Two constraints are enforced, not negotiable:

1. **Causality** — position `t` only reads states at positions `< t`.
2. **No cross-batch leakage** — during training `f` arrives *detached* from the
   previous micro-batch. Stated openly, because the alternative silently conditions
   on future minibatch content.

Without the write-back, L2 would be a terminal head and reasoning would leave no
trace in memory. Without L1, there would be no bounded context for the debate to
occur in.

---

## 7. The four defects we found and fixed

Every one was found by internal review **before** the main runs, each with a proof,
a numerical demonstration, and a retained ablation that exhibits the broken
behaviour. Lead with this section; it demonstrates method.

| # | Defect | What we proved | The fix |
|---|---|---|---|
| **1** | The energy loop was **readout-invariant** | Peer-mean descent (any doubly-stochastic mixing) preserves the mean exactly — `logits(S=0) == logits(S=8)`. Spread falls to **0.356** at η = 0.0625 while the output never changes. | **AGD**: external anchor + per-trajectory gate. Retained as a contrast ablation (Fig. 7). |
| **2** | The agreement loss was **minimized by collapse** | `min C = 0` is attained by collapse; the optimizer is rewarded for defeating the very term meant to detect it. | **Floor**: `max(0, ρ_min − ρ)²` — non-degenerate, minimum on the non-collapsed set. |
| **3** | The evaluation plan was **not measurable at this scale** | GSM8K/HumanEval are scaling-held-out: the smallest model reported past 80% GSM8K is 34B. | RQ2 decided on MedMCQA/MedQA letter accuracy (runnable at 50M); ties declared publishable. |
| **4** | **Ambiguous discretization convention** | Our own worked example cited exact ZOH while computing the first-order form — a **43% discrepancy** on that example. | Resolved explicitly in the discretization section; the general lesson is that specification and implementation must agree on which convention is in force *before* any number is quoted. |

**The strongest demonstration (Fig. 7).** Both variants' energy decreases — the
*broken* one decreases faster. So energy alone cannot detect the disconnection.
Only the readout differs: AGD displaces the consensus by 0.2295, peer-mean leaves it
bit-identical to S = 0.

**Why gate heterogeneity matters too (verified numerically).** Had every gate been
identical, spread would contract by exactly `(1−η)²ˢ = 0.356`. Because gates differ
across trajectories, one term in the deviation survives and never contracts —
`Var_k(λ_k) · |u|²` — so heterogeneous gates convert descent pressure into
persistent spread instead of collapse. Measured retention: **0.356** identical gates,
**0.60–0.75** heterogeneous, matching the reported healthy run (`ρ: 1.95 → 1.22`,
factor 0.626). Gate heterogeneity is load-bearing twice over: it is what makes the
readout move *and* what holds `ρ` clear of the floor.

---

## 8. What helps us — numbers, RQs, and honest status

**Memory (RQ1) — the headline, for the full design with L1 (not built).** At configuration M (H = 512, N = 4, N_T = 6,
d_state = 256, 2 KV heads, bf16):

| | T = 2048 | T = 65536 |
|---|---|---|
| Full attention (no window) | 6.29 MB | 201.33 MB |
| Ekdant (W = 512 + L1 state) | **1.58 MB** | **1.58 MB** |
| Ratio | **4.0×** | **127×** |

Ekdant is flat: a 6 KB L1 recurrent state plus a KV window that saturates at
W = 512. *Analytic projection, not a measurement.* Weights, activations and the
K-trajectory scratch space are excluded — the figure isolates the one term the
architecture is designed to make constant.

**Size budget (design-time, full model).** XS ≈ 12.4M · S ≈ 21.0M · M ≈ 46.0M. L2 rose from 0.6/1.0/2.2M to
0.8/1.3/2.8M when peer-mean was replaced by AGD — the anchor `W_a` and gate `W_g`
are new parameters, and that increase is the honest cost of the correction.
*Closed-form estimates. What is built is the 50M A/B pair: sandbox **50.09M** vs baseline **43.53M**, with L2 measured at **6.56M** vs this 2.8M estimate — the thesis reports the measured numbers.*

**The three research questions:**
- **RQ1 (memory) — future work** — Does a fixed-size SSM state replace the KV cache
  while preserving long-range recall, and what does it cost? Needs L1; the
  protocol is pre-specified and not run.
- **RQ2 (reasoning)** — On MedMCQA/MedQA letter accuracy, does the same backbone
  with K debating trajectories beat the identical backbone with *no sandbox at
  all*? The built A/B is architecture-matched (+6.56M params), so a depth-/parameter-
  matched control is the follow-up that makes a gain claim airtight.
- **RQ3 (honesty)** — Is trajectory agreement calibrated against accuracy, and can
  collapse be detected early by a monitored statistic?

**Honest status — what is done and what is not.**

| Done | Not done |
|---|---|
| ✅ Full thesis source (`thesis.tex`): 7 figures, 70 references, 4 propositions with proofs, no undefined references; `thesis.pdf` must be rebuilt after the scope alignment | ❌ Model source code: T + L2 (no L1) (`09-Code/src/ekdant/`, embed → T → L2 → head); L1 SSM storage still design-only |
| ✅ Model code + **141 unit tests passing** (`09-Code/tests/`); architecture-matched A/B configs ready; **full 3-epoch A/B executed** (MedMCQA val 33.31% vs 32.01% — a tie within noise; MedQA ≈ chance at this scale); **100-epoch sequential protocol prepared** (train+test on all of MedMCQA → seed-locked 60% MedQA fine-tune → untouched 40% benchmark, better-of checkpoint selection, anti-overfit knobs T21/T22) | ❌ No thesis-final numbers yet (1 seed, ~50M class; depth-matched control + K ablations + multi-seed open) |
| ✅ Prior-art audit: 66 local papers, claim-to-evidence index | ❌ `[PENDING]` rows in every results table (Ch. 6) |
| ✅ Design audit: 4 defects found + proven + fixed | ❌ Parameter table needs measured numbers (L2 measured 6.56M vs 2.8M estimate) |
| ✅ Cost-matched protocol fixed *before* any run | ❌ Phase-1 epochs to re-pick: the legacy 100-epoch baseline memorised MedMCQA (33.33 % val, same as 3 epochs) — sandbox diagnostics now logged, LR bug fixed |

---

## 9. Likely questions, and the answer

**Q: Why not just use a bigger standard Transformer?**
Because P1 is not solvable by scaling — the KV cache still grows with context, and
the model stays most expensive on its intended inputs. Ekdant changes the *shape*,
not the size.

**Q: You say two layers — what about the third component?**
There are two layers: **L1** and **L2**. T between them is a plain windowed dense
Transformer — standard, not a contribution, never the subject of a claim. T exists
only to bound the KV cache and give local exact retrieval. Removing it either costs
exact recall (the pure-SSM limitation, documented) or reintroduces an unbounded cache.

**Q: Isn't K=4 debate just self-consistency voting?**
No, and it is measured against self-consistency at matched budget. Voting *merges*
paths; debate *cross-reads and revises* them, then refines with a descent step
against an external anchor. The K=1 control is single-path CoT.

**Q: What if the trajectories just agree with each other?**
That is Defect 2, and it has a proof and a fix. An agreement penalty is minimized by
the collapse it would detect; the floor inverts the constraint so the minimum sits on
the non-collapsed set. Collapse is also logged per step as `spread_ratio`.

**Q: What is the strongest evidence against this?**
**C24 — Rauba et al. 2026, Tiny Autoregressive Recursive Models.** It is the only
compute-matched study of exactly our design question and it reports *no reliable
gain*. We cite it against our own interest: it is why RQ2 is pre-registered with a
depth-matched-no-sandbox control, and why **a tie is a publishable outcome**.
What survives is narrower — stream separation helped (C21/C23), the
recursive-refinement part is the hypothesis under test.

**Q: Why no MoE?**
At this scale (12–50M parameters), routing machinery adds parameters without capacity to exploit
them. L2 spends them on hypothesis diversity instead — the sandbox *is* the mixture.

**Q: Where is the code?**
Implemented in `09-Code/src/ekdant/` — pure PyTorch: `model.py`, `transformer_stack.py`,
`sandbox.py`, `output_head.py`, `train.py`, `data.py`, with **141 passing unit tests**
(`09-Code/tests/`: `test_model.py`, `test_sandbox.py`, `test_transformer.py`,
`test_data.py`, `test_schedule.py`, `test_eval_split.py`) and
architecture-matched A/B configs (`configs/sandbox_50m.json` vs `baseline_50m.json`,
differing only in `use_sandbox`). It is **T + L2 (no L1)**: forward is embed → T (windowed
dense) → L2 (sandbox) → head; the L1 SSM storage remains designed in the vault/thesis,
not coded. **Full 3-epoch A/B executed on a T4** (one seed): MedMCQA val baseline
33.31% vs sandbox 32.01%, MedQA 25.69% vs 26.08% — **a tie within noise at this
scale** (publishable per the C24 framing); the 100-epoch sequential campaign, depth-matched
control, K ablations and multi-seed are the remaining thesis experiments. Sandbox diagnostics
are now logged per epoch (T24). The `train.py` LR schedule
bugs (warmup + stage-2 reset) are **fixed**.

**Q: What is the M3 write-back for?**
So reasoning leaves a trace in memory. It is the only edge between L1 and L2,
guarded by a causality rule (position t reads only positions < t) and a no-cross-batch-leakage
rule (`f` detached between micro-batches).

---

## 10. Connecting every dot

```
P1 memory grows   ──►  L1 selective SSM   ──►  O(1) state, 6 KB flat
P2 stack unauditable ─► the two layers stay separate ─► each ablated independently
P3 agreement ≠ confidence ─► L2 sandbox + floor ─► ρ logged, collapse detectable

L1 (memory) ──forward──► T (bounded window) ──forward──► L2 (reasoning) ──► head
   ▲                                                        │
   └──────────── M3 write-back: L2 consensus → S^sandbox ───┘

Defect 1 (readout-invariant) ──proof──► AGD with external anchor ──► Fig. 7
Defect 2 (penalty collapses) ──proof──► floor, non-degenerate   ──► Fig. 5
Defect 3 (not measurable)  ──protocol─► cost-matched RQ2, ties ok
Defect 4 (convention)      ──fixed────► explicit first-order ZOH

RQ1 memory  ← Fig. 6 (1.58 MB flat vs 127×) — projection, needs L1
RQ2 reasoning ← Fig. 7 (AGD vs inert peer-mean) + controls
RQ3 honesty ← spread_ratio monitor, calibration plot
```

**The claim in one line:** *Ekdant makes a model whose memory is bounded, whose
reasoning is visible, and whose design is falsifiable — at the 50M scale, with
every claim backed by a proof, a local paper, or a number* (memory by design, reasoning
by experiment).