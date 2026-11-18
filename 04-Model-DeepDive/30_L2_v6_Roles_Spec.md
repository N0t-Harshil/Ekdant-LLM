# 30 — L2 v6: four role generators and a selection gate (specification)

> [!note] Status
> **Ekdant v0.1.0 — Phases 1–2 implemented and tested on CPU (2026-11-18); not trained yet** (Phase 3/4 need the GPU, `run.md` A3t). Written by the author after the supervisor meeting of 2026-11-18. The implementation plan is [30a_L2_v6_Roles_Plan.md](30a_L2_v6_Roles_Plan.md). The sandbox of v1–v5 (v0.0.x) is described in [26c_Internal_Sandbox_Build.md](26c_Internal_Sandbox_Build.md).

## Version names (author, 2026-11-18)

| Name | What |
|---|---|
| v0.0.1 | v1, the from-scratch pilot |
| v0.0.2 – v0.0.5 | v2 – v5 on SmolLM2-135M (v0.0.5 = `configs/v5_final.json`) |
| **v0.1.0** | **this work: the sandbox v0.1.0 + the text-generation protocol** (`configs/ekdant_v0.1.0.json`) |
| v1.0.0 | reserved for the version shown to work |

The layer keeps its name: **the sandbox** (code: `SandboxV01`, setting `sandbox_version: "0.1.0"`).

## How the sandbox v0.1.0 works

### In plain words

SmolLM2 reads the text word by word. At one chosen point inside it, every word has a "state": 576 numbers that hold what the model currently thinks about that word. The sandbox takes that state and builds **four alternative versions** of it, each made by a reviewer with its own job:

- **R1, the explorer**, suggests a different reading of the word. It is built so that it cannot simply repeat or enlarge the original.
- **R2, the checker**, looks back at the earlier words of the text and suggests a version that fits what was said before.
- **R4, the best-supported alternative**, also looks back at the text and reads what R1 and R2 proposed, and suggests the version with the most support.
- **R3, the doubter**, looks at how much the other three disagree. If they disagree a lot, it suggests a version that keeps the question open instead of committing early.

Then the four reviewers **see each other's proposals once** and may adjust their own a little. Finally a **judge (the selection gate)** looks at all five versions (the original and the four proposals) and decides, word by word, how much each one counts. If none of the proposals helps, the judge can keep the original. The result goes on through the rest of SmolLM2.

At the start of training every reviewer proposes "no change", so the model is exactly SmolLM2; the reviewers and the judge then learn their jobs from the training data. Which reviewer the judge trusts, how often, and how different the proposals are is logged during training, so we can see whether the roles stay different or collapse into one.

### Technically

At an insertion point (after decoder layer k, or after the final norm), for each token t with stream state $h_t \in \mathbb{R}^{576}$:

1. **Input.** $x_t = \mathrm{RMSNorm}(h_t)$ (learned weight), so the sandbox reads the same scale at every depth (the raw stream has rms ≈ 2 at the end and ≈ 100 inside).
2. **Text memory.** keys $= W_K\,\mathrm{RMSNorm}(x_{\le t})$, values $= \mathrm{RMSNorm}(x_{\le t})$, causal mask (token t sees tokens ≤ t of its own row), history $\bar x_t$ = mean of the normalised states so far.
3. **Role generators** (each `Linear → SiLU → Linear`, hidden $4d$, last layer zero-initialised):
   - R1: $u = F_1(x)$, $\Delta_1 = u - (u\cdot\hat x)\hat x$ (orthogonal to $x$);
   - R2: $\Delta_2 = F_2([x;\ \mathrm{Attn}(W_{Q2}x,\ \text{memory})])$;
   - R4: $\Delta_4 = F_4([x;\ \mathrm{Attn}(W_{Q4}x,\ \text{memory});\ \Delta_1;\ \Delta_2])$;
   - R3: $\Delta_3 = F_3([x;\ \sqrt{\mathrm{Var}(\Delta_1,\Delta_2,\Delta_4)+\epsilon}])$ (per-channel spread).
4. **Candidates.** $C_0 = x$ (kept unchanged), $C_i = x + \Delta_i$.
5. **Communication round** (1 round; own weights): $n = \mathrm{RMSNorm}(C)$, $\mathrm{cross}_i$ = attention of $n_i$ over the other four candidates (never itself); $\Delta_i \leftarrow \Delta_i + \sigma(W_{\gamma,i}[n_i;\mathrm{cross}_i]) \odot W_{o,i}\,\mathrm{cross}_i$ with $W_{o,i}$ zero-initialised, gate bias −2. $C_0$ is read but not changed.
6. **Selection gate.** $s_i = G([\mathrm{RMSNorm}(C_i);\ x;\ \bar x])$, $G$ = `Linear(3d, 256) → SiLU → Linear(256, 1)` (last layer zero-initialised, so $p_i = 0.2$ at the start), $p = \mathrm{softmax}(s)$ over the five candidates, in float32.
7. **Output.** $h'_t = h_t + \mathrm{rms}(h_t)\cdot\sum_{i=1}^{4} p_i\Delta_i$ (equivalently $\sum_i p_i C_i$ in the normalised space, sized back to the stream). At step 0 every $\Delta_i = 0$, so $h' = h$ exactly.
8. **Diagnostics** (real tokens only): `sel_p` (mean weight per candidate), `sel_top` (how often each candidate has the largest weight), `sel_entropy`, `cand_cos` (5×5 mean cosine), `delta_rms`, and per-role gradient norms (`EkdantLM.role_grad_norms`). Candidate ablation: `EkdantLM.force_candidate(i)` and `python -m ekdant.gen_eval candidates <run>`.

### How R1–R4 evolve during training

**In plain words.** The reviewers are not fixed rules; they are small networks that start out proposing "no change" and **learn while the model trains**. Two things are fixed by design and never change: *what each reviewer is allowed to look at* (R1 only the word's state, R2 and R4 also the earlier words, R4 also R1's and R2's proposals, R3 their disagreement) and *R1's rule that its proposal must point in a new direction*. Everything else — what each reviewer actually proposes, how they react to each other, and how much the judge trusts each one — is learned from the training text. The learning signal is simple: after every batch, each reviewer is nudged so that its proposal would have made the model predict the next word better, **in proportion to how much the judge used it**. The judge is nudged towards the reviewers whose proposals helped. Over training, each reviewer should drift into the kind of correction its inputs make it good at.

The weak point is the same feedback loop: a reviewer the judge ignores gets almost no nudge, so it stops improving, so the judge ignores it more (spec §8). That is why every log line shows how often each one wins (`sel=`), how undecided the judge is (`H=`), and how strongly each reviewer is still learning (`gr=`). The names ("explorer", "checker", "doubter") are what the structure is built for; whether a reviewer *really* does that job after training is a result to measure (candidate ablation, `cand_cos`), not a guarantee.

**Technically.** All parameters of $F_1..F_4$, $W_{Q2}, W_{Q4}$, the memory key, the communication round and the gate $G$ receive gradients from the next-token loss $\mathcal{L}_{CE}$ (no auxiliary loss in Phase 1). With $h' = h + \mathrm{rms}(h)\sum_i p_i\Delta_i$:

$$\frac{\partial \mathcal{L}}{\partial \Delta_i} = p_i\,\mathrm{rms}(h)\,\frac{\partial \mathcal{L}}{\partial h'} \;(+\text{ terms through R3, R4 and the round}),\qquad \frac{\partial \mathcal{L}}{\partial s_i} = p_i\Big(\mathrm{rms}(h)\,\Delta_i - \textstyle\sum_j p_j\,\mathrm{rms}(h)\,\Delta_j\Big)\cdot\frac{\partial \mathcal{L}}{\partial h'}$$

So role $i$ learns at a rate scaled by its gate weight $p_i$, and the gate raises $s_i$ for candidates whose change points down the loss gradient more than the current mixture. At step 0 every $\Delta_i = 0$ and $p_i = 0.2$: the last layers of $F_i$ get gradient first (as in LoRA's zero-init $B$), then the earlier layers. What is fixed: the inputs of each $F_i$, R1's projection orthogonal to $x$, causal masking, $C_0 = x$. The training-time evidence that the roles evolve and stay distinct: `gr=` (per-role gradient norm, `EkdantLM.role_grad_norms`), `delta_rms` (size of each proposal), `cand_cos` (5×5 similarity; collapse = off-diagonal → 1), `sel_p` / `sel_top` / `sel_entropy` (gate use), and after training `python -m ekdant.gen_eval candidates <run>` (perplexity with each candidate forced). Role-specific objectives that would push each role towards its job are Phase 5 (one at a time, each with a definition, a coefficient and an experiment — spec §8.3).

### Role objectives — training each reviewer for its job (built, off by default)

**In plain words.** Left alone, the next-word loss only rewards "help the model predict the next word"; it does not tell R3 to be the doubter. Three extra training signals, each switched on separately, push the reviewers towards their jobs:

1. **R3 learns uncertainty.** A tiny probe looks only at R3's proposal and must guess whether the model is about to predict the next word correctly. To make that guess possible, R3's proposal has to carry the information "the model is unsure here".
2. **No reviewer is starved.** Each reviewer's own version of the word is also scored on the next word directly, even when the judge ignores it. So a reviewer keeps learning and can win later, instead of fading out (the collapse loop above). Only at the end placement, where scoring a candidate is one extra output-layer step.
3. **The judge explores first.** For the first steps of training the judge gets a small reward for spreading its trust over the five versions, so it does not lock onto one reviewer before the others have learned anything. The reward fades to zero.

**Technically** (`V2Config` keys, CLI flags of fresh runs; all 0 = off by default; logged as `r3=`, `cand=`, `entropy=`):

| Objective | Definition | Flag (key) |
|---|---|---|
| R3 correctness | $\mathcal{L}_{R3} = \mathrm{BCE}(w^\top\Delta_3 + b,\ \mathbb{1}[\arg\max \text{logits}_t = y_{t+1}])$ on answer tokens; the target is detached; the probe $w,b$ is part of R3 (577 parameters, only when on) | `--r3-loss λ` (`sandbox_r3_loss`) |
| Candidate loss | $\mathcal{L}_{cand} = \frac14\sum_{i=1}^{4}\mathrm{CE}(\mathrm{lm\_head}(h + \mathrm{rms}(h)\Delta_i),\ y)$ on answer tokens, one candidate at a time; end placement only (`validate`) | `--cand-loss λ` (`sandbox_cand_loss`) |
| Early exploration | $-\lambda_t\,\bar H(p)$ with $\lambda_t = \lambda\max(0, 1 - t/N)$ over the first $N$ optimiser steps of stage 1 | `--explore λ --explore-steps N` (`sandbox_explore`, `sandbox_explore_steps`, default N = 1000) |

$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{R3}\mathcal{L}_{R3} + \lambda_{cand}\mathcal{L}_{cand} - \lambda_t\bar H(p)$. None of them runs in evaluation. Per spec §8.3 each is tried **alone first** against the same run without it (`run.md` A3t, Phase 5); a combination only after each one's effect is known.

Size, measured with `addon_params()` on SmolLM2-135M: **23,686,785** add-on parameters (R1 2,657,088 · R2 4,315,968 · R3 3,984,192 · R4 6,970,176 · gate 443,457 · memory + communication 5,314,176; the rest are the site's norms and its unused outer gate), total 158,201,793 — +20.9 % against v0.0.5's 19,594,944. Cost per token: one forward of the sandbox, no extra passes of SmolLM2.

## Decisions taken for Phase 1 (2026-11-18)

The author asked to build the new design into the existing sandbox layer ("this is the same plan for the sandbox layer"). Where the specification leaves a choice open, the audit's recommendations apply:

| Open point | Decision |
|---|---|
| Where the code lives | Inside `09-Code/src/ekdant/sandbox.py` (no new module, RULES §4) as `SandboxV01` (the sandbox v0.1.0); the v0.0.x `Sandbox` stays unchanged and is the default |
| Benchmark (author, 2026-11-18) | The multiple-choice benchmarks are dropped for v0.1.0 (the code stays). Text generation instead: stage 1 = [medical-o1-reasoning-SFT](https://hf.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) (en, question → reasoning → final answer; `--stage1 reasoning`), stage 2 = MedQuAD fine-tune half (unchanged), benchmark = [MedLFQA](https://hf.co/datasets/dmis-lab/MedLFQA) (`--medlfqa`; its questions are removed from all training data) plus the MedQuAD benchmark half |
| Factuality judge (author: option A) | [pritamdeka/PubMedBERT-MNLI-MedNLI](https://hf.co/pritamdeka/PubMedBERT-MNLI-MedNLI) scores comprehensiveness and hallucination; used for scoring only, never trained |
| Selection | Soft (weighted) selection first; hard selection is Phase 5 |
| Communication / refinement | One round first (`roles_rounds = 1`) |
| Role-specific auxiliary losses | Built 2026-11-18 at the author's request, **off by default**: R3 correctness, candidate loss, early exploration (section "Role objectives" above); each tested alone in Phase 5 |
| Answer length (author, 2026-11-18) | `max_answer` 768 in `ekdant_v0.1.0.json` (both arms): the reasoning targets have a median of 535 SmolLM2 tokens (p90 712, p99 935, max 1,472, measured on all 19,704 rows); 512 kept 8,164 rows (41.4 %), 768 keeps 18,596 (94.4 %). v0.1.0 numbers are therefore not comparable with v0.0.x runs (512) |
| Backbone | Frozen for the Phase 3 sanity run (`--l2-warmup`); fully fine-tuned for the Phase 4 placement study, exactly like the baseline arm |
| Evaluation | MedLFQA (perplexity, ROUGE-L / F1, comprehensiveness, hallucination), MedQuAD benchmark, L2-off test, candidate ablation; maths / logic benchmarks would need the author's approval |
| Size and cost | About v0.0.5's size: measured 23,686,785 (+20.9 %, within the agreed ±25 %); training cost below 2× the baseline |
| No-op start | Every generator's output layer starts at zero, so at step 0 every candidate equals h and the model equals the backbone (RULES §4) |

### Role generators, as built in Phase 1

| Role | Mechanism that makes it different |
|---|---|
| R1 Alternative exploration | Its change is projected to be orthogonal to the word state, so it cannot copy or rescale it |
| R2 Consistency and verification | Its change is computed from a causal attention read of the earlier tokens |
| R4 Best-supported alternative | Reads the earlier tokens with its own query **and** the proposals of R1 and R2 |
| R3 Uncertainty and calibration | Reads how much R1, R2 and R4 disagree (their per-channel spread) and the word state |

---

## The specification (as written by the author)

### Purpose of this document
This document specifies the architecture, implementation boundaries, integration strategy, and experimental procedure for the L2 module discussed in the research conversation. It is intended for the coding agent working inside the existing repository. This is an implementation specification for an addition to the existing pretrained LLM. It is not permission to redesign, replace, or rebuild the underlying transformer. The coding agent must inspect the repository before proposing or making changes. The first deliverable is an audit and implementation plan, not an immediate rewrite.

### 1. Non-negotiable project boundaries

**1.1 Work with the existing model.** The existing pretrained LLM is the foundation. Preserve its architecture and behaviour wherever possible. Before changing code, inspect and understand: the model implementation and transformer block structure; the hidden-state dimensions and tensor layouts; the forward pass and the points at which hidden states enter and leave each block; attention, causal masking, positional information, normalization, and residual connections; embedding and output-head interfaces; checkpoint loading, saving, and compatibility; training, fine-tuning, inference, and evaluation pipelines; existing tests, configuration conventions, and experiment logging. Do not assume a model family, framework, tensor layout, or layer numbering convention. Discover these from the repository.

**1.2 Changes must be minimal and reversible.** Do not: rebuild the transformer; replace attention, feed-forward, normalization, embedding, tokenizer, or output-head components without a demonstrated requirement and explicit approval; rewrite the training or inference pipeline unnecessarily; rename or reorganize unrelated modules; modify unrelated files; overwrite the original pretrained checkpoint; introduce broad dependencies for a small feature; silently change the base model's behaviour when L2 is disabled. Implement L2 as a modular component that can be enabled, disabled, configured, tested, and removed without redesigning the base model. If an integration requires a significant change to existing architecture, stop. Explain the technical reason, alternatives, affected files, and risks. Obtain approval before proceeding.

**1.3 Preserve the baseline.** Maintain a reproducible run of the original model without L2. The project must support, as far as the existing code permits: original model, L2 disabled; original model with one L2 insertion point; original model with L2 at other configurable insertion points. Keep original checkpoints intact. Save modified checkpoints and experiment artifacts separately.

### 2. First action: repository audit, not implementation
Before editing files, produce an audit containing: the exact base model and implementation found in the repository; how hidden states flow through the transformer; the layer indexing convention used by the code; the most suitable minimal integration interface; files that would need modification and why; new files or modules that would be added; compatibility risks, including tensor shapes, normalization, causal masking, checkpoint handling, and training; how the original model will remain runnable as a baseline; a proposed minimal first implementation; a proposed test plan. Do not begin broad code changes before presenting this plan and receiving approval. Small, isolated inspection or diagnostic commands are acceptable. Do not modify the repository as part of the audit unless explicitly authorized. If the repository already contains partial L2 implementation, inspect it and build on it. Do not create duplicate modules or replace working code merely to match a preferred design.

### 3. Architectural flow
At a configurable insertion point in the existing transformer, L2 receives the current hidden-state sequence. For token position $t$, denote the original hidden state by $h_t \in \mathbb{R}^{d}$. L2 retains this original representation and generates four additional candidate representations $R_{1,t}, R_{2,t}, R_{3,t}, R_{4,t}$. The five candidates are $C_t = \{h_t, R_{1,t}, R_{2,t}, R_{3,t}, R_{4,t}\}$.

The conceptual processing order is: receive the existing hidden-state sequence; retain the original representation; generate four strategically distinct representations; allow candidate representations to communicate; refine/evolve the representations; evaluate the candidates with a learned selection gate; produce the resulting hidden-state sequence; return that sequence to the remaining original transformer layers.

The candidates are alternative hidden-state representations of the same token positions. They are not vocabulary tokens, independently generated text sequences, or separate full language models. All candidate representations must be compatible with the base model's hidden dimension and downstream interfaces.

### 4. Four mandatory strategic representation generators
The four generators have distinct, mandatory responsibilities. Their internal transformations and parameters should be learnable, but their responsibilities must not be silently removed, merged, or treated as optional.

**R1 — Alternative Exploration.** Explore alternative representations and interpretations that may be under-emphasized by the original hidden state. Requirements: produce a meaningful alternative, not a simple duplicate; explore useful representational directions supported by the input and context; avoid arbitrary noise as a substitute for exploration.

**R2 — Consistency and Verification.** Develop representations that investigate contextual and logical consistency. Requirements: make use of relevant contextual relationships; learn to represent consistency-related evidence and potential conflicts; do not reduce this role to a brittle, hand-coded rule checker.

**R3 — Uncertainty and Confidence Calibration.** Preserve ambiguity and avoid unnecessary commitment to unsupported interpretations. Requirements: represent uncertainty-related information meaningfully; do not merely shrink vector magnitude; do not blindly lower confidence or make the model permanently cautious; preserve plausible alternatives when the evidence does not justify premature commitment.

**R4 — Best-Supported Alternative.** Explore an alternative representation that integrates the strongest available contextual and learned evidence. Requirements: construct a useful alternative based on available evidence; do not simply copy the original representation; do not duplicate the selection gate's function.

**Shared requirements.** Each generator must have its own learnable parameters or a clearly justified parameterization that preserves its distinct role. Keep generator interfaces modular. Use role-specific auxiliary objectives only when their purpose can be defined and evaluated. Avoid rigid hand-coded transformations that prevent the generators from learning. Test whether the generators actually learn distinct and useful representations. The role descriptions define architectural responsibilities; they do not prescribe an unchangeable mathematical implementation.

### 5. Candidate generation
A possible residual formulation is $R_{i,t} = h_t + \Delta_{i,t}$ with $\Delta_{i,t} = F_i(h_t, c_t; \theta_i)$, where $h_t$ is the original hidden state, $c_t$ is relevant contextual information, $F_i$ is a learnable generator transformation and $\theta_i$ its trainable parameters. Treat this as a starting formulation, not a requirement to use it unchanged if the repository or experiments justify a better compatible design.

Requirements: preserve the original representation as an explicit candidate; keep output dimensions compatible with the base model; use context without violating the base model's causal constraints; avoid arbitrary random placement in hidden-state space; monitor candidate norms, activation distributions, normalization, similarity, diversity, and numerical stability; check that generators do not simply reproduce the original representation; do not assume that the most distant candidate is best or that the closest candidate is best. Any use of context must respect the model's attention mask and information-availability rules. In an autoregressive model, a token's candidate must not receive information from future tokens.

### 6. L2 internal exploration and interaction
The candidates should not remain completely independent. They should be able to exchange useful information and influence one another while preserving their strategic distinctions. Potential mechanisms to evaluate include cross-attention between candidate representations, message passing, a learned interaction network, or another lightweight mechanism compatible with the base model. A conceptual refinement expression is $R_{i,t}^{(k+1)} = R_{i,t}^{(k)} + F_i^{comm}(C_t^{(k)}, c_t)$, where $k$ is the refinement round. This is illustrative. Choose an implementation based on the existing model, computational cost, stability, and test results.

Requirements: begin with a simple, lightweight communication design; make the interaction differentiable for the initial training approach; preserve causal constraints; preserve the distinct strategic responsibilities; monitor whether repeated interaction causes all candidates to become nearly identical; test a small number of refinement rounds first; do not assume more rounds improve performance; record the chosen mechanism and its configuration for every experiment.

### 7. L2's learned selection gate
After candidate generation and refinement, a learned gate evaluates all five candidates: the original representation $h_t$, R1, R2, R3, R4. The gate should use relevant contextual information where appropriate and make decisions independently for each token position. The useful candidate may differ between tokens and contexts. A possible formulation is $s_t = G(C_t, c_t; \theta_G)$, $p_{t,i} = \operatorname{softmax}(s_t)_i$. For an initial differentiable implementation, investigate weighted selection $h_t^{new} = \sum_{i=0}^{4} p_{t,i} C_{t,i}$. The resulting sequence is passed into the remaining original transformer layers. Later, investigate hard selection $h_t^{new} = C_{t,\arg\max_i(s_{t,i})}$. Hard argmax is not differentiable. If hard routing is investigated, evaluate suitable training approaches such as Gumbel-Softmax or straight-through estimators. Do not assume these methods are automatically appropriate or superior.

Requirements: validate score and weight shapes; for weighted selection, verify that weights are finite, nonnegative, and sum to one within numerical tolerance; log gate weights and selection statistics; ensure gradients reach the intended gate and generator parameters; compare weighted and hard selection only as controlled experiments; do not silently change inference behaviour relative to the training design. A small two-layer MLP with a suitable nonlinearity may be a reasonable initial gate, but inspect the existing implementation and justify the choice.

### 8. Preventing generator and gate collapse
A critical risk is that one candidate becomes dominant early: one candidate initially performs slightly better; the gate selects it more frequently; other candidates receive less effective learning signal; the dominant candidate receives more useful updates; the difference increases until the other candidates contribute little. This is a failure mode to monitor and investigate.

**8.1 Useful specialization versus collapse.** Useful specialization means different candidates contribute in different contexts or task conditions. Unhealthy collapse may involve: one candidate winning almost everywhere; other candidates receiving negligible learning signals; candidates becoming nearly identical; one generator becoming inactive or ineffective; the gate no longer making meaningful context-dependent choices. Do not force a permanent 20 % selection rate for each candidate. Equal selection frequency is not the objective.

**8.2 Required diagnostics.** Measure and log: selection frequency for all five candidates; gate entropy and weight distributions; selection patterns by task or context where possible; pairwise candidate cosine similarity; candidate vector norms and activation statistics; generator-specific gradient magnitudes; candidate ablation effects; whether each generator receives meaningful training signals; whether the four strategic roles remain distinguishable.

**8.3 Potential mitigation experiments.** Investigate, rather than blindly combine: early-stage exploration; entropy regularization; batch-level load balancing; representation-diversity objectives; role-specific auxiliary objectives; auxiliary learning signals for candidates even when not selected; soft-to-hard routing schedules; alternative gradient-routing strategies. Any added objective must have a precise definition, a stated purpose, a tunable coefficient, and an experiment showing whether it helps. A conceptual total objective might be $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_b\mathcal{L}_{balance} + \lambda_d\mathcal{L}_{diversity} + \lambda_r\mathcal{L}_{roles}$. This is not a prescription to add all three auxiliary losses. Define and test each independently before combining them.

### 9. L2 placement experiments: insertion depth is configurable
The best insertion depth is unknown. Do not hard-code a preferred depth or assume that the middle is best. Different transformer depths may contain different kinds of representations, but the useful insertion point must be determined empirically for this model.

**9.1 Configurable placement.** Implement a configuration mechanism that identifies the insertion point without rewriting the model. Adapt it to the actual model, repository conventions, and layer indexing. Clearly document whether insertion is before or after a given transformer block.

**9.2 First placement study: one L2 block per run.** Initially, each experiment must use exactly one L2 block at one selected depth. For each position: start from the same original pretrained checkpoint; insert one L2 block at that position; keep all other architecture and experimental conditions comparable; train or fine-tune using the defined protocol; evaluate on the same evaluation suite; record gate behaviour, representation diagnostics, and compute overhead; save the configuration, checkpoint, metrics, and logs separately. Do not put L2 at every transformer layer in the first experiment.

**9.3 Controlled comparison.** Keep consistent wherever possible: base checkpoint; training and evaluation data; data order or sampling procedure; training budget and stopping criteria; L2 configuration, except for insertion depth; optimizer and learning-rate protocol; batch size and sequence-length conditions; evaluation prompts and decoding settings; hardware and measurement procedure. Record random seeds. Repeat runs when resources permit. If any condition must differ, document it and explain how it affects comparison.

**9.4 Placement metrics.** Compare: validation loss and next-token metrics; mathematical reasoning; logical consistency; multi-step reasoning; contextual understanding; uncertainty calibration; unsupported conclusions; repetition and generation stability; candidate diversity and gate utilization; training stability; inference latency; memory consumption and parameter/compute overhead. Do not conclude that one depth is superior from a single noisy metric or one run.

**9.5 Multiple L2 blocks are a later study.** Only after single-insertion experiments are understood should multiple L2 blocks be investigated. Test whether their contributions are complementary or whether they interfere, cause collapse, or introduce excessive cost. Do not assume that more blocks produce better results.

### 10. Integration with the original transformer
Existing pretrained transformer layers → configurable insertion point → original hidden-state sequence → L2 candidate generation → original state + R1 + R2 + R3 + R4 → L2 internal exploration and interaction → candidate refinement/evolution → L2 learned selection gate → resulting hidden-state sequence → remaining original transformer layers → original output and prediction pipeline.

Preserve compatibility with: hidden-state dimensions and tensor layouts; residual pathways; normalization behaviour; attention interfaces; positional information; causal masks; autoregressive information flow; gradient flow; checkpoint loading and saving; existing inference and training APIs. Do not assume that matching tensor dimensions alone makes the integration correct. Test the complete forward and backward paths. When L2 is disabled, verify that the original path is unchanged as far as the repository's implementation permits.

### 11. Training strategy
Begin with the existing model's normal next-token prediction objective $\mathcal{L}_{CE} = -\sum_t \log P(x_{t+1} \mid x_{\leq t})$. Do not prescribe a particular optimizer in advance. Inspect the existing training system and choose an optimizer and configuration based on the base model, the new L2 parameters, available hardware and memory, training stability, the fine-tuning strategy, and existing repository conventions. Investigate whether the base model should be frozen while L2 is trained, partially fine-tuned, or fully fine-tuned. Ensure all intended trainable L2 components receive appropriate learning signals. Monitor gradients and parameter updates. Do not add auxiliary losses without defining their mathematical form, purpose, scale, and evaluation.

### 12. Evaluation plan
Training loss alone does not establish improved reasoning. Use the original model as the baseline and compare it against L2-enabled variants under matched evaluation conditions. Evaluate, where suitable datasets or reliable test procedures are available: language-model performance (training loss, validation loss, next-token accuracy, perplexity); reasoning and understanding (mathematical reasoning, logical consistency, multi-step reasoning, contextual understanding, instruction following); reliability (ambiguity handling, confidence calibration, unsupported conclusions, repetition, generation stability); L2-specific behaviour (candidate representation diversity, pairwise similarity, gate selection distribution, context-dependent selection, generator gradient flow, candidate ablation results, collapse indicators); computational cost (training time, inference latency, peak memory, additional parameters, additional computation). Choose evaluation tasks that actually measure the claimed capability. Document benchmark versions, data splits, scoring methods, and decoding settings. Report improvements and regressions. Do not selectively report only successful metrics.

### 13. Tests required before declaring integration successful
At minimum, verify: the original model loads; the original model runs with L2 disabled; L2 can be enabled at a configurable insertion point; the original hidden state is retained as a candidate; all four generators return valid, compatible tensors; the four generators have distinct intended responsibilities in the implementation and diagnostics; communication operates correctly; refinement operates correctly; gate scores have the expected shape; weighted gate outputs are finite and normalized; gradients reach intended trainable components; forward and backward passes complete; causal masking and autoregressive constraints are preserved; the remaining original transformer layers accept the resulting states; checkpoints save and reload correctly; different insertion depths can be tested through configuration; the original baseline remains reproducible; L2 adds no unintended changes when disabled. Include tests for numerical stability, invalid configurations, and batch/sequence shapes supported by the base model. Passing tests establishes implementation correctness, not research effectiveness.

### 14. Experiment tracking and reproducibility
Give each experiment a unique ID. Record: experiment ID and date; base model identity and checkpoint; insertion depth and before/after-layer convention; L2 configuration; generator configuration; communication mechanism; refinement rounds; gate configuration and selection mode; training configuration; optimizer actually used; fine-tuning strategy; dataset and evaluation details; random seed; hardware; metrics and uncertainty across runs where available; latency and memory; failures and anomalies; interpretation and next decision. Save configurations, logs, and checkpoints separately. Never silently overwrite the baseline or prior experiment results. Make experiments reproducible from their saved configuration.

### 15. Development phases and approval gates
- **Phase 0 — Audit.** Inspect the repository and report the architecture, integration point, affected files, risks, and minimal plan. Obtain approval before code modifications.
- **Phase 1 — Minimal modular integration.** One configurable insertion point; original representation retained; four mandatory strategic generators; a simple communication mechanism; a simple refinement mechanism; a learned selection gate; output returned to the remaining original transformer. Avoid adding every proposed anti-collapse method or experimental option at once.
- **Phase 2 — Functional validation.** Run unit and integration tests. Verify the original baseline, L2-enabled forward pass, backward pass, gradients, masks, checkpoint handling, and configurable placement.
- **Phase 3 — Initial sanity experiment.** Establish that the model trains without numerical failure, the L2 components receive gradients, candidate representations are not trivially identical, the gate does not immediately collapse, the modified model remains usable, and the overhead is measurable. This is a sanity test, not proof of reasoning improvement.
- **Phase 4 — Single-insertion placement study.** Independent experiments at different depths, one L2 block per run, against the baseline under controlled conditions.
- **Phase 5 — Targeted improvements.** Use observed evidence to decide on more complex communication, more refinement rounds, other gate designs, anti-collapse mechanisms, auxiliary objectives or other fine-tuning strategies. Change one major variable at a time.
- **Phase 6 — Multiple insertion points.** Only after the single-insertion results are understood.

### 16. Coding-agent operating rules
Inspect before editing. Build on existing code rather than duplicating it. Preserve the current model and baseline. Make the smallest change that can test the hypothesis. Explain significant proposed changes before making them. Separate verified repository facts from hypotheses. Keep the four generator responsibilities intact. Do not hard-code an insertion depth. Do not prescribe an optimizer without inspecting the project. Do not add complexity without an experimental reason. Do not silently change the research objective. Do not overwrite checkpoints or experiment results. Ask for approval before broad or disruptive changes. Report failures honestly. Do not claim success merely because the code runs.

### Research principle
Preserve the existing pretrained model. Integrate L2 as a minimal, modular, reversible addition. Test the complete data flow. Measure candidate behaviour and computational cost. Compare against a reproducible baseline. Expand the architecture only when experimental evidence justifies doing so.

---

## Connections
- Current L2 build manual: [26c_Internal_Sandbox_Build.md](26c_Internal_Sandbox_Build.md)
- Implementation plan: [30a_L2_v6_Roles_Plan.md](30a_L2_v6_Roles_Plan.md)
- Runbook: [../run.md](../run.md)
