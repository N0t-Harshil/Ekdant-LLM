# Ekdant: Memory-Efficient Reasoning with Selective SSM Storage and Parallel Sandbox Consensus

**Harshil Gorasiya** — BEng Thesis, Hochschule Heilbronn
Supervisor: [Supervisor 1] · Co-supervisor: [Supervisor 2]

> Compiled from this vault. Canonical spec: [02_System_Architecture.md](../02-Architecture/02_System_Architecture.md). Sources: [01](../01-Introduction/01_Project_Overview.md) · [19](../03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md) · [20](../03-Background/20_Thesis_Ch2_Sandbox_Papers.md) · [21](../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) · [22](../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) · [05-Data](../README.md) · [07](../06-Training-Eval/07_Training_Pipeline.md) · [12](../02-Architecture/12_Configuration_Guide.md) · [09/14](../README.md) · [11](../07-Management/11_Future_Work.md).

> [!note] This file is the **condensed vault-linked outline**. The full article, with the
> notation tables, proofs, prior-art audit and all 65 references, is [thesis.pdf](thesis.pdf),
> built from [thesis.tex](thesis.tex). Where the two differ, the PDF is authoritative.

---

## Figures

The seven diagrams from the compiled article, exported at 240 dpi.

**Fig. 1 — Colour key.** Fixed code used by every figure and every table row group.

![Colour key: blue = L1 storage, green = stage T, orange = L2 sandbox, red = feedback, grey = neutral](figures/fig1.png)

| Colour | Meaning | Applied to |
|---|---|---|
| Blue `#1F4E79` | Stage L1 — storage / memory | L1 boxes, parameter-table rows |
| Green `#2E6B4F` | Stage T — local dense compute | T boxes, parameter-table rows |
| Orange `#7B3F00` | Stage L2 — reasoning | L2 boxes, sandbox-store row |
| Red `#A11212` | Feedback / defect | M3 arrows, `[PENDING]` rows |
| Grey `#8A94A0` | Stage-neutral tensors, heads | Embedding, head, projections |

**Fig. 2 — Three-stage forward pass.** Solid arrows are the forward chain; the dashed red edge is the
M3 write-back that returns **L2's** consensus into L1's third state store (`S^sandbox`). L1 and L2 are
separate stages; the store is SSM memory, the sandbox is the debate layer.

![Three-stage forward pass with the M3 write-back loop](figures/fig2.png)

**Fig. 3 — One L1 step.** Selection emits per-channel step sizes; three stores integrate the same
token at different rates; a learned gate fuses them with a residual.

![L1 block: selection, three selective stores, gated fusion](figures/fig3.png)

**Fig. 4 — Stage L2 in full.** Debate is horizontal (trajectories interact, anchor not yet defined);
anchored descent is vertical (all trajectories move toward one external point).

![Stage L2: diverse init, debate rounds, anchored gated descent, consensus](figures/fig4.png)

**Fig. 5 — The spread floor.** Exactly zero across the healthy regime; bounded at 0.0025 under total
collapse, so it can flag collapse without causing it.

![Spread floor g(rho) on a log axis](figures/fig5.png)

**Fig. 6 — Inference state vs. sequence length.** Ekdant holds flat at 1.58 MB; full attention grows
linearly, reaching 127x at T = 65536. *Analytic projection, not measured.*

![Inference state versus sequence length: Ekdant flat, full attention linear](figures/fig6.png)

**Fig. 7 — The central ablation.** Energy falls in *both* variants, and faster for the broken one — so
energy cannot detect the disconnection. Only the readout differs: AGD moves the consensus, the inert
peer-mean leaves it bit-identical to S = 0.

![AGD versus inert peer-mean: energy falls in both, only the consensus moves for AGD](figures/fig7.png)

---

---

## Ch.1 — Introduction

**Problem.** Transformer decoders scale quadratically in sequence length (KV-cache growth); a monolithic stack hides which part reasons; softmax confidence is uncalibrated. Long inputs — codebases, records, papers with references — are exactly where reasoning matters and where attention costs most.

**Idea: Ekdant.** Our two layers — ==L1 SSM Storage== (selective state-space memory) and ==L2 Sandbox== (K trajectories + debate/energy consensus) — joined by a ==dense Transformer stack T== (windowed GQA + SwiGLU, no MoE), at ==XS ≈12M==, ==S ≈21M== and ==M ≈46M==. Two losses: ==LM CE + sandbox spread floor==.
- ==Layer 1 — SSM Storage (memory answer).== Selective state-space stack compresses any sequence into a fixed-size recurrent state. Linear-time training, ==O(1) inference==, no KV-cache. ([21](../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md))
- ==Layer 2 - Sandbox (reasoning answer).== L1 output expands into ==K parallel trajectories== that debate (cross-read + revise), then run ==anchored gated descent== on the energy `½‖T_k − a_t‖²` toward a token-conditioned anchor `a_t = W_a·hT_t`, and vote. Agreement is the observable confidence signal; collapse is policed by a spread *floor*, not an agreement penalty. ([22](../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md))

Forward: `input_ids → embed+RoPE → SSMStack → TransformerStack → K trajectories → debate → anchored gated descent → consensus → vocab_head → logits`. Two losses: ==LM CE (w=1.0) + sandbox spread floor (w=0.1)==.
- ==Why dense + windowed:== global attention's growing KV-cache is the wall from Ch.1; T keeps a fixed window (W=512), so memory stays bounded. No MoE — the sandbox is the mixture mechanism.
- ==Bounded-memory story:== L1 fixed state + T window-bounded KV + dense FFN + K-bounded sandbox — peak memory flat in T.

**Research questions.**
- ==RQ1 (memory):== Can fixed-size selective SSM replace the KV-cache at matched quality? Ablations: d_state; SSM vs. tiny-Transformer; memory-vs-length curve.
- ==RQ2 (reasoning):== Do K trajectories with debate+consensus beat single-path decoding at matched compute? Ablations: K=1/4/8; rounds 0/2/3; steps 0/8; vs. self-consistency. **Runnable at XS/S** on perplexity, ARC-easy/HellaSwag/LAMBADA and synthetic known-answer multi-step items; GSM8K/HumanEval are ==scaling-held-out== because the smallest model reported past 80% GSM8K is 34B (arXiv:2312.09241).
- ==RQ3 (honesty):== Does trajectory agreement predict correctness? Measure agreement-vs-accuracy calibration.

**Contributions.** (1) Our three stages (SSM memory → dense T → sandbox reasoning) with no MoE. (2) In-graph debate plus anchored energy descent with cost-matched evaluation, and an audit proving the discarded peer-mean variant inert. (3) ==66-paper local library== ([proof index](../papers/Proof_Library/00_Proof_Index.md), [novelty audit](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md)) mapping every design choice to a downloaded source. (4) A production data pipeline (54 datasets, 8 stages).

**Scope.** Text-only pretraining (+optional SFT/DPO); 5 benchmarks (HumanEval, GSM8K, MMLU, HellaSwag, ARC); honesty probes. Tools/multimodality/MoE out of scope.

---

## Ch.2 — Background

### §B Why selective SSM (L1)
Lineage: S4 (Gu et al. 2021, B1) → diagonal S4D/DSS (B2/B3) → S5 parallel scan (B4) → ==selection + scan: Mamba (Gu & Dao 2023, B5, primary L1 blueprint)== → duality SSD/Mamba-2 (B6). Foundations: HiPPO (B13), LRA benchmark (B14). Family: Hyena (B7), H3 (B8), RetNet (B9), RWKV (B10), Griffin (B17), xLSTM (B19), MambaByte (B18), S4ND (B15), Liquid-S4 (B16). Baselines: FlashAttention (B20). Limitation anchor: ==Jamba hybrid (B11)== + surveys (B12a) — pure SSM trails attention on exact recall; thesis stays pure for falsifiability. Full vault: [19](../03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md).

> L1 follows the selective-SSM line (S4; diagonal; S5; selection+scan; duality), with Hyena/H3/RetNet/RWKV as family context and Jamba + surveys marking the acknowledged retrieval limitation.

### §C Why parallel-trajectory sandbox (L2)
Control: ==CoT single path (Wei et al. 2022, C1) = K=1==. Baseline: ==self-consistency voting (Wang et al. 2022, C2)== — every sandbox claim is measured against it at matched budget. Blueprints: ==multi-agent debate (Du et al. 2023, C3)== for rounds; ==energy calibration for implicit CoT (2025, C11)== for descent, plus the energy-descent lineage (Energy Transformer C25, NRGPT C26) — Ekdant anchors descent to the backbone state, not to the peer mean, because the peer-mean pull is provably consensus-preserving. Support: ToT/GoT search structure (C4/C13), least-to-most decomposition (C17), folded reflection (C6/C7), process supervision (C12/C16), self-improve/STaR loops (C14/C15), divergent init (C19), debate dynamics (C9), debate-judge eval (C18), RASC early-stop (C20), latent-agent internalization as future (C10). ==Closest architectural priors== (all fetched locally): multi-stream hypotheses in one model (C21), heads reading each other (C22), function-separated streams at 29M (C23), latent iterative refinement (C29, TRM), energy descent (C25/C26), layered proposal-aggregation (C27), diversity-collapse evidence that forced the floor (C28) — **and C24, the compute-matched negative result** (Rauba et al. 2026, Tiny Autoregressive Recursive Models: the full Autoregressive TRM "shows no reliable gain / no consistent benefit") — see [29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md). **Novelty claimed is the combination plus the M3 write-back, not any single ingredient.**

> **Cite C24 against your own interest.** It is the only paper in this library that runs a *compute-matched* comparison of exactly our design question, and it concludes that the full recursive TRM "shows no reliable gain (**no consistent benefit**)". Do not cite it as support. Cite it as the reason RQ2 is pre-registered with a **depth-matched-no-sandbox** control (same parameters, same block-pass budget, no L2) in addition to K=1, and as the reason a tie is a publishable outcome. What survives from it is narrower and still useful: *stream separation* helped (dual/two-stream), independently corroborated by C23 at 29M — so defend that part, and treat the recursive-refinement part as the hypothesis under test. Cost honesty (load-bearing): ==MAD critique (Smit et al. 2024, C8)== — report tokens×K and FLOPs always. Scope: acting with tools out (ReAct C5, cited to draw boundary). Full vault: [20](../03-Background/20_Thesis_Ch2_Sandbox_Papers.md).

> Single-path CoT is the K=1 control; self-consistency is the matched-budget baseline. Debate + energy descent are the tested improvements, evaluated under the cost-aware critique with overturn and calibration analysis.

### §D Why a dense Transformer backbone (T)

T follows the standard decoder block (Vaswani et al. 2017) in pre-norm form (Xiong et al. 2020; Zhang & Sennrich 2019), with rotary positions (Su et al. 2021, extended per Peng et al. 2023), grouped queries (Shazeer 2019; Ainslie et al. 2023), gated feed-forward (Shazeer 2020), a Mistral-style fixed window (Jiang et al. 2023) computed with stock math in the FlashAttention lineage (Dao 2023) — adapted to XS/S/M dims, MoE deliberately omitted since the sandbox debates instead. Full vault: [25](../03-Background/25_Thesis_Ch2_Transformer_Papers.md).

---

## Ch.3 — Method (cite [02](../02-Architecture/02_System_Architecture.md))

**System.** Four subsystems; model is the contribution: Data (`src/data/`, 12 modules) → Model (`src/ekdant/`, ==5 files==: `model.py`, `ssm_storage.py`, `transformer_stack.py`, `sandbox.py`, `output_head.py`) → Factory (`model_type="ekdant"`) → Training/eval (`src/training/`, `src/infrastructure/`, `src/evaluation/`, `scripts/`).

**L1 — SSM Storage.** `input_ids [B,T]` → tied embed `[B,T,H]` + RoPE (base 100K, window 4096, YaRN-extendable) → N aSSM blocks (shared trunk + 3 L1 state stores: content/control/sandbox — the third *is an L1 store*, not the L2 layer; per-channel Δ/B/C, scan, gating+residual+norm) → `h [B,T,H]` + pooled triple `compressed [B,3·d_state]` handed to T; the L2 consensus write-back returns via M3. Sizes — XS: H=256, d=96, N=2, V=32K · S: H=320, d=128, N=4, V=32K · M: H=512, d=256, N=4, V=32K; T=2048 all.

**Layer T — Dense Transformer.** `h` → N identical blocks (XS: 3, S: 5, M: 6; windowed GQA W=512 + SwiGLU) → `hT`. KV bounded by W. Ablations: XS→S→M ladder, window 256/512.

**L2 - Sandbox.** `hT` → ==K=4 diverse inits== `[K,B,T,H]` → R=2 debate rounds (cross-attention + revise) → ==S=8 anchored gated descent== (`a_t = W_a·hT_t`, per-trajectory gate `g_k`) → consensus `y` (mean default) → ==single dense `[H,V]` head== → logits. Monitors: `anchor_energy`, `trajectory_variance`, `spread_ratio ρ = trajectory_variance / mean(hT²)`. The only L2 training term is the floor `max(0, ρ_min − ρ)²` — zero unless the hypotheses collapse. A pull toward the *running mean* was tried first and **proved inert** (`logits(S=0) == logits(S=8)`); anchor + gate fixes it, and the inert variant is kept as a reported contrast ablation ([29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md)).

**Losses.** `loss = lm_ce + 0.1 * max(0, 0.05 − ρ)²`, dense causal CE, head run once per micro-batch. The L2 term is a dimensionless collapse floor, deliberately **not** an agreement/MSE penalty — such a penalty is minimized by the very collapse it would be meant to detect.

---

## Ch.4 — Data

Registry: ==54 primary (+4 fallback-only) entries, 8 categories==, weights sum to 1.0 (code 0.30, web 0.20, docs 0.15, wiki 0.10, math 0.10, science/books/knowledge 0.05 each). Pipeline (8 stages): registry → streaming with 4 driver families + fallbacks → 8-component quality + exact/fuzzy dedup + contamination filter → AST code filter + function sampling → boilerplate removal → adaptive weighted sampling → EOS-separated packing (==<5% padding==, `_segments`/`_avg_quality` metadata) → tokenization (32K thesis) → SHA256 disk cache → DataLoader. Docs: 18 scrapers; pipeline, stages and registry details: [05_Data_Pipeline.md](../05-Data/05_Data_Pipeline.md).

---

## Ch.5 — Training

Configs `config_ekdant_xs/s/m.yaml`: ==XS ≈12M / S ≈21M / M ≈46M, 1×A100, 50K steps==, lr 1e-4 cosine, warmup 3K, batch 4×accum 4 (eff 16; ~32K tok/step @2048), AdamW fused, wd 0.1, clip 1.0, BF16 `strategy: none`. Stages: pretrain 50K (must) → SFT (nice) → instruct (optional) → DPO-lite. Checkpointing: `models/ekdant/<stage>/` + `checkpoints/`, latest `checkpoint-*` wins. Smoke (Phase 8): 50 steps over the full L1 → T → L2 stack, loss decreases, save/load/resume +10, O(1) shape check. Knobs: [12](../02-Architecture/12_Configuration_Guide.md). Three sizes share one recipe; XS for smoke, S for ablations, M for the main run.

---

## Ch.6 — Evaluation & Results

**Protocol** ([09](../06-Training-Eval/09_Validation_System.md)): corpus audit + 8 phases (registry load, doc build, doc quality, token distribution ±3%, decode inspection, packing <5%, fallback chain, smoke+resume). READY / WARNINGS / NOT READY.

**Tables to fill (placeholders — record real numbers):**

| Check | T=512 | T=2048 | T=4096 | Pass if |
|---|---|---|---|---|
| `compressed` shape | [B,288] | [B,288] | [B,288] | identical |
| Peak memory | — GB | — GB | — GB | flat ±5% |
| tok/s vs. FlashAttn-2 | — | — | — | report |

| Setting | GSM8K-s | HumanEval-s | Tokens | FLOPs |
|---|---|---|---|---|
| Greedy (K=1) | — | — | — | — |
| Self-consistency K=4 | — | — | — | ==matched== |
| Sandbox K=4 R=2 S=8 | — | — | — | ==matched== |
| K=1/8, R=0/3, S=0/4 | — | — | — | ablations |

Overturn per round (wrong→right vs. right→wrong); agreement-vs-accuracy calibration plot (RQ3); step-energy vs. final-vote rerank; 5-benchmark table (HumanEval/GSM8K/MMLU/HellaSwag/ARC).

---

## Ch.7 — Future Work (from [11](../07-Management/11_Future_Work.md))

Critical (defense): 10-step loss-drop test → O(1) proof → sandbox ablations incl. the `sandbox_rho_min` sweep and anchor-vs-peer-mean contrast (report ties honestly) → 50K run + 3 ckpts → corpus gaps. High: SFT mini-run, 5-benchmark + tiny-Transformer baseline, packing numbers, tidy tree, CI pytest. Medium: Triton scan kernel, 4K→16K RoPE test, custom tokenizer. Low/post-thesis: UI/API/quant/LoRA/Docker — deferred. Boundary: tools, multimodality, MoE, internalized debate each need their own RQ. Research ideas: SSM-vs-Transformer backbone; internalized debate distillation (Yi et al. 2026); energy vs. autoregressive reasoning.

---

## Limitations
Small-SSM exact recall (B11); debate collapse/false consensus (watch variance); cost > gain regimes (C8 — report); streaming nondeterminism; single-run perf numbers. Log: [10](../07-Management/10_Known_Issues.md).

## Reproducibility
Pinned reqs + seed 42 + cache; one schema, three configs (xs/s/m); all ~13 knobs in 
[12](../02-Architecture/12_Configuration_Guide.md); commands in [README](../README.md).
