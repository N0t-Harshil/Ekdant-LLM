# 30 — Design v2: pretrained backbone + generative medical QA

> **Result (2026-10-24):** a practical tie — benchmark ppl 3.7817 vs 3.7795, per-token NLL −0.0006 [−0.0007, −0.0005], gate |g| = 0.005 (`run.md` A7a).
>
> Status: **built and tested** (2026-10-22; plan [31_Plan_v2_Implementation.md](31_Plan_v2_Implementation.md)); GPU runs pending. Measured: backbone 134,515,008 params; sandbox add-on 19,594,944 (total 154,109,952). This note is the record of why v2 exists ([T28](10_Known_Issues.md)).

> [!note] **2026-10-23: the `mlp_control` arm was removed at the user's request.** v2 is a two-arm A/B (`baseline` vs `sandbox`). Consequence, stated in the thesis limitations: the comparison is architecture-matched, not parameter-matched (+19.59M for the sandbox arm). Same day: one config file per arm (`configs/v2_baseline.json`, `configs/v2_sandbox.json`, identical except `addon`) replaced the shared `v2_smollm2_135m.json`, and an optional Hugging Face token is read from the git-ignored `configs/hf_token.txt` (never stored in a config). Also: `transformers` pinned `<5`, because 5.x needs PyTorch ≥ 2.6 and the pod has 2.2.2 (T30).

## 1. Why v2

The v1 experiment trained a 43.53M model from scratch on MedMCQA (~6M tokens per epoch). Every result sat at the **majority-letter rate** (MedMCQA val 32.11 %, `medqa_bench` 25.69 %, `medqa_test` 27.73 %; T27). 2-epoch baseline: 33.91 % / 25.44 % / 29.62 %. A model that knows no medicine cannot show whether L2 helps, so v1 could only ever produce an uninformative tie.

v2 fixes the cause, not the symptom: a **pretrained backbone** that already knows language, a **generative** task where every answer token is scored, and **low-noise measures** (reference-answer perplexity with a paired bootstrap).

## 2. Decisions (what the user chose)

| Decision | Choice | Source |
|---|---|---|
| Backbone | **SmolLM2-135M** (`HuggingFaceTB/SmolLM2-135M`, Apache-2.0) | user, 2026-10-22 |
| Task | generative medical QA (write the answer), replacing MCQ | user |
| Datasets | A = PubMedQA `pqa_artificial` (train); B = MedQuAD, half fine-tune / half benchmark | user |
| L2 size | scaled to the same share of the model as v1 (v1: 6.56M of 50.09M = 13.1 %) | user |
| Thesis scope | may change; the thesis is about L2, not the backbone | user |
| Dependency | `transformers` added (needed to load the backbone) | user (asked for a pretrained model) |
| MCQ path | kept behind `--task mcq` so v1 results reproduce | design |

## 3. Models (two arms, trained identically)

```
input ids ─► SmolLM2-135M (30 layers, H=576, 9 heads / 3 KV, vocab 49,152, tied head)
           ─► last hidden state h            [B,T,576]
           ─► add-on block (arm-specific)    h' = h + g ⊙ (RMSNorm(f(h)) − h),  g ∈ R^576, g = 0 at init
           ─► SmolLM2 lm_head (tied)          logits
```

| Arm | `f(h)` | Added params (measured with the current `Sandbox` code) |
|---|---|---|
| `baseline` | none (`h' = h`) | 0 |
| `sandbox` | L2 consensus: K=4, **R=4 debate rounds, debate MLP 4×**, S=8 anchored descent, mean readout, spread floor ρ_min=0.05, λ_sbx=1.0 | **19.59M** (12.7 % of 154.1M; v1 was 13.1 %) |

- **Zero-init gate:** at step 0 both arms produce the backbone's logits exactly (tested). Any later difference is caused by training the add-on. `|g|` is logged per epoch; it shows whether the model chose to use the add-on.
- **Architecture-matched, not parameter-matched:** the sandbox arm adds 19.59M parameters and more compute per token. A same-size control arm was designed and then removed on 2026-10-23 at the user's request; the thesis states the limitation.
- **Reading the outcome:** sandbox better than baseline (CI below 0) → L2 helps, with the capacity caveat. CI containing 0 → tie (reported; the thesis pre-committed to this).
- L2 is per-position (debate and descent mix the K trajectories at one token position only), so generation reuses the backbone's KV cache and applies the add-on to each new token's hidden state.
- New config fields: `backbone` (`"ekdant"` = v1 from-scratch model, or a HF model id), `addon` (`none|sandbox`), `sandbox_rounds`, `sandbox_mlp_ratio`. The v1 configs keep their old values.

## 4. Data and protocol

| Set | Rows | Role |
|---|---|---|
| PubMedQA `pqa_artificial` train | 211,269 − 2,000 | **Phase 1**: question + abstract → conclusion (`long_answer`) |
| PubMedQA held-out | 2,000 (seed 17) | phase-1 validation; post-fine-tune retention |
| MedQuAD fine-tune half | 50 % of the 14,345 unique questions, minus its 10 % validation slice | **Phase 2** training |
| MedQuAD validation slice | 10 % of the fine-tune half's questions | checkpoint selection (better of `best.pt` / `last.pt`) |
| MedQuAD benchmark half | the other 50 % of unique questions | **scored once**, never trained on, never used for selection |

Measured facts (SmolLM2 tokenizer): PubMedQA abstract p50 324 / p99 627 tokens, conclusion p50 47 / p99 130 (~85M tokens per epoch). MedQuAD: 16,407 rows, 14,345 unique questions (3,474 rows share a question), answer p50 182 / p90 529 / p99 1,636 / max 5,646 tokens; 1,761 answers exceed 512.

- **Split by unique question** (lower-cased, stripped), seed-locked, identical for all arms. All rows sharing a question land in the same half. The same disease can still appear in both halves under different question types; this is reported, not hidden.
- **Prompts** (base model, no chat template):
  - phase 1: `Question: {q}\nContext: {abstract}\nAnswer:` → ` {long_answer}<eos>`
  - phase 2 / benchmark: `Question: {q}\nAnswer:` → ` {answer}<eos>`
- **Loss on answer tokens only** (prompt tokens are `-100`), plus the spread floor in the sandbox arm.
- **Caps:** answers truncated at 512 tokens (89 % of MedQuAD answers fit whole); abstracts trimmed so the sequence is ≤ 1,024 tokens.
- PubMedQA `pqa_labeled` (1,000 expert rows) shares no PubMed ids with `pqa_artificial` (checked: 0 overlap). It is not used in v2 but stays available as a later extra test.

## 5. Evaluation (all into `results.json`)

Scored at four points: **zero-shot** (pretrained, before any training), after phase 1, every phase-2 epoch (validation slice only), and final (benchmark, once).

1. **Reference-answer perplexity** (primary): mean NLL per answer token, on every row of each set. Also broken down by MedQuAD question type (`information`, `symptoms`, `treatment`, …), because templated NIH answers flatter perplexity.
2. **Paired bootstrap** between arms on the benchmark: per-question NLL difference, 1,000 resamples, 95 % CI. A difference counts only if the CI excludes 0.
3. **ROUGE-L and token F1** on greedy generations (stop at EOS; max 256 new tokens for MedQuAD, 128 for PubMedQA), on fixed seeded subsets: 1,000 MedQuAD benchmark rows, 500 PubMedQA held-out rows. Pure-Python implementations (no new metric library).
4. **Add-on diagnostics:** `|g|` per epoch (all add-on arms); `anchor_energy`, `trajectory_variance`, `spread_ratio` (sandbox arm), over answer tokens only.
5. Cost report (rule C8): params, tokens, wall-clock, peak GPU memory per arm.

## 6. Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, wd 0.1 on backbone matrices; two LR groups |
| LR | backbone **5e-5**; add-on + gate **1e-4** (1e-3 until T34); cosine per phase, 200 warmup steps |
| Precision | bf16 autocast (A100) |
| Phase 1 | **1 epoch** of PubMedQA; validation every 25 % of the epoch. More epochs only if the curve is still falling |
| Phase 2 | **3 epochs** of the MedQuAD fine-tune half; validation each epoch → `best.pt` |
| Batch | `--batch-size 8 --grad-accum 4` = effective 32 sequences, identical for all arms. If the sandbox arm does not fit, all arms move to `4 × 8` (still 32) |
| Seed | 17 (single seed first; multi-seed only if a difference appears) |

Resume, the fresh-run guard and offline-first dataset loading carry over. Optimiser state is now saved in `last.pt`, so a resume continues exactly (v1 did not save it).

## 7. Hardware and how to run (the user's machine)

**4× A100-SXM4-80GB, shared with other users.** Snapshot 2026-10-22 17:41: GPU 0 has ~5.8 GB free (76.1 of 81.9 GB used), GPUs 1/2/3 have ~45/52/42 GB free. The v1 sandbox OOM happened because PyTorch defaults to GPU 0.

- **One arm per GPU, in parallel:** `CUDA_VISIBLE_DEVICES=1 … --addon none`, `=2 … sandbox`, `=3 … mlp`. Three runs finish in the time of one. Wall-clock is noisier on a shared card, so speed is reported as tokens/s plus the caveat, never as the result.
- **Pre-flight check in `train.py`:** log free/total memory of the chosen GPU at start and refuse to start if free memory is below `--min-free-gb` (default 20). `--gpu auto` picks the visible GPU with the most free memory.
- Expected footprint: well under 30 GB per arm (135M model + 19.6M add-on with AdamW ≈ 2.5 GB of states; the rest is activations and the 49,152-wide logits). Measured on the first run, then written into `run.md`.
- Time: not yet measured. Guess for the baseline: under 1 h per PubMedQA epoch; the sandbox arm slower (R=4 rounds). The first `eta=` line replaces the guess.

## 8. Code layout

RULES §105 says no new files under `src/ekdant/` without asking. **This spec asks for three:**

| File | Holds |
|---|---|
| `src/ekdant/backbone.py` | load the HF backbone; expose `hidden(input_ids, mask, past)` and `lm_head` |
| `src/ekdant/gen_data.py` | PubMedQA/MedQuAD loaders, prompts, answer-only labels, question-grouped split |
| `src/ekdant/gen_eval.py` | perplexity, per-type breakdown, greedy generation with KV cache, ROUGE-L, token F1, paired bootstrap |

Changed: `model.py` (add-on block + gate, `backbone` switch), `sandbox.py` (`mlp_ratio` reaches `DebateRound`), `train.py` (`--task gen|mcq`, `--addon`, LR groups, optimiser state in checkpoints, GPU pre-flight), `configs/` (three v2 configs), `pyproject.toml` (`transformers`, version pinned to what the training image supports).

## 9. Tests (CPU, no downloads: a tiny random `LlamaConfig` stands in for SmolLM2)

1. At init, `sandbox` logits equal `baseline` logits exactly.
2. Labels are `-100` on every prompt token; loss counts answer tokens only.
3. The MedQuAD split puts no question in two sets; it is identical across calls with the same seed.
4. ROUGE-L and token F1 give known values on fixed strings; bootstrap CI contains the true difference on a synthetic case.
5. KV-cache greedy generation equals full-forward argmax decoding, with the sandbox add-on on.
6. Only the `baseline` and `sandbox` arms exist.
7. Resume restores optimiser state (loss after resume equals uninterrupted loss on a toy run).
8. All existing v1 tests keep passing (`--task mcq` unchanged).

## 10. Risks

| Risk | Handling |
|---|---|
| Add-on gate stays ≈ 0 | Reported as a finding ("the model did not use L2"); `|g|` is logged |
| A tie | Pre-committed result (C24 framing) |
| Templated MedQuAD answers dominate perplexity | per-question-type breakdown |
| Shared-GPU OOM | pre-flight free-memory check; one arm per non-zero GPU |
| Catastrophic forgetting of phase 1 | PubMedQA held-out scored after phase 2 |
| `transformers` version on the pod | **resolved (T30):** pinned `>=4.45,<5` in `pyproject.toml` and `requirements.txt`; tested on PyTorch 2.2.2 + transformers 4.57.6 |

## 11. Vault updates when built (same change as the code)

README, STATUS, `run.md` (all sections), RULES (facts, hard rules: two arms, perplexity is primary, majority-rate rule kept for `--task mcq`), `project_explanation.md`, 01 overview, 02 architecture + codebase map + config guide + API reference, 22/26c (L2 at R=4, 4× MLP, gated residual), 05 data pipeline (new datasets and split), 07/09/13/14 training-eval, 10 Known Issues (v1 → v2 decision), 11 Future Work, `thesis.tex` + `thesis.md` (backbone and data chapters rewritten; v1 kept as the "from-scratch pilot" that motivated v2), GLOSSARY.

## 12. Out of scope for v2

L1 (still design-only), backbones larger than 135M, instruction/chat tuning, LLM-as-judge scoring, multi-seed (until a difference appears), MCQ benchmarks (kept only as the v1 path).

## 13. v2.1 — L2 inside the backbone (2026-10-24)

**Why.** v2.0 put L2 after SmolLM2's final norm, where nothing reads its output: each token's state is refined in isolation and L2 cannot influence any other token. The paired v2.0 runs tied (benchmark ppl 3.7817 vs 3.7795; gate |g| = 0.005). (At the time this was framed as approximating an M3 write-back from L2 into L1; T38 removed that coupling — L1 and L2 are independent — so v2.1 is an extension beyond the design.) v2.1 places L2 **after decoder layer 20 of 30**: layers 21–30 read L2's output and, through attention, carry it to later tokens. 

```
v2.1 (current):  ids ─► SmolLM2 layers 1–20 ─► h ─► h + g ⊙ RMSNorm(L2(RMSNorm(h))) ─► layers 21–30 ─► final norm ─► lm_head
v2.0 (the tie):  ids ─► SmolLM2 layers 1–30 ─► final norm ─► h ─► h + g ⊙ (RMSNorm(L2(h)) − h) ─► lm_head
```

- **Formula (mid placement):** `h + g ⊙ RMSNorm_out(L2(RMSNorm_in(h)))`, the standard pre-norm residual form. The residual stream at layers 15–25 has rms ≈ 103 with outliers ≈ 2×10⁴ (measured), so L2 reads a normalised input; g = 0 at init keeps both arms identical (tested on the real SmolLM2).
- **Gate learning rate:** the 576-entry gate gets its own LR 1e-3 (`--gate-lr`); L2's matrices stay at 1e-4 (T34). In v2.0 the gate shared 1e-4 and opened only to 0.005.
- **Longer phase 1:** 2 PubMedQA epochs for both arms (the baseline's curve was still falling at 1 epoch); MedQuAD stays at 3.
- **Params:** add-on 19,595,520 (+576 input norm); total 154,110,528.
- **Configs:** `configs/v2_baseline.json` / `v2_sandbox.json` carry `"addon_layer": 20`; `configs/v2_sandbox_final.json` reproduces v2.0 (`-1`).
- **Tests (`tests/test_midlayer.py`):** identity at init; L2's change at token t reaches the predictions of later tokens and never earlier ones, while the v2.0 placement reaches none; KV-cache decoding equals the full forward; bounded against massive activations; layer validation; separate gate LR group; shipped configs.

**v2.1 result (2026-10-24):** a tie — benchmark ppl 3.914 vs 3.913, per-token NLL −0.0002 [−0.0003, −0.0001], gate 0.029 (5× v2.0); generation slightly lower with L2; the 2nd PubMedQA epoch hurt both arms (3.914 vs 3.782).

## 14. v2.2 — the same L2 at several depths (2026-10-24)

**Idea (user):** run L2 more than once inside the model, spaced through its depth. The thesis iterates L2 internally (debate rounds, descent steps) and places it once, as an independent final stage — so this is an extension, not the original design.

- **One shared L2** after layers 10, 20 and 30 (`configs/v2_sandbox_multi.json`: `"addon_layers": [10, 20, 30]`, `"addon_shared": true`). Parameters 19,598,976 — only 3,456 more than one site — so a gain cannot be credited to extra size. `"addon_shared": false` gives three copies (58,786,560) as a fallback if sharing proves unstable.
- **Per site:** its own zero-init gate and its own input/output RMSNorm, because the stream's scale differs strongly by depth (rms ≈ 6 at layer 10, ≈ 100 at 20, ≈ 2 at 30). All gates in the gate LR group (1e-3). The spread floor applies at every site; diagnostics are averaged; `gate_per_site` is logged.
- **The experiment (`run.md` A3):** four runs with identical flags at 1 PubMedQA epoch — baseline, L2 after layer 20, the multi-site L2, L2 at the end — run one after another.
- **Tests (`tests/test_multisite.py`, 8, + 1 end-to-end):** identity at init; one Sandbox shared; the separate-copies switch; every site's gate gets gradient; KV-cache equality; all gates in one LR group; per-site gate magnitudes; the shipped config; a tiny training run keeps the weights shared after checkpointing.

## 15. v2.3 — gate control: open fast, never burst (2026-10-25)

**Why (user):** in v2.0/v2.1 the gate opened slowly (0.005 / 0.029); the user asked for a gate that opens quickly at the start and closes when it starts bursting.

- **Boost:** gate LR × `gate_boost(step)` — 10 at step 0, cosine-faded to 1 at step 1,000 of phase 1 (`--gate-boost`, `--gate-boost-steps`).
- **Soft cap:** effective gate `cap · tanh(θ / cap)` with `cap = 1.0` (`V2Config.gate_cap`; 0 = uncapped, as in v2.0–v2.2). Identity and slope at θ = 0 unchanged (tested).
- **Brake (`GateBrake`):** a step bursts when L2's pre-clip gradient norm > 3 × its running average (after 20 steps) or L2's size > 50; then gates × 0.9, L2 + gate LR factor × 0.5 (floor 0.05), recovering × 1.01 per healthy step. State saved in `last.pt`; events in `cost.brake_events`. Non-finite steps stay with the skip guard.
- **Scope:** all sandbox runs of the four-run experiment use it (mid, multi, end); the baseline has no gate.
- **Tests:** `tests/test_gate_control.py` (8) and `test_brake_fires_on_forced_spike_and_is_recorded`.
- **Result (2026-10-27):** no brake events; L2 at the end −0.0017 [−0.0019, −0.0016] per token (3.783 → 3.776), ROUGE-L/F1 +0.006 — about 3× the uncontrolled end placement (−0.0006). Inner placements less (`run.md` A7d).

## 16. Gate floor — force L2 to contribute (2026-10-27, T39)

Optional (`--gate-floor`, default 0 = off). After `--gate-floor-start` (200) healthy steps a floor `f` ramps by a cosine from 0 to the target (0.7) over `--gate-floor-steps` (2,000) and is held; the effective gate is `f + (1 − f) · tanh(θ)`, so the network can still move it within [2f − 1, 1]. Every brake event moves the ramp back by 10 % of its length (`GateRamp`), so an unstable L2 is never pushed open. The ramp position is saved in `last.pt`. Experiment run 4 of `run.md` A3b, together with a seed-23 repeat of the end placement and three single-lever runs (L2 LR 3e-4, K = 8, `lowest_energy`).

## 17. The mix — L2 inside and at the end (2026-10-27, T40)

One shared L2 at 9 sites: after layers 10, 15, 20, 22, 24, 26, 28, 30 (pre-norm residual, as §13) **plus** after the final norm (the end placement, as v2.0). `V2Config.addon_final = True` adds the end site to `MultiAddOn` (`final_site`); every site has its own zero-init gate and norms; gate control (boost, cap, brake) acts on all 9 gates. 19,608,768 parameters. `sandbox_checkpoint = True` recomputes L2 in the backward pass (non-reentrant `torch.utils.checkpoint`) — tested to give identical loss and gradients — because storing 9 sites' activations would need ~70 GB. Compared against `v22_baseline` and `v22_end` (`run.md` A3c).

## 18. More epochs (2026-10-27, T41)

The next comparison (`run.md` A3d) reruns baseline, end and mix with 3 PubMedQA epochs and 5 MedQuAD epochs (was 1 and 3), identical for all arms. The whole model is trained, as before. LoRA on the backbone was briefly added and removed again at the author's request: the thesis trains all of SmolLM2 together with L2. The earlier 1 + 3-epoch runs are not comparable with these.

## 19. Forced L2 (2026-10-29, T42)

The thesis claim is that L2 should carry real weight in the model, not be an optional extra the network may ignore. Until now every gate started at 0 and was free; the model opened L2 to 0.076 (end) and 0.120 (mix, mean of 9) in the v25 runs. `--gate-force` with `--gate-floor 0.5` makes L2 mandatory: (1) the effective gate is `max(f, f + (1 − f)·tanh θ)` per channel, with a straight-through gradient so θ keeps learning and the gate can open towards 1; (2) the output norm's scale is frozen, so L2's contribution cannot be shrunk to zero behind the gate; (3) a mid site adds `g · rms(h) · RMSNorm(c)`, sized per token to the stream, so g = 0.5 adds half the stream's magnitude; at the end `h + g·(RMSNorm(c) − h)` already means at least half the final state is L2's. The floor ramps in after 200 steps over 2,000 steps; the brake still moves the ramp back on a burst. `gate_force` is stored in the config, so `--eval-only` rebuilds the forced model, and the ramp's settings are read from the checkpoint. Compared against the v25 runs with identical flags (`run.md` A3e).

## 20. Fast opening with rollback (2026-10-29, T43)

`--gate-rollback`: the trainer keeps an in-memory CPU snapshot of the weights, the optimiser state and the ramp position every 100 optimiser steps (and at the start of each stage, so it never goes back into the previous stage). When the brake detects a burst, the trainer restores the snapshot and drops that step's update; the ramp returns to the snapshot's position, takes twice as long to re-open (`--floor-slowdown`) and aims 0.05 lower (`--floor-step-down`), never below 0.8 × the floor (`--gate-floor-min`). The brake still halves L2's learning rate. With this safety net the forced runs open fast: the floor starts at step 100 and reaches 0.5 in 500 steps. The ramp's target, length and rollback count are saved in `last.pt`; `results.json` reports `cost.rollbacks`. Snapshot cost: one copy of the weights and optimiser state (~2 GB for the mix) every 100 steps.

**Fix (2026-11-01, T45).** The first forced mix run hit bursts on consecutive steps; every rollback doubled the ramp length while the floor was measured from the ramp's start, so the floor at the snapshot fell towards 0, the forced minimum no longer held and the model broke. A rollback now restarts the rise from the floor the snapshot had (capped at the lowered target): the floor drops by at most 0.05 and never below 0.8 × floor. At most `--max-rollbacks` (3) rollbacks go to the same snapshot; after that the brake handles bursts until the next healthy snapshot. The brake's running average now also moves on a burst (capped at the spike threshold), so a lasting rise in L2's gradient becomes the new normal instead of a burst on every step.

## 21. Measuring and building dependence on L2 (2026-10-29, T44)

**L2-off test.** The thesis claim is that L2 matters to the model. A gate value cannot show that: a forced gate only guarantees L2 is in the computation, and L2 may learn to reproduce its input or downstream layers may learn to cancel it. After training, every run with L2 therefore scores the MedQuAD benchmark and PubMedQA again with all L2 sites switched off (`EkdantLM.l2_off()`; the model is then exactly its backbone, tested). The importance of L2 is the per-token NLL rise, with a paired bootstrap over questions (`results.json` `test.l2_off.importance`). A CI above 0 means the trained model depends on L2.

**L2 warm-up.** The free gates in v25 first closed: at step 0 L2 is random and can only hurt a pretrained backbone. `--l2-warmup N` freezes SmolLM2 for the first N optimiser steps of phase 1 so that only L2 (with its gates, already forced open by the floor) learns; afterwards the whole model trains as before. The backbone then receives N fewer updates than in the other arms, which works against L2, not for it. Off by default; refused for the baseline.

## 22. Does L2 help the model reason? Three tests on finished runs (2026-11-02, T46, T47)

Perplexity and ROUGE/F1 measure how well the model reproduces reference wording, not whether it reasons. Three evaluation-only tests on every finished run: (1) multiple-choice accuracy on MedMCQA, MedQA and PubMedQA pqa_labeled, the pick being the option with the highest mean log-probability per token in the training prompt format; (2) the same with L2 switched off; (3) thinking time on PubMedQA: the debate stack is run 0 or 2 times instead of once (`Sandbox.debate_passes`; the four rounds have their own weights, so "more rounds" means repeating the stack) and the anchored descent takes 0, S/2, 2S or 4S steps instead of S at the same step size. Evidence for iterative reasoning would be accuracy that falls with less thinking (and possibly rises with more); a flat profile means L2 acts like an ordinary extra layer. Every difference is a paired bootstrap CI over questions. The same run (T48) also rebuilds the MedQuAD benchmark split exactly as training did and scores its perplexity with L2 on, off and at every thinking setting, and records L2's spread ratio and anchor energy, so one run yields every table of the paper.

## 23. L2 v3: reading the text, confidence, per-token gate (2026-11-02, T50)

**What was missing in L2 (found 2026-11-02, T50).** Three gaps between
the design idea ("a sandbox inside the LLM that reasons and is confident
like a human") and the v2 code:

1. **L2 saw one token at a time.** The debate (`CrossRead`) mixes the K
   trajectories *at the same position only*; nothing in L2 looked at the
   other tokens. So L2 could not read the question or the abstract — every
   bit of reading was done by SmolLM2, and L2 was a per-token refinement
   (a fancy MLP), not a place where hypotheses are tested against the text.
2. **No confidence.** The K trajectories were averaged; how much they
   agreed was thrown away, so the model never knew whether its "thoughts"
   agreed.
3. **One gate for every token.** The gate is one number per channel, the
   same for easy and hard tokens: L2 could not count more where it helps.

(The anchored descent stays as it was: it pulls the trajectories towards a
learned anchor of the token's state; with very many steps they collapse onto
it, which is what the thinking-time test shows at 32 steps.)

The v3 changes, all opt-in (`V2Config.sandbox_context`, `sandbox_confidence`): (a) **context read** — each debate round starts with causal multi-head attention of every trajectory over the earlier tokens (shared weights across the K trajectories, zero-init output projection, padding masked); (b) **verifier readout** — the consensus is a per-token softmax-weighted mean of the trajectories, the weights from a zero-init linear verifier (zero = the old mean); (c) **confidence** — c_t = exp(-d_t), d_t the weighted mean squared distance of the trajectories from the consensus relative to its size, detached so it cannot be gamed by collapsing the trajectories (the spread floor still keeps them apart); (d) **per-token gate** — g_t = g * 2*sigmoid(u . norm(consensus_t) + v * c_t), zero-init so it starts as the per-channel gate; under forced L2 the floor holds per token. At step 0 the v3 model equals the backbone exactly. Causal reading keeps the backbone's KV cache exact at a mid site, but L2 has no cache of its own, so generation recomputes the sequence. Size: L2 24.9M parameters (v2 19.6M), so a win must also be weighed against the extra 5.3M.

**Revision (T51, same day, before any v3 run).** (e) **Alignment** — v2 started each trajectory as a random projection of the model's state and anchored the descent to a random rotation of it, then normalised L2's output to size 1: L2 worked in a different space than the model, which the gate had to fight. With `sandbox_aligned` trajectory k = (I + 0.25 W_k) h (identity plus a small random nudge, no bias), the anchor starts as the identity, and L2's output is scaled to the model's per-token size. (f) **Reading with the model** — the context read now uses the model's own states as values (unchanged) and one key projection per forward, shared by all trajectories and rounds; each round adds only a query projection and a zero-init per-channel scale; the K trajectories are folded into the query length. The states are cached with the backbone's KV cache, so cached generation stays exact. (g) **Vote and history** — K = 5 (odd: no ties); the verifier score of trajectory k is own(k) + fit(k, history_t) + vote * support(k), support = mean over the others of exp(-||T_k - T_j||^2 / ||mean||^2), history_t = mean of the model's states up to t. (h) **Learned confidence** — a head on detached inputs (agreement, top verifier weight, vote support, L2's output) is trained with BCE (weight 0.1) to predict whether the model's next token is right; the per-token gate reads it. It changes no gradient of the language model (tested). L2 21.9M parameters.

**Final review (T52).** The aligned identity is stored as identity + a learned delta (weight decay pulls towards the model's state, not 0); the nudge is 0.6 (spread ratio 0.061 on SmolLM2, just above the 0.05 floor; 0.25 gave 0.011, an almost-identical start); the soft cap is applied again after the per-token factor (no overshoot past L2's output at the end placement); the vote's distances come from the Gram matrix; the new L2 signals are logged and saved.

**v3b results and v3c (T54, 2026-11-05).** v3b's confidence tells right from wrong next words (AUROC 0.814 / 0.818, two seeds), the model depends on L2 more than ever (−12.5 PubMedQA points without it), and the per-token gate stays open; but its answers equal v22's (59.9 vs 60.4 %), and a second debate pass at test time costs 6.5 points because L2 was trained with one. v3c trains thinking twice: each training forward runs the debate stack 1 or 2 times at random, so pass 2 learns to refine pass 1, and the model answers with 2 passes. The benchmark now asks what the confidence is worth: against the model's own top-word / option probability, per next word and per question (AUROC; % right on the surer vs less sure half — answer when sure, defer otherwise).

**v3d and the gain report (T55, 2026-11-05).** v3c answered no better than v22 or v3b (A7i). The clearest difference to v22 is the spread of L2's trajectories (v22 0.22–0.46, v3 0.05 = the floor): v3 starts every trajectory at the model's own state and pulls it back there, so the five trajectories act almost as one. v3d (`sandbox_free_views`) keeps every v3b part but gives L2 v22's thinking space: independent starting views and a learned anchor. A `gain` report gives one answer to "how much better than the baseline": the gain pooled over every seed with a 95 % CI, the relative gain and the share of the baseline's errors fixed. Untrained; `run.md` A3m.

**v4: L2 without its limits, and a second benchmark (T56, 2026-11-05).** Three limits of the L2 design were removed: (1) L2 also runs after layer 15 of 30, so 15 later layers build on its result (the same weights also run at the end); (2) inside the network its output is sized to the residual stream, not to its normalised input (rms ~1 against ~100), so its gate actually changes what later layers read; (3) its text memory also holds the model's states of every word at layers 8 and 22 (zero-init weights), information the last layers compress away. A second benchmark asks the model to write: the expert's explanation of a MedMCQA answer and the authors' conclusion of a PubMedQA abstract (both never trained on), scored by reference perplexity, word overlap and whether its explanation names the right answer. Untrained; `run.md` A3n.

**Result of v4 and v5 (T57, 2026-11-07).** v4 was not better than v22 (two seeds: average −0.49\* points, PubMedQA −1.35; `run.md` A7j): using L2 more, inside the network and with more to read, did not give better answers. The final model, v5, is therefore the v22 recipe unchanged, checked for reproduction on seed 17 and extended with a third seed. One new idea is tested on it: **L2 dropout** (`--l2-drop 0.3`). The trained v22 model is worse than its own baseline when L2 is switched off, so the backbone has handed part of its job to L2 and L2 partly replaces it instead of adding to it. Switching L2 off for 30 % of the training steps keeps the backbone good alone, so L2 is kept only for what it adds, as dropout and stochastic depth make units add instead of co-adapt. Untrained; `run.md` A3o.

**The output-scale gap (T58, 2026-11-07).** A review of the v22 L2 found that at the end its output starts at half the model's size: L2's output norm starts at 1, SmolLM2's final norm at ~1.8 (final state rms ~2.1). In `h + g*(y - h)` the difference `y - h` then points almost at `-h` (cosine 0.90), so the gate mostly scales the model's own state down instead of mixing in L2's result, and the gate settled at ~0.02. `sandbox_out_from_backbone` starts L2's output norm as a copy of the model's final norm (size 0.89x, cosine 0.75), keeping step 0 = SmolLM2. It is tested alone on the v22 recipe (`v5out`, three seeds), because the v3 version of this idea (`sandbox_aligned`) came together with several other changes.

**Result of v5, L2 dropout and the output-scale fix (2026-11-08, `run.md` A7k).** v5 reproduces v22 and on three seeds gives PubMedQA +2.07\* points, average +0.72\*, explanation ppl −1.78 %\* over the baseline. Neither change helps: L2 dropout −1.10 on PubMedQA (CI includes 0) and +1.21 %\* explanation ppl against v5; the output-scale fix −0.93\* on PubMedQA and −0.35\* on average. Two design lessons: (1) with dropout, switching L2 off costs almost nothing (+0.002 NLL instead of +0.07–0.09), so the large L2-off drop of v22 measured the backbone's reliance on L2, not what L2 adds; (2) with L2's output at the model's scale the gate still settled at ~0.03, so the small gate is not caused by the output scale.

**Train L2 first (T44 warm-up, T60, 2026-11-08).** In every run so far L2 and SmolLM2 learned together from step 0: L2 starts random, and the layers after it change while it does. The next runs train L2 alone for 2,000 steps with SmolLM2 frozen, so the layers after L2 cannot move and L2 must learn to give them what they accept; then the whole model learns, or (at the author's request) L2 plus LoRA adapters while SmolLM2's own weights stay fixed. After layer 20 (10 layers follow L2) and at the end, three seeds each (`run.md` A3q).

**First v3 run and v3b (T53, 2026-11-05).** `v3_end` gave the best perplexity of all models (3.764, −0.5 %) and made the model depend on L2 twice as much as v2 (−11.3 PubMedQA points without it) but answered no better (58.3 %). Its log showed the per-token gate falling to 0.03 during PubMedQA (L2 switched off for the reasoning phase) and the debate collapsing into agreement (vote 0.69 → 0.83, spread at the floor). v3b bounds the per-token factor to (0.5, 2) and gives each trajectory its own learned query offset, so the trajectories ask the text different questions; the brake re-learns its baseline at each stage; the confidence's AUROC on right vs wrong next tokens is measured.

Connections: why → [10_Known_Issues](10_Known_Issues.md) (T27) · L2 → [22](../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) · runbook → [run.md](../run.md)
