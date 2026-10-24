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

**Why.** v2.0 put L2 after SmolLM2's final norm, where nothing reads its output: each token's state is refined in isolation and L2 cannot influence any other token. The paired v2.0 runs tied (benchmark ppl 3.7817 vs 3.7795; gate |g| = 0.005). In the thesis design L2 is not a dead end: the **M3 write-back** feeds L2's consensus into L1's sandbox store, which conditions the processing of the next token. v2 has no L1, so v2.1 approximates that coupling by placing L2 **after decoder layer 20 of 30**: layers 21–30 read L2's output and, through attention, carry it to later tokens. (The coupling differs in kind: attention in the same pass instead of recurrent memory to the next step; the thesis says so.)

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

**Idea (user):** run L2 more than once inside the model, spaced through its depth. The thesis iterates L2 internally (debate rounds, descent steps) and once per token via M3, but never at several depths — so this is an extension, not the original design.

- **One shared L2** after layers 10, 20 and 30 (`configs/v2_sandbox_multi.json`: `"addon_layers": [10, 20, 30]`, `"addon_shared": true`). Parameters 19,598,976 — only 3,456 more than one site — so a gain cannot be credited to extra size. `"addon_shared": false` gives three copies (58,786,560) as a fallback if sharing proves unstable.
- **Per site:** its own zero-init gate and its own input/output RMSNorm, because the stream's scale differs strongly by depth (rms ≈ 6 at layer 10, ≈ 100 at 20, ≈ 2 at 30). All gates in the gate LR group (1e-3). The spread floor applies at every site; diagnostics are averaged; `gate_per_site` is logged.
- **The experiment (`run.md` A3):** four runs with identical flags at 1 PubMedQA epoch — baseline, L2 after layer 20, the multi-site L2, L2 at the end — run one after another.
- **Tests (`tests/test_multisite.py`, 8, + 1 end-to-end):** identity at init; one Sandbox shared; the separate-copies switch; every site's gate gets gradient; KV-cache equality; all gates in one LR group; per-site gate magnitudes; the shipped config; a tiny training run keeps the weights shared after checkpointing.

Connections: why → [10_Known_Issues](10_Known_Issues.md) (T27) · L2 → [22](../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) · runbook → [run.md](../run.md)
