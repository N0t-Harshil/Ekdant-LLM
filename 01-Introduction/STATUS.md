# STATUS — build status + release readiness

> Merged from the former `16_Project_Status.md` + `17_Release_Readiness_Report.md`. Scores are for a **thesis build**, not a production launch.

## 1. Current build status

**LoRA + more epochs (2026-10-27, T41):** SmolLM2 frozen with rank-16 LoRA adapters, L2 fully trained; next runs are baseline / end / mix with 3 PubMedQA + 5 MedQuAD epochs (`run.md` A3d); 193 tests.

**The mix built (2026-10-27, T40):** one shared L2 at 9 sites (layers 10, 15, 20–30 every 2nd, and the end) with L2 checkpointing; `run.md` A3c; 186 tests.

**Next (2026-10-27):** `run.md` A3b — seed-23 repeat of the end-placement win, plus single-lever runs (L2 LR 3e-4, gate floor 0.7 (T39), K = 8, `lowest_energy`); 193 tests.

**v2.2 + gate control result (2026-10-27): first clear win.** L2 as the independent final stage: benchmark ppl 3.783 → 3.776 (per-token NLL −0.0017 [−0.0019, −0.0016]), ROUGE-L / F1 +0.006 (above same-seed spread), less forgetting; inner placements helped less (−0.0009 three depths, −0.0003 layer 20). One seed. L1 and L2 declared independent (T38) (`run.md` A7d).

**v2.3 gate control built (2026-10-25, T37):** boost (10× gate LR fading over 1,000 steps), `tanh` cap at 1.0, brake on L2 gradient spikes / size; used by all sandbox runs of the four-run comparison; 193 tests.

**v2.1 result:** a tie (benchmark ppl 3.914 vs 3.913; gate 0.029); the 2nd PubMedQA epoch hurt both arms (`run.md` A7c). **v2.2 built (T36):** one shared L2 at layers 10/20/30; a four-run comparison at 1 epoch (baseline, mid, multi, end) is ready to run (`run.md` A3); 193 tests.

**v2.1 built (2026-10-24, T35):** L2 moved inside SmolLM2 (after layer 20 of 30) so later layers and tokens use it — an extension beyond the design (L1/L2 independent, T38); gate LR 1e-3; 2 PubMedQA epochs; 193 tests. GPU runs pending.

**v2 paired result (2026-10-24): a practical tie.** Benchmark perplexity 3.7817 (baseline) vs 3.7795 (sandbox); per-token NLL −0.0006, 95 % CI [−0.0007, −0.0005] — detectable but 0.06 %; ROUGE-L / F1 within same-seed run-to-run spread; gate |g| = 0.0054 (L2 barely used); +26 % wall-clock, +6.2 GB peak memory. Seed-to-seed variation untested (`run.md` A7a).

**v2 first GPU runs (2026-10-23):** baseline complete — MedQuAD benchmark perplexity 7.578 → **3.782**, ROUGE-L 0.316 / F1 0.380, PubMedQA held-out 9.703 → 7.534 → 8.192 (`run.md` A7b). The sandbox run diverged (T32, fixed); both arms are being re-run as a pair.

**v2 (current, 2026-10-22) — built and tested.** Pretrained SmolLM2-135M + zero-init gated add-on: `baseline` 134.52M, `sandbox` +19.59M L2 (R=4, 4× MLP). Data: PubMedQA → MedQuAD fine-tune half; benchmark = MedQuAD other half (split by question), perplexity primary with a paired bootstrap. 193 tests pass; a CPU smoke run with the real SmolLM2 completed end to end (pipeline check, not a result). Why: every v1 result sat at the majority-letter rate (T27, T28). Spec [30](../07-Management/30_Design_v2_Pretrained_Backbone.md), plan [31](../07-Management/31_Plan_v2_Implementation.md), runbook `run.md` Part A.

**v1 (from-scratch MCQ pilot, kept for reproducibility):**

**Model code (`09-Code/`, T + L2, no L1, implemented):**
- ✅ `09-Code/src/ekdant/`: `model.py`, `transformer_stack.py`, `sandbox.py`, `output_head.py`, `train.py`, `data.py` — pure PyTorch. Forward is T + L2 (no L1): embed → T (windowed dense) → L2 (sandbox) → head. L1 SSM storage remains a thesis/vault design, not code.
- ✅ `09-Code/configs/sandbox_50m.json` + `baseline_50m.json` — architecture-matched A/B pair (identical N_T=12, H=512, ~50M-class backbone; only `use_sandbox` differs; 50.09M vs 43.53M — the 6.56M delta is L2 itself).
- ✅ `09-Code/tests/`: **193 unit tests passing** (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py`). Both A/B arms trained end-to-end on a T4 (full 3-epoch A/B, `run.md` §6); ops runbook at `run.md` with the **sequential protocol + 100-epoch campaign recipe** (train+test on all of MedMCQA → fine-tune on a seed-locked 60% of MedQA → benchmark the untouched 40% once; `--epochs 2 --ft-lr 5e-5 --ft-epochs 10 --answer-weight 10`, 8,884 steps, §3; phase-1 epochs chosen from the validation curve, T25), live per-step telemetry (loss / lr / grad-norm / ETA), per-epoch validation eval + `best.pt` tracking, and final better-of(best,last) checkpoint selection recorded in `results["selected"]`.
- 🔨 **Full 3-epoch A/B complete** (T4, full data, one seed, per-stage LR fix): MedMCQA val baseline **33.31%** vs sandbox **32.01%**; MedQA test 25.69% vs 26.08% (both ≈ chance — stage 2 is only ~638 optimizer steps). **No significant difference between arms at this scale** (a tie, publishable per the C24 framing). Pipeline-scale result, not thesis numbers (50M class, 1 seed). Runbook: `run.md` §6. **Next run (optional):** the campaign recipe in `run.md` §3 — but the legacy 100-epoch baseline below shows phase 1 memorising MedMCQA, so pick `--epochs` from the phase-1 validation curve first (T25).
- 🔨 **100-epoch baseline, legacy recipe** (no `--ft-lr`/`--ft-epochs`, so phase 2 also ran 100 epochs at 3e-4): MedMCQA val after phase 1 **33.33 %** (train CE 0.58 vs val CE 4.61 — memorisation; the 3-epoch run already reached 33.31 %), best fine-tune checkpoint 31.80 % (fine-tune epoch 0), `medqa_bench` **26.89 %**, `medqa_test` **27.02 %** — near chance. Sandbox arm not run on this recipe.
- 🔨 **v1 2-epoch A/B** (`--epochs 2 --ft-lr 5e-5 --ft-epochs 10 --answer-weight 10`): baseline done — MedMCQA val 33.91 % after phase 1 (majority 32.11 %), `medqa_bench` **25.44 %** (majority 25.69 %), `medqa_test` **29.62 %** (majority 27.73 %). Sandbox reached 32.23 % after phase 1, then hit CUDA OOM at the start of phase 2 (shared GPU, other processes held ~75 GiB). Not re-run: v1 was superseded by v2 (T28). All results so far sit within ~2 points of the majority-letter rate, not 25 % chance (T27).
- L1: **design-only** — vault/thesis docs; no `ssm_storage.py` in the code.

**Data pipeline (implemented, tested):** MedMCQA → MedQA loaders with schema normalisation, prompt template, shared BPE tokenizer, seed-locked 60/40 MedQA partition, offline-first dataset reads, letter-accuracy evaluation (`data.py`, `train.py`). The earlier 54-dataset pre-training design (registry, doc scrapers, 8-phase corpus harness) was dropped when the project narrowed to medical MCQ and is not part of any result.

**Thesis build targets:**
- ✅ `19`/`20_Thesis_Ch2_*_Papers.md` + `papers/` library (66 PDFs: 20 SSM + 30 Sandbox + 10 Transformer + 6 training/eval).
- 🔨 Ablation tables (K=1/4/8, debate rounds, descent steps, ρ_min, anchor vs peer-mean) — need training runs; SSM ablations need L1.

**Out of scope for this thesis:** tool use, multimodality, mixture-of-experts, SFT/DPO, general pre-training corpora, internalized debate, demo UIs, serving infrastructure — each is future work with its own research question (see [11_Future_Work.md](../07-Management/11_Future_Work.md)).

**Remaining build work:** run v2 for both arms (`run.md` Part A) · measure v2 time and memory · multi-seed if a CI excludes 0 · L1 · CI (`pytest -q`, 193 tests today).

## 2. Readiness score — 7.0/10 (thesis-ready scaffold)

> Ready to run; not yet defended. **Ready:** data, docs corpus, config skeleton, validation harness, research vault (66 papers), 3-stage spec, model code (L2+T in `09-Code/`, 193 unit tests passing, architecture-matched A/B configs), **full 3-epoch A/B run on a T4 (result: tie within noise — 33.31% vs 32.01% MedMCQA val)**, **100-epoch sequential campaign recipe in `run.md` §3 (MedMCQA-trained/tested first, 60% MedQA fine-tune, untouched 40% benchmark, anti-overfit knobs T21/T22)**. **To do:** depth-matched control + 3 ablations + multi-seed, 5-benchmark eval, Ch. 6 results; log sandbox diagnostics; resolve L2 size overage; L1 storage remains design-only.

| Category | Score (thesis) | Rationale |
|----------|----------------|-----------|
| Architecture | 8/10 | Two learned layers (L1, L2) joined by a dense Transformer stack, falsifiable, single config. T+L2 code now implemented and tested (T + L2 (no L1) forward; L1 storage design-only). Loses 2 pts: L1 storage not in code, no benchmark numbers yet. |
| Reliability | 7/10 | Data path production-ready; model path unproven until training runs. |
| Maintainability | 8/10 | 5 model files + trimmed schema; tidy tree. |
| Testing | 7/10 | Data + model + schedule + split tests green; **193 unit tests passing** in `09-Code/tests/` (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py`). Loses pts: no training-run/integration tests yet; diagnostics-logging gap open. |
| Validation | 7/10 | Unit suite + smoke run per arm + leakage-free held-out protocol (T20); sandbox diagnostics now persisted per epoch (T24); loses points for no CI. |
| Documentation (this vault) | 8/10 | After the rewrite + research vault; needs generated API site only post-thesis. |
| Performance | 6/10 | Design O(1) + single-head; no target-hardware numbers yet (thesis must add). |
| Code Quality | 7/10 | Typed, logged; new files must stay <250 lines each. |
| Security | 5/10 | No secrets in repo (`HF_TOKEN` env); the model executes no external tools. |
| Reproducibility | 7/10 | Seed-locked split, config JSONs, saved tokenizer, resume; one seed, no CI yet. |
| **Overall** | **7.0/10 thesis-ready scaffold** | Ready to run; not yet defended. |

## 3. Gaps before defense (in order)

1. L2 parameter overage: measured 6.56M vs 2.8M thesis estimate (HIGH — the thesis now reports the measured number; decide whether to slim/tie L2 or keep and restate before the final tables).
2. **A/B result is a tie at pipeline scale** — sandbox 50.09M vs baseline 43.53M shows no significant MedMCQA/MedQA difference (33.31% vs 32.01%; 25.69% vs 26.08%). (a) Report it honestly — a tie is publishable (C24). (b) To *separate* L2 from raw capacity: the depth-/parameter-matched control and K=1/4/8 ablations are still open. (c) MedQA stayed at chance (~638 opt steps in that pilot) — the 100-epoch campaign with `--ft-lr`/`--ft-epochs`/`--answer-weight` is the fix. ~~Log sandbox diagnostics into `results.json`~~ **DONE** (T24). ~~LR-schedule warmup bug~~ **FIXED** (per-stage reset) — `tests/test_schedule.py`.
3. 100-epoch campaign for both arms + checkpoints (HIGH).
4. Ch. 6 results tables (`[PENDING]`) + cost report + multi-seed (MEDIUM).
5. L1 (`ssm_storage`) integration into `model.py` — currently design-only; unlocks the O(1) proof + L1→T→L2 handoff asserts (HIGH, but outside the minimum thesis claim if L1 stays as future work).
6. CI `pytest` on push; rebuild `thesis.pdf` from the updated `thesis.tex` (MEDIUM).
7. Perf table on target hardware (LOW — record even if single run).

## 4. File health

| File | State |
|------|-------|
| `09-Code/src/ekdant/model.py` | embed → T → L2 → head; LM + answer-letter + spread-floor loss; config readout names match `Sandbox` |
| `09-Code/src/ekdant/{transformer_stack,sandbox,output_head}.py` | implemented, tested |
| `09-Code/src/ekdant/{data,train}.py` | MedMCQA/MedQA pipeline; sequential protocol, resume, eval-only |
| `08-Thesis/thesis.tex` / `thesis.md` | aligned to the built T + L2 model; L1 marked specified-not-built |
| `08-Thesis/thesis.pdf` | **stale** — predates the alignment; rebuild with `pdflatex thesis.tex` ×3 |

## 5. Version history (append, don't rewrite)

| Date | Version | Notes |
|------|---------|-------|
| 2025-Q1–2026-Q2 | 0.1–0.6-beta | Early prototypes (see git log) |
| 2026-09 | **0.7-thesis (Ekdant)** | 3-stage spec: SSM Storage + dense T + Sandbox; 66-paper library; single config |
| 2026-10 | **0.8-thesis (T + L2 on MedMCQA/MedQA)** | Scope aligned to the built 50M A/B; L1 specified-not-built; docs, config guide and thesis resynced; 94 tests |
| 2026-10-22 | **0.9-thesis (v2: pretrained backbone)** | SmolLM2-135M + zero-init gated L2 (19.59M, R=4, 4× MLP) + same-size MLP control; generative PubMedQA → MedQuAD protocol with perplexity + paired bootstrap; v1 MCQ path kept (`--task mcq`); 131 tests; GPU runs pending (T28) |
| 2026-10-23 | **0.9.1-thesis** | `mlp_control` arm removed (user decision): v2 is baseline vs sandbox, architecture-matched; `transformers` pinned `<5` for the pod's PyTorch 2.2.2 (T30), mirrored in a new `09-Code/requirements.txt`; one config per arm (`configs/v2_baseline.json`, `v2_sandbox.json`) with an optional git-ignored Hugging Face token file; first-download fix for `transformers` 4.x (T31); first GPU runs: baseline healthy, sandbox diverged → L2 output now RMS-normalised before the gate + non-finite-step guard (T32), sandbox arm to be re-run; second sandbox run still exploded → pre-norm in every debate round (T33); add-on LR 1e-3 → 1e-4 and L2 size logged (T34); 145 tests |
