# STATUS — build status + release readiness

> Merged from the former `16_Project_Status.md` + `17_Release_Readiness_Report.md`. Scores are for a **thesis build**, not a production launch.

## 1. Current build status

**Model code (`09-Code/`, L2-only, implemented):**
- ✅ `09-Code/src/ekdant/`: `model.py`, `transformer_stack.py`, `sandbox.py`, `output_head.py`, `train.py`, `data.py` — pure PyTorch. Forward is L2-only: embed → T (windowed dense) → L2 (sandbox) → head. L1 SSM storage remains a thesis/vault design, not code.
- ✅ `09-Code/configs/sandbox_50m.json` + `baseline_50m.json` — architecture-matched A/B pair (identical N_T=12, H=512, ~50M-class backbone; only `use_sandbox` differs; 50.09M vs 43.53M — the 6.56M delta is L2 itself).
- ✅ `09-Code/tests/`: **70 unit tests passing** (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`). Both A/B arms smoke-verified end-to-end on CPU (train → resume → eval-only); ops runbook at `run.md`.
- 🔨 Real training runs — both A/B arms trained on a T4 (1 epoch × MedMCQA 60k + MedQA). Re-scored after the T18 fix on the **labelled validation split**: MedMCQA baseline **29.90%** vs sandbox **27.96%** (n=4,170; chance 25%); MedQA baseline 24.35% vs sandbox 25.06% (both ≈ chance — stage 2 is only ~212 optimizer steps). Pipeline-validated, not thesis numbers (1 epoch, 60k subset, single seed). Full 3-epoch runs next (the LR bug is fixed, so the full run now uses a correct cosine schedule). Run via `python -m ekdant.train --variant sandbox|baseline --out runs/<v>` (ops runbook: `run.md`).
- L1: **design-only** — vault/thesis docs; no `ssm_storage.py` in the code.

**Data pipeline + infra (completed, keep):** registry (54+4, 8 cats), streaming+fallbacks, quality+dedup+contamination, AST+function sampling, packing+WeightedMixedDataset, boilerplate removal, health/sanity, corpus audit, 18 doc scrapers, template-URL hardening, CUDA/PostgreSQL/cuDNN/Linux-kernel fixes, Pydantic schema + migration, tokenizer loading, distributed single-GPU + FSDP path, experiment tracker, 8-phase validation harness, reproducibility utils.

**Thesis build targets:**
- ✅ `19`/`20_Thesis_Ch2_*_Papers.md` + `papers/` library (66 PDFs: 20 SSM + 30 Sandbox + 10 Transformer + 6 training/eval).
- 🔨 Ablation tables (SSM depth, SSM vs. transformer-tiny, K=1/4/8) — need training runs.

**Partially kept (trimmed):** training pipeline (three-stage wiring L1 → T → L2) · alignment (DPO only) · eval (5 benchmarks) · tracking (as-is).

**Out of scope for this thesis:** tool use, multimodality, mixture-of-experts, internalized debate, extra alignment methods, demo UIs, serving infrastructure — each is future work with its own research question (see [11_Future_Work.md](../07-Management/11_Future_Work.md)).

**Remaining build work:** `src/data/pipeline.py` use the registry path only · `scripts/production_validation.py` Phase 8 runs the three-stage smoke · data + model tests exist (70 passing in `09-Code/tests/`).

## 2. Readiness score — 7.1/10 (thesis-ready scaffold)

> Ready to run; not yet defended. **Ready:** data, docs corpus, config skeleton, validation harness, research vault (66 papers), 3-stage spec, model code (L2+T in `09-Code/`, 70 unit tests passing, architecture-matched A/B configs, both arms smoke-verified: train/resume/eval-only). **To do:** training runs (smoke done → full MedMCQA → MedQA), 3 ablations, eval table, Ch. 6 results; fix LR-schedule bug + log sandbox diagnostics; resolve L2 size overage; L1 storage remains design-only.

| Category | Score (thesis) | Rationale |
|----------|----------------|-----------|
| Architecture | 8/10 | Two learned layers (L1, L2) joined by a dense Transformer stack, falsifiable, single config. T+L2 code now implemented and tested (L2-only forward; L1 storage design-only). Loses 2 pts: L1 storage not in code, no benchmark numbers yet. |
| Reliability | 7/10 | Data path production-ready; model path unproven until training runs. |
| Maintainability | 8/10 | 5 model files + trimmed schema; tidy tree. |
| Testing | 7/10 | Data + model tests green; **70 unit tests passing** in `09-Code/tests/` (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`). Loses pts: no training-run/integration tests yet; LR bug + diagnostics-logging gap open. |
| Validation | 8/10 | 8 phases kept; Phase 8 updated to the three-stage smoke. |
| Documentation (this vault) | 8/10 | After the rewrite + research vault; needs generated API site only post-thesis. |
| Performance | 6/10 | Design O(1) + single-head; no target-hardware numbers yet (thesis must add). |
| Code Quality | 7/10 | Typed, logged; new files must stay <250 lines each. |
| Security | 5/10 | No secrets in repo (`HF_TOKEN` env); the model executes no external tools. |
| Reproducibility | 7/10 | Pinned reqs + seed + cache; streaming nondeterminism noted. |
| **Overall** | **7.1/10 thesis-ready scaffold** | Ready to run; not yet defended. |

## 3. Gaps before defense (in order)

1. Log sandbox diagnostics (`anchor_energy` / `trajectory_variance` / `spread_ratio`) into `results.json` (HIGH). ~~LR-schedule warmup bug~~ **FIXED** (per-stage reset) — LR set per optimizer step, counter resets at each stage boundary, `opt_step` checkpointed (`tests/test_schedule.py`).
2. L2 parameter overage: measured 6.56M vs 2.8M thesis estimate (HIGH).
3. Training runs: both A/B arms trained (1 epoch × 60k; MedMCQA val 29.90% vs 27.96% — barely above chance, pipeline check only). Full 3-epoch runs + 3 ablations still open (HIGH). ~~MedMCQA scoring~~ validated on the labelled val split after T18 fix.
4. Full validation PASS + 3 checkpoints (HIGH).
5. Ch. 6 results tables (`[PENDING]`) + 5-benchmark eval + baseline comparison (MEDIUM).
6. L1 (`ssm_storage`) integration into `model.py` — currently design-only; unlocks the O(1) proof + L1→T→L2 handoff asserts (HIGH).
7. Tidy tree + CI pytest (MEDIUM).
8. Perf table on A100 (LOW — record even if single run).

## 4. File health

| File | Action |
|------|--------|
| `src/data/pipeline.py` 2604 lines | Keep; registry path only |
| `09-Code/src/ekdant/model.py` | Exists; L2-only three-stage wiring (embed → T → L2 → head) |
| `scripts/production_validation.py` 778 lines | Keep; Phase 8 is the three-stage smoke test |
| `src/config/schema.py` 765 lines | Trim to ~300 (EkdantConfig + pipeline configs) |

## 5. Version history (append, don't rewrite)

| Date | Version | Notes |
|------|---------|-------|
| 2025-Q1–2026-Q2 | 0.1–0.6-beta | Early prototypes (see git log) |
| 2026-09 | **0.7-thesis (Ekdant)** | 3-stage spec: SSM Storage + dense T + Sandbox; 66-paper library; single config |
