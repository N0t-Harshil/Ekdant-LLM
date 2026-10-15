# STATUS — build status + release readiness

> Merged from the former `16_Project_Status.md` + `17_Release_Readiness_Report.md`. Scores are for a **thesis build**, not a production launch.

## 1. Current build status

**Model code (`09-Code/`, L2-only, implemented):**
- ✅ `09-Code/src/ekdant/`: `model.py`, `transformer_stack.py`, `sandbox.py`, `output_head.py`, `train.py`, `data.py` — pure PyTorch. Forward is L2-only: embed → T (windowed dense) → L2 (sandbox) → head. L1 SSM storage remains a thesis/vault design, not code.
- ✅ `09-Code/configs/sandbox_50m.json` + `baseline_50m.json` — architecture-matched A/B pair (identical N_T=12, H=512, ~50M-class backbone; only `use_sandbox` differs; 50.09M vs 43.53M — the 6.56M delta is L2 itself).
- ✅ `09-Code/tests/`: **78 unit tests passing** (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py`). Both A/B arms trained end-to-end on a T4 (full 3-epoch A/B, `run.md` §6); ops runbook at `run.md` with the 100-epoch campaign recipe (`--epochs 100`, 401,400 steps, §3), live per-step telemetry (loss / lr / grad-norm / ETA), per-epoch **seen-half** eval + `best.pt` selection, and a final seen/unseen/full benchmark protocol (unseen halves scored once, on `best.pt`).
- 🔨 **Full 3-epoch A/B complete** (T4, full data, one seed, per-stage LR fix): MedMCQA val baseline **33.31%** vs sandbox **32.01%**; MedQA test 25.69% vs 26.08% (both ≈ chance — stage 2 is only ~638 optimizer steps). **No significant difference between arms at this scale** (a tie, publishable per the C24 framing). Pipeline-scale result, not thesis numbers (50M class, 1 seed). Runbook: `run.md` §6. **Next run:** the 100-epoch campaign (`runs/*_100ep`, 401,400 steps — measured 6.3 steps/s on the T4 ≈ 17.5 h baseline + ~26 h sandbox ≈ 1.8 days sequential, whole train data, split-half benchmarks) is prepared in `run.md` §3 — if it separates the arms, the follow-up is the 500M-class scale-up (needs a bigger GPU than a T4).
- L1: **design-only** — vault/thesis docs; no `ssm_storage.py` in the code.

**Data pipeline + infra (completed, keep):** registry (54+4, 8 cats), streaming+fallbacks, quality+dedup+contamination, AST+function sampling, packing+WeightedMixedDataset, boilerplate removal, health/sanity, corpus audit, 18 doc scrapers, template-URL hardening, CUDA/PostgreSQL/cuDNN/Linux-kernel fixes, Pydantic schema + migration, tokenizer loading, distributed single-GPU + FSDP path, experiment tracker, 8-phase validation harness, reproducibility utils.

**Thesis build targets:**
- ✅ `19`/`20_Thesis_Ch2_*_Papers.md` + `papers/` library (66 PDFs: 20 SSM + 30 Sandbox + 10 Transformer + 6 training/eval).
- 🔨 Ablation tables (SSM depth, SSM vs. transformer-tiny, K=1/4/8) — need training runs.

**Partially kept (trimmed):** training pipeline (three-stage wiring L1 → T → L2) · alignment (DPO only) · eval (5 benchmarks) · tracking (as-is).

**Out of scope for this thesis:** tool use, multimodality, mixture-of-experts, internalized debate, extra alignment methods, demo UIs, serving infrastructure — each is future work with its own research question (see [11_Future_Work.md](../07-Management/11_Future_Work.md)).

**Remaining build work:** `src/data/pipeline.py` use the registry path only · `scripts/production_validation.py` Phase 8 runs the three-stage smoke · data + model tests exist (78 passing in `09-Code/tests/`).

## 2. Readiness score — 7.1/10 (thesis-ready scaffold)

> Ready to run; not yet defended. **Ready:** data, docs corpus, config skeleton, validation harness, research vault (66 papers), 3-stage spec, model code (L2+T in `09-Code/`, 78 unit tests passing, architecture-matched A/B configs), **full 3-epoch A/B run on a T4 (result: tie within noise — 33.31% vs 32.01% MedMCQA val)**, **100-epoch split-half campaign recipe in `run.md` §3 (whole train data; unseen halves scored once on `best.pt`)**. **To do:** depth-matched control + 3 ablations + multi-seed, 5-benchmark eval, Ch. 6 results; log sandbox diagnostics; resolve L2 size overage; L1 storage remains design-only.

| Category | Score (thesis) | Rationale |
|----------|----------------|-----------|
| Architecture | 8/10 | Two learned layers (L1, L2) joined by a dense Transformer stack, falsifiable, single config. T+L2 code now implemented and tested (L2-only forward; L1 storage design-only). Loses 2 pts: L1 storage not in code, no benchmark numbers yet. |
| Reliability | 7/10 | Data path production-ready; model path unproven until training runs. |
| Maintainability | 8/10 | 5 model files + trimmed schema; tidy tree. |
| Testing | 7/10 | Data + model + schedule + split tests green; **78 unit tests passing** in `09-Code/tests/` (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py`). Loses pts: no training-run/integration tests yet; diagnostics-logging gap open. |
| Validation | 8/10 | 8 phases kept; Phase 8 updated to the three-stage smoke. |
| Documentation (this vault) | 8/10 | After the rewrite + research vault; needs generated API site only post-thesis. |
| Performance | 6/10 | Design O(1) + single-head; no target-hardware numbers yet (thesis must add). |
| Code Quality | 7/10 | Typed, logged; new files must stay <250 lines each. |
| Security | 5/10 | No secrets in repo (`HF_TOKEN` env); the model executes no external tools. |
| Reproducibility | 7/10 | Pinned reqs + seed + cache; streaming nondeterminism noted. |
| **Overall** | **7.1/10 thesis-ready scaffold** | Ready to run; not yet defended. |

## 3. Gaps before defense (in order)

1. L2 parameter overage: measured 6.56M vs 2.8M thesis estimate (HIGH — decide: slim/tie L2, or update the thesis table, before publishing).
2. **A/B result is a tie at pipeline scale** — sandbox 50.09M vs baseline 43.53M shows no significant MedMCQA/MedQA difference (33.31% vs 32.01%; 25.69% vs 26.08%). (a) Report it honestly — a tie is publishable (C24). (b) To *separate* L2 from raw capacity, still open: the **depth-matched control** (param-matched plain Transformer, no L2) and K=1/4/8 + ablations on the thesis configs. (c) MedQA stays at chance (~638 opt steps) — either accept, or plan more MedQA steps/epochs. **Open code item:** log sandbox diagnostics (`anchor_energy` / `trajectory_variance` / `spread_ratio`) into `results.json`. ~~LR-schedule warmup bug~~ **FIXED** (per-stage reset) — `tests/test_schedule.py`.
3. Full validation PASS + 3 checkpoints (HIGH).
4. Ch. 6 results tables (`[PENDING]`) + 5-benchmark eval + baseline comparison (MEDIUM).
5. L1 (`ssm_storage`) integration into `model.py` — currently design-only; unlocks the O(1) proof + L1→T→L2 handoff asserts (HIGH).
6. Tidy tree + CI pytest (MEDIUM).
7. Perf table on A100 (LOW — record even if single run).

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
