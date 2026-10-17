# Ekdant — Thesis Vault

Specialized reasoning model (T dense Transformer + L2 sandbox; L1 SSM designed in vault, not in code). MSc thesis, Hochschule Heilbronn — Harshil Gorasiya.

**This README is the single map for the vault** — reading paths, folder guide, day-1 commands, and status. There are no hub files.

## Reading paths

- **Thesis:** [01](01-Introduction/01_Project_Overview.md) → [02](02-Architecture/02_System_Architecture.md) → [19](03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md)/[20](03-Background/20_Thesis_Ch2_Sandbox_Papers.md)/[25](03-Background/25_Thesis_Ch2_Transformer_Papers.md) → [21](04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md)/[22](04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) → [05](05-Data/05_Data_Pipeline.md) → [07](06-Training-Eval/07_Training_Pipeline.md) → [thesis.md](08-Thesis/thesis.md)
- **Builder:** [02](02-Architecture/02_System_Architecture.md) → [03](02-Architecture/03_Codebase_Map.md) → [26b](04-Model-DeepDive/26b_Dense_Transformer_for_T.md)/[26c](04-Model-DeepDive/26c_Internal_Sandbox_Build.md) → [12](02-Architecture/12_Configuration_Guide.md) → [09-Code](09-Code/README.md)
- **Presentation:** [project_explanation.md](project_explanation.md) — full project story + 30-min supervisor-meeting order

## Folders

| Folder | Holds | Thesis |
|---|---|---|
| [01-Introduction](01-Introduction/01_Project_Overview.md) | Overview + [STATUS](01-Introduction/STATUS.md) (readiness 7.1/10) | Ch. 1 |
| [02-Architecture](02-Architecture/02_System_Architecture.md) | ==Canonical spec==, codebase map, config guide, API reference | Ch. 3 |
| [03-Background](03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md) | Paper annotations: 20 SSM (B) + 30 sandbox (C) + 10 Transformer (T) | Ch. 2 |
| [04-Model-DeepDive](04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) | L1/L2 working, build manuals 26a/b/c, worked examples, novelty audit | Ch. 3 §1/§2 |
| [05-Data](05-Data/05_Data_Pipeline.md) | 8-stage pipeline, 54-dataset registry, 18 doc scrapers | Ch. 4 |
| [06-Training-Eval](06-Training-Eval/07_Training_Pipeline.md) | Training pipeline, 8-phase validation, debugging, performance | Ch. 5/6 |
| [07-Management](07-Management/10_Known_Issues.md) | Issue log (T1–T15), defense-ordered future work | Ch. 7 |
| [08-Thesis](08-Thesis/thesis.md) | thesis.md + thesis.tex/pdf + [glossary](08-Thesis/GLOSSARY.md) | Draft |
| [09-Code](09-Code/README.md) | **L2-only training code** — pure PyTorch, 81 tests, A/B configs | Build |
| [papers/](papers/00_Papers_Index.md) | 66 PDFs (all thesis-cited) + indexes | Sources |

## Day-1 (new collaborator)

**Full runbook — train, resume, test, benchmark both models: [run.md](run.md).**

```bash
cd 09-Code
pip install -e .[dev]     # note: trailing dot is required
python -c "import ekdant; print('ok')"
pytest -q                    # 81 tests, CPU, ~10s
python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4   # smoke ~2 min
```

Debugging: see [13_Debugging_Guide](06-Training-Eval/13_Debugging_Guide.md). Issues: [10_Known_Issues](07-Management/10_Known_Issues.md).

## Status (honest)

**Problem:** Transformers' KV cache grows with context; monolithic stacks hide which part reasons; confidence is uncalibrated.

**Ekdant answer:** T (windowed GQA + SwiGLU, W=512, no MoE) + L2 sandbox (K=4 trajectories → 2 debate rounds → 8 anchored-gated-descent steps → consensus) + spread floor `max(0, ρ_min − ρ)²`. L1 (selective SSM, O(1) state) is designed in the vault and thesis, not yet in code.

| Done | Not done |
|---|---|
| ✅ L2+T code, 67 green tests, A/B configs (sandbox 50.09M vs baseline 43.53M — only `use_sandbox` differs); both arms smoke-verified end-to-end (train/resume/eval) | ❌ L1 integration |
| ✅ Thesis 26pp, 65 refs, 4 proofs, 0 build errors | ❌ Ch.6 result tables `[PENDING]` (real runs started — baseline trained on Kaggle; eval metric fixed, awaiting both arms' numbers) |
| ✅ 66-paper library, all cited | ❌ Ch. 6 result tables `[PENDING]` |
| ✅ 4 design defects found + proven + fixed | ❌ L2 size open item (6.56M measured vs 2.8M thesis) |
| ✅ Data pipeline + validation harness documented | ❌ LR bug in `train.py`; diagnostics not logged |

Readiness: **7.1/10** — details in [STATUS](01-Introduction/STATUS.md). Fix order: [11_Future_Work](07-Management/11_Future_Work.md).

## Executive summary

**Problem:** Transformers scale badly with context (KV-cache growth); monolithic stacks hide which part reasons; confidence is uncalibrated.

**Ekdant answer:**
1. L1 SSM Storage: selective SSM (linear time, O(1) state), handoff to T. *(vault design)*
2. T Dense Transformer: windowed GQA + SwiGLU (window 512, KV bounded). No MoE.
3. L2 Sandbox: K parallel trajectories + debate/energy consensus + single dense head.
4. Two losses: LM CE + sandbox spread floor `max(0, ρ_min − ρ)²`.

**Strongest features:** production data pipeline; falsifiable design (every knob has an ablation); 66-paper library mapped to thesis chapters; 4 defects caught by internal review before any run.

**Read first:** 01 → 02 (canonical) → 09-Code → 20/22 (L2 papers + working) → 12 (config) → 07 (training).
