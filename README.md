# Ekdant — Thesis Vault

Does a parallel debate sandbox (L2) help a language model reason? **v2 (current):** L2 sits on a pretrained SmolLM2-135M as a zero-init gated residual and is compared with the plain backbone on generative medical QA (PubMedQA → MedQuAD). **v1 (pilot):** a windowed dense Transformer (T) + L2 trained from scratch on medical multiple-choice QA (MedMCQA → MedQA) — its results sat at the majority-letter rate, which motivated v2. The L1 selective-SSM storage layer is **specified in the vault and thesis but not implemented**. MSc thesis, Hochschule Heilbronn — Harshil Gorasiya.

**This README is the single map for the vault** — reading paths, folder guide, day-1 commands, and status. There are no hub files.

## Reading paths

- **Thesis:** [01](01-Introduction/01_Project_Overview.md) → [02](02-Architecture/02_System_Architecture.md) → [19](03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md)/[20](03-Background/20_Thesis_Ch2_Sandbox_Papers.md)/[25](03-Background/25_Thesis_Ch2_Transformer_Papers.md) → [21](04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md)/[22](04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) → [05](05-Data/05_Data_Pipeline.md) → [07](06-Training-Eval/07_Training_Pipeline.md) → [thesis.md](08-Thesis/thesis.md)
- **Builder:** [02](02-Architecture/02_System_Architecture.md) → [03](02-Architecture/03_Codebase_Map.md) → [26b](04-Model-DeepDive/26b_Dense_Transformer_for_T.md)/[26c](04-Model-DeepDive/26c_Internal_Sandbox_Build.md) → [12](02-Architecture/12_Configuration_Guide.md) → [09-Code](09-Code/README.md)
- **Presentation:** [project_explanation.md](project_explanation.md) — full project story + 30-min supervisor-meeting order
- **AI agents (OpenCode etc.):** [RULES.md](RULES.md) — project facts, hard rules, workflow, code↔vault sync map (auto-loaded via [AGENTS.md](AGENTS.md))

## Folders

| Folder | Holds | Thesis |
|---|---|---|
| [01-Introduction](01-Introduction/01_Project_Overview.md) | Overview + [STATUS](01-Introduction/STATUS.md) (readiness 7.0/10) | Ch. 1 |
| [02-Architecture](02-Architecture/02_System_Architecture.md) | ==Canonical spec==, codebase map, config guide, API reference | Ch. 3 |
| [03-Background](03-Background/19_Thesis_Ch2_SSM_Storage_Papers.md) | Paper annotations: 20 SSM (B) + 30 sandbox (C) + 10 Transformer (T) | Ch. 2 |
| [04-Model-DeepDive](04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) | L1/L2 working, build manuals 26a/b/c, worked examples, novelty audit | Ch. 3 §1/§2 |
| [05-Data](05-Data/05_Data_Pipeline.md) | MedMCQA → MedQA pipeline, split, tokenizer, length audit | Ch. 4 |
| [06-Training-Eval](06-Training-Eval/07_Training_Pipeline.md) | Training pipeline, validation + held-out protocol, debugging, performance | Ch. 5/6 |
| [07-Management](07-Management/10_Known_Issues.md) | Issue log (T1–T27), defense-ordered future work, [v2 design spec](07-Management/30_Design_v2_Pretrained_Backbone.md) (pretrained backbone + generative QA) and its [implementation plan](07-Management/31_Plan_v2_Implementation.md) | Ch. 7 |
| [08-Thesis](08-Thesis/thesis.md) | thesis.md + thesis.tex/pdf + [glossary](08-Thesis/GLOSSARY.md) | Draft |
| [09-Code](09-Code/README.md) | **T + L2 training code** — pure PyTorch, 193 tests, A/B configs | Build |
| [papers/](papers/00_Papers_Index.md) | 66 PDFs (all thesis-cited) + indexes | Sources |

## Day-1 (new collaborator)

**Full runbook — train, resume, test, benchmark both models: [run.md](run.md).**

```bash
cd 09-Code
pip install -e .[dev]     # note: trailing dot is required (pins: 09-Code/requirements.txt)
python -c "import ekdant; print('ok')"
pytest -q                    # 193 tests, CPU, ~10s
python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4   # smoke ~2 min
```

Debugging: see [13_Debugging_Guide](06-Training-Eval/13_Debugging_Guide.md). Issues: [10_Known_Issues](07-Management/10_Known_Issues.md).

## Status (honest)

**Problem:** Transformers' KV cache grows with context; monolithic stacks hide which part reasons; confidence is uncalibrated.

**Ekdant answer:** T (windowed GQA + SwiGLU, W=512, no MoE) + L2 sandbox (K=4 trajectories → 2 debate rounds → 8 anchored-gated-descent steps → consensus) + spread floor `max(0, ρ_min − ρ)²`. L1 (selective SSM, O(1) state) is designed in the vault and thesis, not yet in code.

**The experiment (v2):** two arms on SmolLM2-135M — `baseline` (134.52M) and `sandbox` (+19.59M L2, architecture-matched) — identical except L2, trained on PubMedQA (question + abstract → conclusion), fine-tuned on half of MedQuAD, and benchmarked once on the other half (split by unique question) by reference-answer perplexity with a paired-bootstrap 95 % CI, plus ROUGE-L / token F1. Built and tested; GPU runs pending ([run.md](run.md) Part A).

**v1 pilot:** an architecture-matched A/B — sandbox 50.09M vs baseline 43.53M, identical except `use_sandbox` — trained on all of MedMCQA, fine-tuned on a seed-locked 60 % of MedQA, and benchmarked on the untouched 40 % plus the official MedQA test.

| Done | Not done |
|---|---|
| ✅ v2.2 four runs with gate control: L2 as the final stage wins on every measure (ppl 3.783 → 3.776, ROUGE-L/F1 +0.006); inner placements less ([run.md](run.md) A7d) | ❌ second seed to confirm |
| ✅ v2.1 run: L2 after layer 20 — a tie (3.914 vs 3.913), 2nd epoch hurt both (T35/T36) | ❌ v2.2 four-run comparison incl. one shared L2 at layers 10/20/30 (`run.md` A3) |
| ✅ v2 paired runs done: a practical tie — benchmark ppl 3.7817 vs 3.7795 (−0.06 %, CI excludes 0), generation tie, gate \|g\| = 0.005 ([run.md](run.md) A7a) | ❌ seed-to-seed variation (a second seed for both arms) |
| ✅ v2 built: SmolLM2-135M + gated L2, generative data + metrics, 193 green tests, CPU smoke with the real backbone | ❌ v2 GPU runs for both arms (`run.md` Part A) |
| ✅ T + L2 code, 193 green tests, A/B configs; both arms smoke-verified end-to-end (train/resume/eval) | ❌ L1 integration (RQ1 untested) |
| ✅ Full 3-epoch A/B on a T4: MedMCQA val 33.31 % vs 32.01 %, MedQA 25.69 % vs 26.08 % — a tie; legacy 100-epoch baseline: 33.33 % / `medqa_bench` 26.89 % | ❌ 100-epoch sequential campaign for both arms (prepared in `run.md` §3) |
| ✅ Thesis aligned to the built model (T + L2, MedMCQA/MedQA); 4 proofs, 70 refs | ❌ Ch. 6 result tables `[PENDING]`; `thesis.pdf` needs a rebuild |
| ✅ 66-paper library, all cited | ❌ Depth-/parameter-matched control, K/R/S ablations, multi-seed |
| ✅ 4 design defects found + proven + fixed | ❌ Phase-1 epoch count to re-pick from the validation curve (T25) |
| ✅ Leakage-free protocol (T20), training-loop fixes (T21–T23), sandbox readout/diagnostics fixes (T24) | ❌ L2 size open item (6.56M measured vs 2.8M estimate) |

Readiness: **7.0/10** — details in [STATUS](01-Introduction/STATUS.md). Fix order: [11_Future_Work](07-Management/11_Future_Work.md).

## Executive summary

**Problem:** Transformers scale badly with context (KV-cache growth); monolithic stacks hide which part reasons; confidence is uncalibrated.

**Ekdant answer:**
1. L1 SSM Storage: selective SSM (linear time, O(1) state), handoff to T. *(vault design, future work)*
2. T Dense Transformer: windowed GQA + SwiGLU (window 512, KV bounded). No MoE. *(built)*
3. L2 Sandbox: K parallel trajectories + debate/anchored-descent consensus + single dense head. *(built)*
4. Losses: LM CE + answer-letter CE + sandbox spread floor `max(0, ρ_min − ρ)²`.

**Strongest features:** falsifiable design (every knob has an ablation); 66-paper library mapped to thesis chapters; 4 defects caught by internal review before any run; a pre-committed willingness to report a tie.

**Read first:** 01 → 02 (canonical) → 09-Code → 20/22 (L2 papers + working) → 12 (config) → 07 (training).
