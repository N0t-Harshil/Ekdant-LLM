# Codebase Map (Ekdant thesis cut)

> Canonical spec: `02_System_Architecture.md`. Two views: **what exists now** (`09-Code/`) and the **thesis-target layout** (design — clearly labeled below).

## What exists now — `09-Code/` (shipped, L2-only)

Runnable code lives at `09-Code/src/ekdant/` — a 7-module package. Current forward: `embed → T → L2 → head` (no L1 in the package).

| Module | Responsibility |
|--------|----------------|
| `model.py` | `EkdantConfig` (dataclass), `EkdantModel` (plain `nn.Module`), `ModelOutput`, `baseline_config`, `estimate_model_size` — embed → T → L2 → head |
| `transformer_stack.py` | T: N dense blocks — windowed GQA + SwiGLU (A/B: N_T=12) |
| `sandbox.py` | L2: K trajectories, cross-read debate, anchored descent, consensus + aux (`anchor_energy`, `trajectory_variance`, `spread_ratio`) |
| `output_head.py` | Head: single tied `hidden → vocab` projection |
| `train.py` | Entry point: `python -m ekdant.train --variant sandbox|baseline` (sequential: MedMCQA → test → MedQA 60% fine-tune → 40% benchmark) |
| `data.py` | MedMCQA/MedQA MCQ pipeline (prompt template, letter-token eval) |
| `__init__.py` | Re-exports |

Plus: `tests/` (`test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py` — 85 tests, all green), `configs/` (`sandbox_50m.json`, `baseline_50m.json` — identical architecture, only `use_sandbox` differs), `pyproject.toml`, `pytest.ini`, `README.md`. No `main.py`, no `scripts/`, no YAML size configs in this vault. Full shipped signatures: [API_Reference.md](API_Reference.md). Ops runbook (train/resume/test/benchmark): [`run.md`](../run.md).

> L1 (`ssm_storage.py`, selective SSM) was **removed from the code** — it exists as thesis design in `26a`, `21`, and `02 §3`, not in the current package.

## Thesis-target layout (DESIGN — not the current package)

The sections below describe the thesis build target (root `src/` tree, 5-file model incl. `ssm_storage.py`, scripts, YAML configs). They remain the design reference; they are **not** what is currently in `09-Code/`.

## Root (design)

| File | Purpose (thesis) |
|------|------------------|
| `main.py` | CLI — train / test / chat / config-validate (single `ekdant` arch: L1 → T → L2) |
| `config_ekdant_xs/s/m.yaml` | THE configs — XS ≈12M / S ≈21M / M ≈46M, 1× A100, 32K vocab, 50K steps |
| `PROJECT_DOCUMENTATION.md` / `PROJECT_DESCRIPTION.md` | Thesis narrative (see rewrite notes in Ch. 02) |
| `requirements.txt` | torch, transformers, datasets, pydantic, tokenizers, pyyaml — remove flash-attn/wandb/chromadb/networkx/flask unless used; never mamba-ssm/causal-conv1d/triton (internal scan — see the internal-only rule in [02](02_System_Architecture.md)) |

## `src/data/` — KEEP AS-IS (12 files, production-ready)

`registry.py`, `streaming.py`, `drivers.py`, `metadata_cache.py`, `shards.py`, `pipeline.py`, `quality.py`, `doc_builder.py`, `ast_filter.py`, `function_sampler.py`, `health_reporter.py`, `sanity.py`. No deletions. This is your thesis's data chapter.

## `src/config/schema.py` — TRIM to EkdantConfig

Keep: `Config`, `ModelConfig`, `ModelArchitectureConfig(ekdant only)`, `EkdantConfig(~13 fields)`, `TrainingConfig(pretrain/sft/instruction)`, `DataConfig`, `DistributedConfig(minimal)`, `TokenizerConfig`, `OutputConfig`, `GenerationConfig`, `EvaluationConfig(minimal)`.
Delete: `NSLTConfig`, `MoEConfig`, `VisionConfig`, `MultimodalConfig`, old `EkdantConfig(40+ fields)`, `AlignmentMethodConfig(ORPO/SimPO/KTO)` (keep DPO only), `RLHFConfig`/`SafetyConfig` (minimal honesty probe only).

## `src/ekdant/` — 5 FILES ONLY (design target)

Thesis design: exactly these 5 model files (L1 → T → L2). L1/`ssm_storage.py` is the part **not** in the current `09-Code` package (see build manual `26a`).

| File | Lines (target) | Responsibility |
|------|----------------|----------------|
| `model.py` | ~200 | `EkdantConfig`, `EkdantModel(L1→L2)`, `EkdantForCausalLM` |
| `ssm_storage.py` | ~200 | L1: embed + RoPE + selective SSM stack |
| `transformer_stack.py` | ~150 | T: N identical dense blocks — windowed GQA + SwiGLU, no MoE (S: 5, M: 6) |
| `sandbox.py` | ~200 | L2: K trajectories + debate/energy + consensus |
| `output_head.py` | ~40 | L2 head: single dense projection |
| `__init__.py` | — | Re-exports only these 5 |

The model is exactly these 5 files — there are no other model modules in this architecture.

There is no second architecture in this thesis — Ekdant is the only model.

## `src/training/` / `src/infrastructure/` / `src/models/` / `src/alignment/` / `src/evaluation/` — TRIMMED

- `training/pipeline.py`: keep `run / run_stage / _train_stage / _build_trainer`; delete executive-gate branches, aux-loss wiring, LTC/MoE/MCTS branches, curriculum stages (keep single pretrain schedule + optional SFT).
- `infrastructure/distributed.py`: keep single-GPU + FSDP-full-shard path; delete DeepSpeed/DDP/multi-node.
- `infrastructure/tracking.py`: keep (wandb/mlflow/tensorboard optional).
- `models/factory.py`: `create_ekdant` only + tokenizer loader + `estimate_model_size` (three-stage formula: embed + L1 aSSM + T + L2 + head). Delete `ARCH_CONFIG_MAP` for 6 archs, `ARCH_FSDP_LAYER_MAP` (single entry: `SelectiveSSMBlock`).
- `alignment/`: `dpo_trainer.py` (DPO).
- `evaluation/`: `benchmarks.py` (HumanEval/GSM8K/MMLU/HellaSwag/ARC) + `reporting.py`; `safety.py` holds the honesty probe.

## `src/` root modules — AUDIT

- Keep: `model.py(wrapper)`, `trainer.py`, `validator.py`, `generator.py`, `tokenizer_trainer.py`, `utils/*`.
- Housekeeping (non-model files, only if unused): `dataset.py`(curated examples — move to data/)`, `benchmark.py(custom 100+ problems — keep only if used in eval)`, `knowledge_graph.py`, `massive_data_collector.py`, `synthetic_labels.py`.

## `scripts/` — thesis scripts

- Thesis scripts: `verify_datasets.py`, `production_validation.py` (Phase 8 = L1+T+L2 smoke test), `corpus_audit.py`.

## `tests/` — REWRITE MAP (design target; shipped tests are the 3 files in `09-Code/tests/`)

- Data, quality, config, and validation tests as-is.
- `test_ekdant.py`: forward shapes, gradient flow through L1/L2, O(1) memory (seq 512 vs. 4096, same peak state), sandbox K = 1 vs. K = 4, 10-step loss decrease.
- Target: fewer files, all green, every test maps to a thesis claim.

## `data/` / `models/` / `docs/` / `notebooks/`

- `data/`, `models/ekdant/`, `docs/` — keep.
- `notebooks/`: keep `setup_check.ipynb`; rewrite `train.ipynb` for the three-stage smoke run.
- Research vault: `19_Thesis_Ch2_SSM_Storage_Papers.md` (20 SSM), `20_Thesis_Ch2_Sandbox_Papers.md` (30 sandbox), `21_Thesis_Ch3_Layer1_SSM_Storage.md`, `22_Thesis_Ch3_Layer2_Sandbox.md`, `25_Thesis_Ch2_Transformer_Papers.md` (10 Transformer), `papers/` library (66 PDFs total), and the L2 novelty audit `29_Prior_Art_and_Novelty_Check.md`.
