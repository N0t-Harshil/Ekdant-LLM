# Codebase Map (Ekdant)

> Canonical spec: `02_System_Architecture.md`. This map describes what is in the repo **now**; the one planned extension (L1) is listed at the end.

## `09-Code/` — the runnable package (L2 + T)

Runnable code lives at `09-Code/src/ekdant/` — a 7-module, pure-PyTorch package. Forward: `embed → T → L2 → head` (no L1 in the package).

| Module | Responsibility |
|--------|----------------|
| `model.py` | `EkdantConfig` (dataclass), `EkdantModel` (plain `nn.Module`), `ModelOutput`, `baseline_config`, `estimate_model_size` — embed → T → L2 → head; LM loss + optional answer-letter loss + spread floor |
| `transformer_stack.py` | T: N dense blocks — windowed GQA + SwiGLU + RoPE + RMSNorm (A/B: N_T=12) |
| `sandbox.py` | L2: K trajectories, cross-read debate, anchored descent, consensus + aux (`anchor_energy`, `trajectory_variance`, `spread_ratio`) |
| `output_head.py` | Head: single tied `hidden → vocab` projection |
| `train.py` | Entry point: `python -m ekdant.train --variant sandbox|baseline` (sequential: MedMCQA → validation score → MedQA 60 % fine-tune → untouched 40 % benchmark) |
| `data.py` | MedMCQA/MedQA MCQ pipeline (prompt template, tokenizer, letter-token eval) |
| `__init__.py` | Re-exports |

Plus:
- `tests/` — `test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py` (92 tests, CPU)
- `configs/` — `sandbox_50m.json`, `baseline_50m.json` (identical architecture, only `use_sandbox` differs)
- `pyproject.toml`, `pytest.ini`, `README.md`

No `main.py`, no `scripts/`, no YAML size configs. Full shipped signatures: [API_Reference.md](API_Reference.md). Ops runbook (train / resume / test / benchmark): [`run.md`](../run.md).

## Vault layout

| Folder | Holds |
|--------|-------|
| `01-Introduction/` | overview + `STATUS.md` |
| `02-Architecture/` | canonical spec, this map, config guide, API reference |
| `03-Background/` | paper annotations (20 SSM, 30 sandbox, 10 Transformer) for Ch. 2 |
| `04-Model-DeepDive/` | L1/L2 working, build manuals 26a/b/c, worked examples, novelty audit |
| `05-Data/` | MedMCQA/MedQA data pipeline |
| `06-Training-Eval/` | training pipeline, validation, debugging, performance |
| `07-Management/` | issue log, future work |
| `08-Thesis/` | `thesis.tex` / `thesis.md` / `thesis.pdf`, glossary, figures |
| `09-Code/` | the package above |
| `papers/` | 66 PDFs + indexes |

## Planned, not built — L1

`ssm_storage.py` (selective SSM front end, three stores, M3 write-back) is specified in `26a`, `21` and `02 §3`. Adding it means: a new module, an `n_ssm_layers` / `d_state` config, an L1 → T handoff shape assertion (Known Issues T9), and the O(1)-state proof. Until then the memory claim (RQ1) is design only.
