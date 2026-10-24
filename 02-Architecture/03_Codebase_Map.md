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
- `tests/` — `test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py` (162 tests, CPU)
- `configs/` — `sandbox_50m.json`, `baseline_50m.json` (identical architecture, only `use_sandbox` differs)
- `pyproject.toml`, `pytest.ini`, `README.md`

No `main.py`, no `scripts/`, no YAML size configs. Full shipped signatures: [API_Reference.md](API_Reference.md). Ops runbook (train / resume / test / benchmark): [`run.md`](../run.md).

## v2 modules (current experiment, `--task gen`)

| File | Holds |
|---|---|
| `src/ekdant/backbone.py` | `load_backbone` (SmolLM2-135M, offline-first, cast to float32), `load_tokenizer` (pad falls back to EOS), `tiny_llama` (test stand-in) |
| `src/ekdant/model.py` (v2 part) | `V2Config`, `GatedAddOn` (zero-init gated residual around the L2 sandbox), `EkdantLM` (backbone → add-on → lm_head; answer-only loss + spread floor; KV cache; LR groups) |
| `src/ekdant/gen_data.py` | `GenExample`, `load_pubmedqa`, `load_medquad`, `format_gen_prompt`, `GenDataset` (answer-only labels, 512/1,024 caps), `gen_collate`, `split_by_question`, `holdout` |
| `src/ekdant/gen_eval.py` | `answer_nll`, `by_type`, `rouge_l`, `token_f1`, `score_generations`, `generate_greedy` (left-padded, KV cache), `paired_bootstrap`, `compare` CLI |
| `src/ekdant/train_gen.py` | v2 protocol: GPU pre-flight, zero-shot → PubMedQA → MedQuAD fine-tune → benchmark; exact mid-epoch resume |
| `configs/v2_sandbox_final.json` | reproduces the v2.0 placement (`addon_layer: -1`, L2 after the final norm) |
| `configs/v2_sandbox_multi.json` | v2.2: one shared L2 after layers 10, 20, 30 (T36) |
| `tests/test_multisite.py` | 8 tests for the multi-site L2 (T36) |
| `tests/test_midlayer.py` | 8 tests for the v2.1 mid-layer placement (T35) |
| `requirements.txt` | the dependency pins, identical to `pyproject.toml` (`tests/test_requirements.py`); `transformers<5` for the pod's PyTorch 2.2.2 (T30) |
| `configs/v2_baseline.json`, `configs/v2_sandbox.json` | one config per arm, identical except `addon`; both name the git-ignored token file `configs/hf_token.txt` |
| `tests/test_addon.py`, `test_backbone.py`, `test_gen_data.py`, `test_gen_eval.py`, `test_train_gen.py` | 37 v2 tests (tiny random Llama, no downloads) |

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
