> Debugging flows for data, training, and the model (`data` / `train` / `transformer_stack` / `sandbox` / `output_head`; L1 `ssm_storage` is design-only, not in code). Model details live in `02_System_Architecture.md`.

---

# Debugging Guide

All commands run from `09-Code/` after `pip install -e .[dev]`. Logs go to stdout; run artifacts go to `--out` (`runs/<variant>/`: `results.json`, `last.pt`, `best.pt`, `tokenizer.json`, `config.json`).

## v2 (`--task gen`) failures

- **`error: GPU n has only x GB free`** — other users' jobs fill that GPU; choose another with `CUDA_VISIBLE_DEVICES` (check `nvidia-smi`).
- **`expected m1 and m2 to have the same dtype … BFloat16`** — old code: transformers 5 loads SmolLM2 in bf16; `load_backbone` now casts to float32 (T28).
- **`OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files` on the first v2 run** — the empty-cache download was blocked inside `transformers` 4.x (T31, fixed): `git pull`.
- **Sandbox `gnorm` explodes to 10¹¹ / `inf`, loss in the thousands** — the L2 output entered the logits unnormalised (T32, fixed by the output RMSNorm + skip guard): `git pull`, re-run the sandbox arm.
- **`|g|` stays ≈ 0** — the model is not using the add-on. Not a bug; report it.
- **Perplexity gap between arms** — real only if `python -m ekdant.gen_eval compare` shows a 95 % CI that excludes 0.
- **Batched generation differs from one-at-a-time** — position ids must count real tokens (`cumsum(mask) − 1`); covered by `test_generate_batched_equals_single`.

## Common Failures & Diagnosis

### 1. Dataset loading fails or finds nothing
- **Gated / auth error** — set `HF_TOKEN` (and accept the dataset terms on the Hub) if a source requires it. MedMCQA and MedQA are public.
- **`0 examples` / empty dataset** — a schema mismatch in `data.py::_normalise_row` (MedMCQA ships flat `opa..opd` + `cop`, 0-indexed int; MedQA ships an `options` dict). This silently trained on nothing once (T16); `load_examples` now raises when a split is fully unlabelled.
- **Accuracy equals the "A" rate / 0.00 %** — you are scoring the unlabelled MedMCQA public test split (`cop = -1`, T18) or reading the wrong logit position (T17). Score the MedMCQA *validation* split; the evaluator reads `logits[answer_pos - 1]` against the letter *index*.
- **First run downloads, later runs are offline** — `data.py` defaults `HF_HUB_OFFLINE=1`; it goes online for exactly one attempt when the cache is empty. Set `HF_HUB_OFFLINE=0` only to force hub checks.
- **`ConnectionError: Couldn't reach 'openlifescienceai/medmcqa' on the Hub (OfflineModeIsEnabled)` on a fresh machine** — the first-run download was blocked because the online toggle missed the flag newer `datasets` reads (T26, fixed). Pull the latest code. On an old checkout, start the first run with `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python -m ekdant.train …` once to fill the cache.
- **Verify:** `python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 50 --epochs 1 --batch-size 4`.

### 2. Checkpoint / resume failures
- **Symptom:** `--resume` errors or the fresh-run guard refuses to start.
- **Fresh-run guard:** a run into an `--out` that already has `last.pt` aborts unless you pass `--resume` (or use a new directory) — by design, to avoid overwriting a training state.
- **Causes of a bad resume:** a different `--epochs` / `--ft-epochs` / `--ft-lr` / `--answer-weight` than the original run (the epoch counters will not line up), or a changed model config (resume reads the exact `config.json` in `--out`). Keep the original flags.
- **Expected after resume:** optimizer moments restart cold (a small loss bump) and the shuffle generator restarts; the seed-locked 60/40 MedQA partition never changes.
- **`--eval-only`** needs an existing `--out` with `last.pt` and `tokenizer.json`.

### 3. Training divergence
- **Symptom:** loss → NaN / inf, or `gnorm` ≫ 1 sustained (clip threshold is 1.0).
- **Check:** the `--log-every` lines (`loss lr gnorm tok/s eta`).
- **Diagnosis:** lower `--lr` / `--ft-lr`; confirm the fp16 path (a T4 has no bf16, so GradScaler is in use); try `--no-compile`; isolate with `--limit 400 --epochs 1 --batch-size 4`.

### 4. Training looks fine but accuracy does not move
- **Train CE collapses (≈ 0.05) while held-out accuracy plateaus** — memorisation of the small MedQA fine-tune set (T21). Use `--ft-lr 5e-5 --ft-epochs 10`.
- **Accuracy equals the majority rate (MedMCQA val 32.11 %, `medqa_bench` 25.69 %, `medqa_test` 27.73 %)** — the model learned only the label prior; `pred_counts` in `results.json` shows it piled on one letter (T27).
- **Out of memory with PyTorch holding only a few GiB** — other processes share the GPU; check `nvidia-smi` before blaming the batch size.
- **Train CE low, letter accuracy at chance** — supervision dilution (T22): the letter is 1 token in ~34 / ~173. Use `--answer-weight 10`.
- **Sandbox arm only: `spread_ratio` pinned at the floor** — the K trajectories have collapsed (T8) and L2 is doing nothing. Raise `rho_min` first; report it either way. (`spread_ratio` is computed in `aux` but is not yet written to `results.json`.)

### 5. Training stuck at step 0 / progress not moving
- **Symptom:** no `[stage] epN step X/Y …` lines after the `=== stage … ===` header; GPU idle.
- **Cause:** forked DataLoader workers deadlocking on locks inherited at fork time.
- **Fix:** lower `num_workers` in `train.py::build_dataloader` (ships 4 for training, 0 for evals).
- **Inspect without root:** `py-spy dump --pid <pid>` against the `python -m ekdant.train` process.

### 6. Hangs silently right before / inside an eval
- **Symptom:** last log line is an epoch summary; process alive, GPU idle, no output for minutes.
- **Cause (T19, fixed):** every `load_dataset` call revalidated against the Hub (a network HEAD per split per call); on a dead or IPv6-only route the socket sits in `SynSent` forever.
- **Fix (shipped):** offline-first loading in `data.py`. Pull the latest code; confirm with `py-spy dump` if it recurs (a stall parked in a socket read means the offline default was overridden).

### 7. CUDA out of memory
- Reduce `--batch-size` and raise `--grad-accum` to keep the effective batch (e.g. 6 × 8 = 48 on a 16 GB T4).
- The sandbox arm adds a `[K, B, T, H]` scratch tensor (the main activation cost; wall-clock is estimated at ≈ 1.5× the baseline).
- Memory only becomes a concern with near-2048-token inputs; MedMCQA p99 is 118 tokens and MedQA p99 is 364.

### 8. Tests fail
- `pytest -q` (195 tests, CPU, ~10 s). Run one file with `pytest tests/test_model.py -v` (or `test_sandbox.py` / `test_transformer.py` / `test_data.py` / `test_schedule.py` / `test_eval_split.py`).
- A config change that makes `EkdantConfig` and `Sandbox` disagree (e.g. a new readout name) is caught by `test_every_config_valid_readout_builds_a_model`.

### How to reproduce an issue
1. `cd 09-Code && pip install -e .[dev] && pytest -q`
2. Smoke the arm in question: `python -m ekdant.train --variant sandbox|baseline --out /tmp/x --limit 400 --epochs 1 --batch-size 4`
3. Compare against the pilot reference magnitudes in `run.md` §6 (33.31 % / 32.01 % MedMCQA validation after 3 epochs).

## Troubleshooting Flowchart

```
Training fails
    ├── No `[stage] ep0 step X/Y` lines after the stage header?
    │   ├── Stuck on first batch → DataLoader worker deadlock → lower num_workers
    │   └── py-spy dump --pid <pid>
    ├── Stuck silently right before `epN eval`?
    │   └── Pre-fix hub etag stall (T19) → pull offline-first data.py
    ├── NaN loss / gnorm >> 1?
    │   ├── Lower lr / ft-lr
    │   ├── Check fp16 scaler path on T4
    │   └── Re-run the 400-row smoke
    ├── OOM?
    │   ├── Lower batch-size, raise grad-accum
    │   └── Sandbox arm needs more activation memory ([K,B,T,H] scratch)
    ├── Resume refuses / misaligned?
    │   ├── Same --epochs / --ft-epochs / --ft-lr / --answer-weight as the original run
    │   └── Use --resume (fresh-run guard), keep config.json in --out
    └── Accuracy at chance?
        ├── Scoring the unlabelled test split? (T18)
        ├── Train CE collapsed? → --ft-lr / --ft-epochs (T21)
        └── Letter diluted? → --answer-weight 10 (T22)
```
