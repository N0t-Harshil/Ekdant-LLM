> Debugging flows for data, training, and the model (`transformer_stack` / `sandbox` / `output_head` — L1 `ssm_storage` is design-only, not in code). Model details live in `02_System_Architecture.md`.

---

# Debugging Guide

## Common Failures & Diagnosis

### 1. Dataset Loading Failures
- **Symptom**: `datasets` library raises `GatedDataset` error
- **Cause**: Dataset requires HF access token or approval
- **Check**: Is HF_TOKEN set? Has access been granted at huggingface.co/datasets/{path}?
- **Fix**: Set HF_TOKEN env var or token in config; request access for gated datasets
- **Log location**: Console output from `python -m ekdant.train` / `09-Code/src/ekdant/data.py::load_examples` (the old `streaming.py _handle_load_error()` belongs to the pretrain data pipeline, which is not in this vault)

### 2. Dataset Not Found (404)
- **Symptom**: `bigcode/the-stack-v2-dedup does not exist` error
- **Cause**: URL encoding issue with `#` in config name (C# case)
- **Fix**: Remove C# from STACK_V2_LANGUAGES (already done — that module belongs to the pretrain data pipeline, which is not in this vault)
- **Verify**: `python -m ekdant.train --variant sandbox --limit 50` (no `scripts/verify_datasets.py` in this vault — data loads through `09-Code/src/ekdant/data.py`)

### 3. Documentation Scraper Failures
- **Symptom**: "0 pages" scraped for a source
- **Causes**: 
  - Cloudflare challenge (PyTorch)
  - Site redesign (RFC, PostgreSQL)
  - JavaScript-rendered navigation (Docker, MDN)
  - Timeout (server too slow)
- **Check**: Run scraper individually with `--sources [name]` (scraper scripts are not in this vault)
- **Verify**: Check JSONL output files in data/docs/{source_name}/documents.jsonl (the `data/docs` corpus build is not in this vault either)

### 4. Packing Validation Failures
- **Symptom**: Padding > 5% or quality score too low
- **Check**: Phase 6 output in the production report (8-phase harness script is not in this vault)
- **Diagnosis**: 
  - Too many short sequences → increase packing efficiency
  - Too many long sequences → increase max_seq_length or improve window sampling
- **Fix**: Adjust max_seq_length or improve random_window_sample

### 5. Checkpoint Load Failures
- **Symptom**: Model fails to resume from checkpoint
- **Check**: Phase 8 output for checkpoint_found and resume_return_code (Phase 8 = SKIPPED here — harness script absent)
- **Causes**: 
  - Incompatible config change between save and resume
  - Missing optimizer state keys
  - FSDP world size mismatch
- **Fix**: Ensure identical config, world size, and FSDP settings

### 6. Training Divergence
- **Symptom**: Loss goes to NaN or infinity
- **Check**: Gradient norms, learning rate, mixed precision settings
- **Diagnosis**:
  - Try max_grad_norm lower (0.5 instead of 1.0)
  - Reduce learning rate
  - Disable mixed precision (use_full_precision)
  - Check for corrupted data samples
- **Verify**: Re-run with `python -m ekdant.train --variant sandbox --limit 400 --epochs 1 --batch-size 4` to isolate (no `--smoke-steps` flag exists here)

### 7. Training Stuck at Step 0 / Progress Bar Not Moving
- **Symptom**: Progress bar pinned at `0/50000` for hours; GPU idle; main thread futex-waiting
- **Cause**: Forked DataLoader worker processes inherit internal locks held by the async-prefetch/streaming threads at fork time; every worker deadlocks on its first batch fetch
- **Fix**: drop or lower `num_workers` in `09-Code/src/ekdant/train.py::build_dataloader` (ships `num_workers=4`, `persistent_workers=True`); the `dataloader_num_workers=0` pretrain fix belongs to `src/training/pipeline.py`, which is not part of this vault
- **Verify**: `09-Code/src/ekdant/train.py` logs `=== stage ... ===` then `[stage] epN loss=` lines — expect one within seconds of the first `load_examples` download completing

### 8. Dumping Thread Stacks Without Root (`kill -USR1 <pid>`)
- **Symptom**: Need to know where a stuck training process is blocked but have no ptrace/sudo access
- **Fix**: `kill -USR1 <pid>` dumps ALL thread stacks only if a SIGUSR1 handler is installed — the `main.py` handler older notes refer to is **not shipped in this vault**; use `py-spy dump --pid <pid>` or a debugger against `python -m ekdant.train` instead
- **Use with**: Silent stalls where the process is alive but throughput is zero

### 9. `[TRAIN] step 0/N ... awaiting first batch` Heartbeat Line
- **Symptom**: New workers want a health signal for a possibly-stuck trainer
- **Cause**: A `_TrainHeartbeatCallback` (in the pretrain pipeline, **not present in this vault**) logs this line immediately when training starts, then a heartbeat every 100 steps; `09-Code/src/ekdant/train.py` does not emit it
- **Read**: The first line means the training loop entered but the first batch has not arrived yet (data pipeline stall). A heartbeat cadence every 100 steps makes silent stalls visible within seconds

### Where Logs Are Stored
- Console: stdout/stderr (always visible)
- File: the shipped trainer writes to `--out` (`runs/<variant>/results.json`, `last.pt`) — `cfg.output.log_dir`/`logs/` belongs to the thesis pretrain pipeline, not in this vault
- Experiment tracker: none in the shipped A/B trainer (wandb/mlflow/tensorboard via `cfg.tracking` is thesis-pipeline design)

### How to Reproduce Issues
- The standard reproduction sequence (`scripts/final_verify.sh`) and the `production_validation.py` phases are **not in this vault** — reproduce with the runnable workflow below
- For dataset issues: `python -m ekdant.train --variant sandbox --limit 50` (loads MedMCQA/MedQA through `09-Code/src/ekdant/data.py`)
- For pipeline issues: the smoke run + `pytest -q` in `09-Code/`

### How to Verify Fixes
1. `cd 09-Code && pip install -e .[dev]`
2. Check dataset access with `python -m ekdant.train --variant sandbox --limit 50` (set `HF_TOKEN` if a source is gated)
3. Full validation: the 8-phase harness script is absent here (Phase 8 SKIPPED) — use the smoke run for now
4. Run specific test: `pytest tests/test_model.py -v` (or `test_sandbox.py` / `test_transformer.py`)

### Troubleshooting Flowcharts

#### Dataset Loading Flowchart
```
Dataset Load Error
    ├── Gated error?
    │   ├── Yes → Is HF_TOKEN set?
    │   │       ├── Yes → Request access at HF Hub
    │   │       └── No → Set HF_TOKEN in config or env
    │   └── No → Continue
    ├── "does not exist" error?
    │   ├── Check path for special characters (#, etc.)
    │   ├── Check dataset name spelling
    │   └── Verify dataset still exists on HF Hub
    ├── Timeout?
    │   ├── Increase datasets timeout
    │   └── Check network connectivity to HF Hub
    └── Other error?
        ├── Check datasets library version (need 2.20+)
        ├── Python-script-based datasets don't work without trust_remote_code
        └── Check for arrow/parquet file corruption
```

#### Training Failure Flowchart
```
Training fails
    ├── Stuck at step 0 / progress bar not moving?
    │   ├── No `=== stage ... ===` / `[stage] epN loss=` print after the dataset download? (`ekdant.train` ships no heartbeat line)
    │   ├── If stuck on first batch: forked DataLoader worker deadlock (locks inherited at fork)
    │   │   └── Lower `num_workers` in `09-Code/src/ekdant/train.py::build_dataloader` (ships 4)
    │   ├── Dump thread stacks with `py-spy dump --pid <pid>` (no `main.py` SIGUSR1 handler here)
    │   └── Confirm the `[stage] epN loss=` line advances
    ├── NaN loss?
    │   ├── Reduce learning rate
    │   ├── Increase gradient clipping (lower max_grad_norm)
    │   ├── Check for corrupted data samples
    │   └── Disable mixed precision
    ├── OOM (CUDA out of memory)?
    │   ├── Reduce batch_size
    │   ├── Enable gradient checkpointing
    │   ├── Reduce max_seq_length
    │   └── Use FSDP with CPU offload
    ├── Checkpoint save fails?
    │   ├── Check disk space
    │   ├── Check write permissions
    │   └── Reduce save frequency
    └── Slow convergence?
        ├── Increase learning rate
        ├── Check data quality
        ├── Verify weight initialization
        └── Increase warmup steps
```
