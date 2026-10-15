# Training Pipeline (Ekdant) — Ch. 5

> Map: [README](../README.md) · Runbook: [run.md](../run.md) · Spec: [02](../02-Architecture/02_System_Architecture.md) · Builds: [26a](../04-Model-DeepDive/26a_Internal_SSM_Build.md) · [26b](../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) · [26c](../04-Model-DeepDive/26c_Internal_Sandbox_Build.md) · Config: [12](../02-Architecture/12_Configuration_Guide.md)

`09-Code/src/ekdant/train.py` orchestrates Ekdant training (`python -m ekdant.train --variant sandbox|baseline`, MedMCQA → MedQA): one architecture, two losses (LM CE + spread floor), A/B pretrain only. The SFT/instruct/DPO stages in §1 are thesis design, not shipped.

## 1. Thesis design spec (Ch. 5 — documented, not all runnable)

Config targets `config_ekdant_xs/s/m.yaml` (one schema, three sizes): XS ≈12M / S ≈21M / M ≈46M, 1× A100 80GB. **These YAMLs are not runnable files in this vault** — they are the thesis spec. The runnable configs are `09-Code/configs/{sandbox,baseline}_50m.json` (the 50.09M/43.53M A/B), loaded by `ekdant.train`.

- **Arch (`ekdant`):** XS: H=256/V=32K/d=96/N_ssm=2/N_T=3/K=4 · S: H=320/V=32K/d=128/N_ssm=4/N_T=5/K=4 · M: H=512/V=32K/d=256/N_ssm=4/N_T=6/K=4 · all: rounds=2, sandbox_steps=8, seq 2048, rope_theta 100K, tie embeddings true.
- **Train:** 50K steps, lr 1e-4 cosine, warmup 3K, batch 4 × accum 4 (eff 16), AdamW fused, wd 0.1, clip 1.0, BF16 single-GPU (`strategy: none`).
- **Data:** registry mode, 54 primary / 8 cats, 8-stage step plan optional (or single 50K run for thesis simplicity).
- **Losses:** `lm_weight=1.0`, `sandbox_weight=0.1`, `sandbox_rho_min=0.05` (shipped λ differs — §3). No `head_ce` flag (dense only). No aux dict beyond the three reported monitors.
- **Stages:** 1. **Pretrain (must):** 50K causal LM + spread floor on the full registry weighted mix. 2. **SFT (nice):** 10–100K steps, lr 5e-6, prompt-masked labels. 3. **Instruct (optional):** skip unless examiner asks. 4. **Alignment (minimal):** DPO only, on 2-layer outputs.
- **Pydantic path:** `ModelConfig → ModelArchitectureConfig(ekdant) → EkdantConfig`; validators cover `save_steps/total_limit` and `lr/warmup` sanity.
- **Checkpointing (thesis spec):** `model.safetensors` + `config.json` + tokenizer + `training_args.bin` under `models/ekdant/<stage>/`; resume scans `checkpoint-*`; W&B/MLflow/TensorBoard trackers. *(Not in this vault — design only.)*
- **Scale-up (optional):** same file with hidden 10240 / d_state 4096 / layers 6 / vocab 128K / seq 4096 / FSDP full-shard + CPU offload / 1M steps. No code change.

**Forward + loss (the math the thesis defends):**

```
h, compressed = ssm_storage(input_ids)      # L1 — thesis design only, not in code
hT = transformer_stack(h)                   # T: dense windowed blocks
y, sandbox_aux = sandbox(hT)                # L2 direct handoff
logits = output_head(y)                     # tied single projection
loss = lm_weight*CE(shift(logits), shift(labels)) + sandbox_weight * max(0, rho_min - rho)**2
```

The shipped code (`09-Code/src/ekdant/model.py`) runs **embed → T → L2 → head** (the `ssm_storage` line is L1 — thesis design only) and its loss is CE + `lambda_sbx · relu(rho_min − spread_ratio)²` from `aux`.

## 2. Shipped runner (`09-Code/src/ekdant/train.py`)

**A/B, architecture-matched.** Two variants — `--variant sandbox|baseline` — share the identical backbone; the baseline is the same config with `use_sandbox: false` (50.09M vs 43.53M; delta = L2's 6.56M). Run one variant at a time.

**Two stages, one invocation.** `STAGES = (("medmcqa","train"), ("medqa","train"))` — MedMCQA (182,496 train after the loader drops 326 unlabelled `cop=-1` rows from the 182,822 raw; 3,802 opt/epoch at effective batch 48) then MedQA-USMLE (10,178 train, 212 opt/epoch); then `TESTS` evaluates the benchmarks — MedMCQA **validation** 4,183 (the public test split ships `cop=-1`, unlabelled, T18) and MedQA test 1,273 — each **split 50/50 in half by `split_half()`**: the **seen** half is the per-epoch validation (monitored, selects `best.pt`), the **unseen** half is scored exactly once at the end, and both are merged into a `*_full` score. The split is `--seed`-derived → identical halves for both A/B arms. Data loads through `data.py::load_examples` (offline-first: cached loads skip the hub entirely); each example is formatted `question\nA. …\nAnswer: <letter>` and the answer-token index is cached (`answer_pos`). The collator pads to **batch max** (prompts ≈ 40–80 tokens; `max_seq_len` is a truncation cap) — measured 6.3 opt steps/s on a T4.

**Tokenizer trained once, shared vocabulary.** On first run, a BPE tokenizer (NFD+lowercase, whitespace pre-tokenizer, `vocab_size` from config, special `<pad>/<bos>/<eos>/<unk>`) is trained on ≤20K MedMCQA train samples (prompt+answer), saved to `<out>/tokenizer.json`, and **reloaded if present** — both variants therefore see the same vocabulary (required for a fair A/B). `bind_letter_ids` resolves the A/B/C/D token ids for evaluation.

**CLI flags:**

| flag | default | notes |
|---|---|---|
| `--variant` | required | `sandbox` \| `baseline` |
| `--config` | `configs/<variant>_50m.json` | any `EkdantConfig` JSON |
| `--out` | required | output dir (`runs/<variant>`) |
| `--batch-size` | 24 | per GPU |
| `--grad-accum` | 2 | effective batch = 24 × 2 × n_gpu |
| `--lr` | 3e-4 | cosine, 200 warmup optimizer steps (fixed) |
| `--warmup` | 200 | optimizer steps (fixed, §4) |
| `--epochs` | 3 | per stage |
| `--limit` | none | cap rows per dataset (smoke) |
| `--no-compile` | off | skip `torch.compile` |
| `--log-every` | 100 | progress line every N optimizer steps (`0` = epoch summaries only) |
| `--eval-every` | 1 | **seen-half** eval every N epochs → history + incremental `results.json` + `best.pt` on `medmcqa_seen` (`0` = final eval only) |
| `--seed` | 17 | also seeds the row sampler |
| `--resume` | off | continue `<out>/last.pt`; skips finished (stage, epoch) pairs |
| `--eval-only` | off | no training; re-run the final protocol (seen/unseen/full, `best.pt` if present) on an existing run dir |

**Optimization:** AdamW (wd 0.1, betas 0.9/0.95), BF16 autocast on CUDA, `GradScaler(enabled=False)` (bf16 needs no loss scaling), grad clip 1.0 at the accumulation boundary, `torch.compile` unless `--no-compile`. **DDP:** `torchrun --nproc_per_node=4 -m ekdant.train …` — `setup_dist()` reads `RANK`, NCCL, `DistributedSampler` per stage (set_epoch per epoch), `find_unused_parameters=bool(use_sandbox)`, rank-0-only logging/saving.

**Live telemetry (`--log-every`, default 100):** every N optimizer steps the runner prints
`[stage] ep0 step 1200/3802 (32%) loss=.. lr=.. gnorm=.. tok/s=.. eta=..` — step/total and % of the *stage* budget (the "how far gone" number), an EMA of CE, the current schedule LR, the **pre-clip global grad norm** (clip threshold 1.0; sustained ≫1 = clipping every step, collapse toward 0 = stalled), throughput, and stage ETA (epoch-0 ETAs are inflated by `torch.compile` warmup and converge within the first few thousand steps — the window resets each epoch). Each epoch closes with a summary (`done loss=… gnorm=… lr_end=… tok/s=… time=…`) whose loss is the true sample-mean CE (comparable to eval loss). Full glossary: [run.md](../run.md) §3 "Watching training progress".

**Checkpointing + outputs:** `<out>/config.json` at start; `<out>/last.pt` (`{model, cfg, step, completed, opt_step, history, best_medmcqa}`) after **every epoch** (single checkpoint, no rotation); with `--eval-every 1` each epoch also scores the **seen** benchmark halves (rank 0 on the unwrapped module — DDP-safe), appends `history[..].test = {medmcqa_seen, medqa_seen}` (accuracy/loss/n), rewrites `results.json` incrementally (curve survives a crash), and saves `<out>/best.pt` on a new `medmcqa_seen` best. **Final protocol:** load `best.pt` (selected on `medmcqa_seen`), score seen + unseen + merged full halves of both benchmarks **once**, record `results["selected"] = {checkpoint, selected_on, best_medmcqa_seen}` — `best.pt` is the checkpoint whose `*_unseen` numbers to quote; `last.pt` remains the resume artifact.

**Resume + eval-only (both verified end-to-end).** `--resume` loads `last.pt` weights/step/history, skips already-completed (stage, epoch) pairs (tagged `completed = stage_index × epochs + epoch + 1`), and reuses the `--out` tokenizer + exact `config.json` (which takes precedence over `configs/`); optimizer moments restart cold. `--eval-only` skips training entirely and re-runs the final protocol (seen/unseen/full for both benchmarks, from `best.pt` if present else `last.pt`). Keep `--variant`/`--out`/`--epochs` identical to the original run. Details: [run.md](../run.md) §4.

**Evaluation = letter accuracy (fixed — was T17).** `evaluate()` gathers the logits row at `answer_pos - 1` (the position that *predicts* the answer, since `logits[t]` predicts token `t+1`), restricts to the 4 letter-token ids (`LETTER_IDS`), and argmaxes vs `answer_target` — which is the **letter index 0–3**, not the answer's vocab id. One forward per example instead of four option-continuation scorings (plus mean CE loss). Both halves are load-bearing: reading `answer_pos` (the off-by-one) or comparing against the vocab id each silently collapse accuracy to ~0% while CE stays correct, so a trained model scores zero. Regression-tested by `test_evaluate_scores_answer_position_and_index`.

## 3. Losses (shipped vs thesis — one flag difference)

Shipped: `loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100) + lambda_sbx · relu(rho_min − rho)²·mean`, where `rho = spread_ratio = trajectory_variance / mean(hT²)` comes from `Sandbox._aux`, `rho_min = 0.05`.

> [!warning] Weight mismatch: shipped `lambda_sbx` = **1.0** (`configs/*.json`), thesis spec `sandbox_weight` = **0.1**. The floor is a hinge (0 while healthy), so the practical effect is small — but any published run must state which λ it used. The term is a **spread floor**, never an agreement penalty `MSE(traj, consensus)` (that loss is minimized by the exact collapse it forbids — [29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) §3.2).

## 4. Known issues (open — see [10_Known_Issues](../07-Management/10_Known_Issues.md))

1. **LR schedule bugs — FIXED (per-stage cosine).** (a) `lr_at(step, total, base, warmup)` used to be called **once per epoch** with `step` incremented by 1 per epoch, while `total_steps` counted **batches** — so with defaults `step` only ever reached 5, `step < warmup` always held, and lr stayed ≤ 1.5% of 3e-4. (b) The first fix used a single global `opt_step` that never reset between stages, so stage 2 (MedQA) inherited stage 1's spent budget and trained at **lr = 0** for all epochs (losses flat at ~20.82 — inflated ×batch-size, true ≈3.5 at batch 6). **Now:** LR is set per **optimizer step** inside the batch loop, the counter **resets at each stage boundary** so each stage runs its own warmup+cosine, `total_opt = ceil(len(loader)/grad_accum) × epochs`, the cosine term is clamped to `p ∈ [0,1]`, and `opt_step` is saved in `last.pt` for resume. Regression tests: `tests/test_schedule.py`.
2. **Sandbox diagnostics computed but never logged.** `ModelOutput.aux` carries `anchor_energy`, `trajectory_variance`, `spread_ratio`, `per_trajectory_retention`, `anchor_distance`, but `train_stage` ignores `out.aux` — history now carries `loss`/`lr`/`gnorm` per epoch, but still no ρ trace for any run (the collapse monitor the thesis requires). Fix: accumulate `aux` scalars per epoch and append to history.
3. **Train-loss scale bug — FIXED.** The epoch mean divided Σ(loss × batch_size) by the batch *count*, so every pre-fix train loss was inflated ×`--batch-size` (e.g. "22.28" at batch 6 ≈ true 3.7 — consistent with eval CE ≈ 3.2–3.6). Fixed to a true sample mean; eval losses were always correct; same-batch comparisons are unaffected.
4. **L2 size open item:** measured 6.56M vs thesis 2.8M (2.3×; full H×H q/k/v cross-projections + untied rounds) — decide slim-vs-restate before any results table ([14](14_Performance_Analysis.md) §1, [09-Code/README](../09-Code/README.md)).
5. Not bugs but design gaps: diagnostics (§4.2); harness Phase 8 (`09_Validation_System.md`) is SKIPPED — the smoke run below is the substitute. *(Resume previously absent — now shipped as `--resume`/`--eval-only`.)*
6. **Eval hang before first result — FIXED (offline-first data loading).** Per-epoch evals called `load_dataset`, which performs a hub etag HEAD on every load; with bad connectivity that request stalls indefinitely (dead route, IPv6) and training appears frozen with no error line. `data.py` now defaults `HF_HUB_OFFLINE=1` (cache-only reads) and flips online only for a first-run download (`tests/test_data.py` locks the loader contract; the stall itself was reproduced live, not unit-testable).

## 5. Smoke test (Phase 8 substitute)

```bash
cd 09-Code && pip install -e .[dev]
pytest -q    # 78 tests, CPU, ~10s
python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
```

Verified on CPU (limit 64, 1 epoch): **both arms** run train → resume → eval-only end-to-end; all four output files written. Full procedure incl. benchmark comparison + reporting rules (C8): [run.md](../run.md). *(The thesis 8-phase harness script is absent — Phase 8 SKIPPED; the 50-step XS/O(1)-curve checks are design targets.)*

## Connections
- Builds: [26a](../04-Model-DeepDive/26a_Internal_SSM_Build.md) · [26b](../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) · [26c](../04-Model-DeepDive/26c_Internal_Sandbox_Build.md) · Data: [05](../05-Data/05_Data_Pipeline.md) · Validation: [09](09_Validation_System.md) · Debug: [13](13_Debugging_Guide.md) · Perf: [14](14_Performance_Analysis.md)
