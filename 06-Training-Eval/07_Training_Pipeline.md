# Training Pipeline (Ekdant) — Ch. 5

> Map: [README](../README.md) · Runbook: [run.md](../run.md) · Spec: [02](../02-Architecture/02_System_Architecture.md) · Builds: [26a](../04-Model-DeepDive/26a_Internal_SSM_Build.md) · [26b](../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) · [26c](../04-Model-DeepDive/26c_Internal_Sandbox_Build.md) · Config: [12](../02-Architecture/12_Configuration_Guide.md)

`09-Code/src/ekdant/train.py` orchestrates Ekdant training (`python -m ekdant.train --variant sandbox|baseline`: **phase 1 MedMCQA train+test → phase 2 MedQA 60% fine-tune → 40% benchmark**, one invocation, never jointly trained): one architecture, two losses (LM CE + spread floor), A/B pretrain only. The SFT/instruct/DPO stages in §1 are thesis design, not shipped.

## 0. v2 runner (`train_gen.py`, `--task gen`) — current

1. **Pre-flight:** pick the GPU (`--gpu auto` = most free memory) and refuse below `--min-free-gb 20` — the pod's A100s are shared (the v1 sandbox OOM happened on a full GPU 0). Fresh-run guard as in v1.
2. **Zero-shot:** perplexity of the untrained arm on PubMedQA held-out, MedQuAD validation and MedQuAD benchmark (`results.zero_shot`).
3. **Phase 1 — PubMedQA** (v2.1: 2 epochs, `--epochs 2`; v2.0: 1): AdamW with two LR groups (backbone 5e-5, L2 matrices 1e-4 (was 1e-3: it kept inflating L2's weights, T34), the gate alone 1e-3 (`--gate-lr`, T35); no weight decay on the gate or 1-D params), cosine per phase, 200 warm-up steps, clip 1.0 applied separately to the backbone and the add-on (`clip_grads` — the add-on's gradient must not shrink the backbone's update relative to the baseline), bf16 on A100, effective batch 32 (8 × 4). Validation on the 2,000 held-out rows every 25 % of the epoch; `last.pt` at each point.
4. **Phase 2 — MedQuAD fine-tune half** (3 epochs): validation slice each epoch → `best.pt` on lowest NLL.
5. **Final:** better of `best.pt` / `last.pt` on the validation slice (ties keep `best.pt`), then the benchmark half once: perplexity, per-type perplexity, ROUGE-L / token F1 on 1,000 greedy generations (≤ 256 new tokens), and PubMedQA held-out retention (+ 500 generations, ≤ 128 tokens). `cost`: params, tokens, wall-clock, peak memory.
6. **Divergence guard (T32):** an optimiser step whose loss or gradient is non-finite is skipped (whole accumulation group dropped) and counted in `cost.skipped_steps`; `--max-bad-steps` (20) consecutive skips stop the run with a clear message.
7. **Resume:** `last.pt` stores the optimiser state, the stage/epoch and the batch position; each epoch's order comes from a generator seeded by (seed, stage, epoch), so `--resume` continues exactly (`test_resume_mid_epoch_matches_uninterrupted`); a resume with changed flags or variant is refused, zero-shot is never measured on loaded weights, and wall-clock accumulates across resumes.

Sections 1–5 below describe the v1 runner.

## 1. Design history (superseded)

The original thesis plan was a general pre-training run: three sizes (XS ≈12M / S ≈21M / M ≈46M) with an L1 SSM front end, 50K steps on a 54-dataset code/web/docs mixture, then SFT and DPO-lite, driven by `config_ekdant_xs/s/m.yaml`. None of that exists in this repo. The project narrowed to a single **50M-class T + L2 A/B on medical multiple-choice QA**; L1 is a specified extension, not code. The shipped model and loss:

```
x  = embed(input_ids)
hT = transformer_stack(x)            # T: 12 dense windowed blocks (H=512, W=512)
y, aux = sandbox(hT)                 # L2 (sandbox arm only)
logits = output_head(ln_f(y))        # tied single projection
loss = CE(shift(logits), shift(labels))
     + answer_weight * CE(logits[answer_pos-1], answer_letter)     # --answer-weight, 0 = off
     + lambda_sbx * relu(rho_min - rho)**2                         # sandbox arm only, rho = spread_ratio
```

Runnable configs are `09-Code/configs/{sandbox,baseline}_50m.json`. The L1 line `h, compressed = ssm_storage(input_ids)` of the full design is not in code.

## 2. Shipped runner (`09-Code/src/ekdant/train.py`)

**A/B, architecture-matched.** Two variants — `--variant sandbox|baseline` — share the identical backbone; the baseline is the same config with `use_sandbox: false` (50.09M vs 43.53M; delta = L2's 6.56M). Run one variant at a time.

**Two phases, one invocation, strictly sequential.** `STAGES = (("medmcqa","train"), ("medqa_ft","train"))` —

1. **Phase 1 — MedMCQA:** trains on the **full** MedMCQA train split (182,496 labelled rows, 3,802 opt/epoch at effective batch 48; the 326 unlabelled `cop=-1` rows are dropped, T18 rule) for `--epochs` epochs, then records the **MedMCQA validation** result once (`results["phase1"]["medmcqa_val"]` — the public test split ships unlabelled).
2. **Phase 2 — MedQA fine-tune:** fine-tunes on the **60% partition** of MedQA train (6,106 of 10,178 rows; 128 opt/epoch) for `--epochs` epochs. The partition is `split_frac(train, FT_FRAC=0.6, --seed)` — identical rows for both A/B arms (fairness rule); the complementary **40% (4,072 rows) is never trained on**.
3. **Final protocol:** better of `best.pt` vs `last.pt` on the untouched **MedMCQA validation** split (4,183 rows, 4,170 usable — the public test split ships `cop=-1`, unlabelled, T18; selection set, evaluated every phase-2 epoch via `--eval-every`) → benchmark the 40% MedQA partition **exactly once** (`medqa_bench`) + official MedQA test (`medqa_test`, also untouched) + selection score, all into `results.json` (`test{medmcqa_val, medqa_bench, medqa_test}` + `phase1` + `selected`).

Data loads through `data.py::load_examples` (offline-first: cached loads skip the hub entirely); each example is formatted `question\nA. …\nAnswer: <letter>` and the answer-token index is cached (`answer_pos`). The collator pads to **batch max** — measured with the real tokenizer on all four splits: MedMCQA p50=34 / p99=118, MedQA p50=173 / p99=364, max 857, **0 of 198k rows exceed the 2048 cap** (nothing truncated or dropped; full prompts verified by decode round-trip) — measured 6.3 opt steps/s on a T4. Note the model never receives `attention_mask`: padding is **right-side** and attention is **causal**, so real tokens cannot attend to pads (pads carry `-100` labels = no loss).

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
| `--epochs` | 3 | phase 1 (MedMCQA) epochs |
| `--ft-lr` | `--lr` | **phase-2 (MedQA fine-tune) cosine base LR** — anti-overfit knob (e.g. `5e-5`); phase 1 never uses it |
| `--ft-epochs` | `--epochs` | **phase-2 epochs** — cap the fine-tune (e.g. `10`) instead of replaying `--epochs` over 6,106 rows |
| `--answer-weight` | 0.0 | adds `weight × CE(logits[answer_pos−1], answer_id)` to the LM loss (letter = 1 token in ~34/173 → plain CE dilutes the decision ~97–99%; `10` recommended; `0` = legacy) |
| `--limit` | none | cap rows per dataset (smoke) |
| `--no-compile` | off | skip `torch.compile` |
| `--log-every` | 100 | progress line every N optimizer steps (`0` = epoch summaries only) |
| `--eval-every` | 1 | untouched MedMCQA validation every N epochs — phase 1: monitoring only (history, T25); phase 2: also `best.pt` selection (`0` = final only) |
| `--seed` | 17 | seeds the row sampler **and the 60/40 partition** (identical for both arms) |
| `--resume` | off | continue `<out>/last.pt`; skips finished (stage, epoch) pairs |
| `--eval-only` | off | no training; re-run the final protocol (selection + 40% benchmark) |

**Optimization:** AdamW (wd 0.1, betas 0.9/0.95), BF16 autocast on CUDA, `GradScaler(enabled=False)` (bf16 needs no loss scaling), grad clip 1.0 at the accumulation boundary, `torch.compile` unless `--no-compile`. Gradient accumulation flushes the **trailing partial group** at each epoch end (the schedule budgets `ceil(steps/accum)` steps — without the flush the tail grads leaked into the next epoch's first step; T23), and the train DataLoader shuffles with a dedicated **`--seed`-locked generator** so both A/B arms see identical row order (global-RNG shuffle diverged between arms — model init consumes different RNG amounts per arm; after `--resume` the generator restarts: order noise only, never the partition). **DDP:** `torchrun --nproc_per_node=4 -m ekdant.train …` — `setup_dist()` reads `RANK`, NCCL, `DistributedSampler` per stage (set_epoch per epoch), `find_unused_parameters=bool(use_sandbox)`, rank-0-only logging/saving.

**Live telemetry (`--log-every`, default 100):** every N optimizer steps the runner prints
`[stage] ep0 step 1200/3802 (32%) loss=.. lr=.. gnorm=.. tok/s=.. eta=..` — step/total and % of the *stage* budget (the "how far gone" number), an EMA of CE, the current schedule LR, the **pre-clip global grad norm** (clip threshold 1.0; sustained ≫1 = clipping every step, collapse toward 0 = stalled), throughput, and stage ETA (epoch-0 ETAs are inflated by `torch.compile` warmup and converge within the first few thousand steps — the window resets each epoch). Each epoch closes with a summary (`done loss=… gnorm=… lr_end=… tok/s=… time=…`) whose loss is the true sample-mean CE (comparable to eval loss). Full glossary: [run.md](../run.md) §3 "Watching training progress".

**Checkpointing + outputs:** `<out>/config.json` at start; `<out>/last.pt` (`{model, cfg, step, completed, opt_step, history, best_medmcqa}`) after **every epoch** (single checkpoint, no rotation); during phase 2 with `--eval-every 1` each epoch also scores the **untouched MedMCQA validation** split (rank 0 on the unwrapped module — DDP-safe), appends `history[..].test = {medmcqa_val}` (accuracy/loss/n), rewrites `results.json` incrementally (curve survives a crash), and saves `<out>/best.pt` on a new `medmcqa_val` best. The phase-1 MedMCQA-validation result is written once at the phase boundary (`results["phase1"]`). **Final protocol:** evaluate `best.pt` and `last.pt` on `medmcqa_val`, load the better (ties keep `best.pt` — "change the best with last if it is good"), record `results["selected"] = {checkpoint, selected_on: "medmcqa_val", best_val_accuracy, last_val_accuracy}`, then score the **40% MedQA partition** (`medqa_bench`, once, never trained on) + official MedQA test (`medqa_test`) — **that is the number to quote**; `last.pt` remains the resume artifact.

**Resume + eval-only (both verified end-to-end).** `--resume` loads `last.pt` weights/step/history, skips already-completed (stage, epoch) pairs (tagged `completed = (epoch counts of earlier stages summed) + epoch + 1` — the legacy `stage_index × epochs` when both phases share `--epochs`), reuses the `--out` tokenizer + exact `config.json` (which takes precedence over `configs/`), and preserves the recorded `phase1` result in `results.json`; optimizer moments restart cold. `--eval-only` skips training entirely and re-runs the final protocol (better-of selection on `medmcqa_val`, then the 40% benchmark + official MedQA test). Keep `--variant`/`--out`/`--epochs` (and `--ft-epochs`/`--ft-lr`, if used) identical to the original run. Details: [run.md](../run.md) §4.

**Evaluation = letter accuracy (fixed — was T17).** `evaluate()` gathers the logits row at `answer_pos - 1` (the position that *predicts* the answer, since `logits[t]` predicts token `t+1`), restricts to the 4 letter-token ids (`LETTER_IDS`), and argmaxes vs `answer_target` — which is the **letter index 0–3**, not the answer's vocab id. One forward per example instead of four option-continuation scorings (plus mean CE loss). It also returns `pred_counts` / `gold_counts` (A–D histograms) and `majority_accuracy` — the floor every result is read against, not 25 % (T27). Both halves are load-bearing: reading `answer_pos` (the off-by-one) or comparing against the vocab id each silently collapse accuracy to ~0% while CE stays correct, so a trained model scores zero. Regression-tested by `test_evaluate_scores_answer_position_and_index`.

## 3. Losses (shipped vs thesis — one flag difference)

Shipped: `loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100) + lambda_sbx · relu(rho_min − rho)²·mean`, where `rho = spread_ratio = trajectory_variance / mean(hT²)` comes from `Sandbox._aux`, `rho_min = 0.05`.

Optional (`--answer-weight w`, default 0 = legacy): `+ w · CE(logits[answer_pos−1], answer_vocab_id)` — the answer letter is only **1 token in ~34 (MedMCQA) / ~173 (MedQA)**, so pure LM CE spends ~97–99% of its gradient on predicting question text (40-epoch pilot: train CE → ~0.05 while accuracy sat at ~35%). The term supervises the exact position/metric `evaluate()` scores, while the LM loss (needed by the sandbox's generation) is kept at full weight. Recommended campaign value: `10`.

> [!warning] Weight mismatch: shipped `lambda_sbx` = **1.0** (`configs/*.json`), thesis spec `sandbox_weight` = **0.1**. The floor is a hinge (0 while healthy), so the practical effect is small — but any published run must state which λ it used. The term is a **spread floor**, never an agreement penalty `MSE(traj, consensus)` (that loss is minimized by the exact collapse it forbids — [29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) §3.2).

## 4. Known issues (open — see [10_Known_Issues](../07-Management/10_Known_Issues.md))

1. **LR schedule bugs — FIXED (per-stage cosine).** (a) `lr_at(step, total, base, warmup)` used to be called **once per epoch** with `step` incremented by 1 per epoch, while `total_steps` counted **batches** — so with defaults `step` only ever reached 5, `step < warmup` always held, and lr stayed ≤ 1.5% of 3e-4. (b) The first fix used a single global `opt_step` that never reset between stages, so stage 2 (MedQA) inherited stage 1's spent budget and trained at **lr = 0** for all epochs (losses flat at ~20.82 — inflated ×batch-size, true ≈3.5 at batch 6). **Now:** LR is set per **optimizer step** inside the batch loop, the counter **resets at each stage boundary** so each stage runs its own warmup+cosine, `total_opt = ceil(len(loader)/grad_accum) × epochs`, the cosine term is clamped to `p ∈ [0,1]`, and `opt_step` is saved in `last.pt` for resume. Regression tests: `tests/test_schedule.py`.
2. **Sandbox diagnostics — FIXED (T24).** `anchor_energy`, `trajectory_variance` and `spread_ratio` are averaged per epoch into `history[*].sandbox` (also printed as `[stage] epN sandbox …`) and into every `evaluate()` result (`results.json` → `test.*.sandbox`). They are computed over real tokens only (the attention mask now reaches L2). Sandbox arm only; the baseline has no `aux`.
3. **Train-loss scale bug — FIXED.** The epoch mean divided Σ(loss × batch_size) by the batch *count*, so every pre-fix train loss was inflated ×`--batch-size` (e.g. "22.28" at batch 6 ≈ true 3.7 — consistent with eval CE ≈ 3.2–3.6). Fixed to a true sample mean; eval losses were always correct; same-batch comparisons are unaffected.
4. **L2 size open item:** measured 6.56M vs thesis 2.8M (2.3×; full H×H q/k/v cross-projections + untied rounds) — decide slim-vs-restate before any results table ([14](14_Performance_Analysis.md) §1, [09-Code/README](../09-Code/README.md)).
5. Not bugs but design gaps: diagnostics (§4.2); the smoke run below plus the unit suite is the validation harness (`09_Validation_System.md`). *(Resume previously absent — now shipped as `--resume`/`--eval-only`.)*
6. **Phase-2 memorization risk — MITIGATED (per-phase knobs).** Phase 2 used to inherit `--lr`/`--epochs`, so the campaign recipe spent 12,800 steps @ 3e-4 on only 6,106 MedQA rows — the 40-epoch pilot (`runs/baseline_40ep`) memorized it (train CE → ~0.05, held-out MedQA stuck at chance ≈25%, MedMCQA validation decaying = forgetting). **`--ft-lr` / `--ft-epochs`** (defaults: `--lr`/`--epochs`, so unchanged unless passed) fine-tune only phase 2; recommended `--ft-lr 5e-5 --ft-epochs 10` (phase 2 = 1,280 steps) in [run.md](../run.md) §3. Regression: `tests/test_schedule.py` (`stage_hparams`, `stage_offsets`).
7. **Supervision dilution — MITIGATED (`--answer-weight`).** Plain LM CE trains one answer token out of ~34 (MedMCQA) / ~173 (MedQA) — ~97–99% of the gradient goes to predicting question text, which is why the 40-epoch pilot's train CE collapsed to ~0.05 while letter accuracy plateaued at ~35%. The full prompt itself is intact (verified: 0 of 198k rows exceed `max_seq_len` 2048; decode round-trips complete). **`--answer-weight`** (default 0 = legacy) adds a re-weighted CE at the same position/metric as eval; recommended `10` in [run.md](../run.md) §3. Regression: `tests/test_model.py` (`test_answer_weight_*`), `tests/test_data.py` (`test_collate_carries_answer_vocab_ids`).
8. **Training-loop correctness (accum tail / eval loss / A/B shuffle) — FIXED (T23).** (a) The trailing partial grad-accumulation group was never stepped and leaked its grads into the next epoch's first step (small `--limit` runs fired **zero** optimizer steps per epoch); now flushed with a proper unscale/clip/step at each epoch end. (b) Eval loss was a per-batch mean — now a true sample mean (batch-size weighted). (c) Shuffle order diverged between A/B arms (global RNG + different model-init consumption) — now a dedicated `--seed`-locked generator. Regression: `tests/test_schedule.py`, `tests/test_data.py` (T23 rows).
9. **Eval hang before first result — FIXED (offline-first data loading).** Per-epoch evals called `load_dataset`, which performs a hub etag HEAD on every load; with bad connectivity that request stalls indefinitely (dead route, IPv6) and training appears frozen with no error line. `data.py` now defaults `HF_HUB_OFFLINE=1` (cache-only reads) and flips online only for a first-run download (`tests/test_data.py` locks the loader contract; the stall itself was reproduced live, not unit-testable).

## 5. Smoke test

```bash
cd 09-Code && pip install -e .[dev]
pytest -q    # 153 tests, CPU, ~10s
python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
```

Verified on CPU (limit 64, 1 epoch): **both arms** run train → resume → eval-only end-to-end; all four output files written. Full procedure incl. benchmark comparison + reporting rules (C8): [run.md](../run.md). *(The thesis 8-phase harness script is absent — Phase 8 SKIPPED; the 50-step XS/O(1)-curve checks are design targets.)*

## Connections
- Builds: [26a](../04-Model-DeepDive/26a_Internal_SSM_Build.md) · [26b](../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) · [26c](../04-Model-DeepDive/26c_Internal_Sandbox_Build.md) · Data: [05](../05-Data/05_Data_Pipeline.md) · Validation: [09](09_Validation_System.md) · Debug: [13](13_Debugging_Guide.md) · Perf: [14](14_Performance_Analysis.md)
