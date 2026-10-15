# run.md — Ekdant operations runbook

How to **start training, resume, test, and benchmark** both models. Every
command here was executed and verified (CPU smoke: 64 rows, 1 epoch, both
arms; full unit suite: 67 passed). All commands run from `09-Code/` unless
stated otherwise.

## 0. What you are running

The code (`09-Code/src/ekdant/`) is **L2-only**: `embed → T (windowed dense
Transformer) → L2 (sandbox) → head`. The thesis L1 SSM is design-only and
not in this package. The experiment is an **architecture-matched A/B** —
two models identical in every knob, differing only in `use_sandbox`:

| variant | total | embed | T (12 blocks) | L2 | config |
|---|---|---|---|---|---|
| `sandbox` | **50.09M** | 16.777M | 26.75M | 6.56M | `configs/sandbox_50m.json` |
| `baseline` | **43.53M** | 16.777M | 26.75M | 0 | `configs/baseline_50m.json` |

Run **one variant at a time** — each uses the full node.

## 1. Prerequisites

| need | notes |
|---|---|
| Python 3.10+ | verified on 3.12 |
| PyTorch 2.x | verified on 2.9; CUDA build required for real runs (CPU only works for smoke) |
| `datasets` (HF) | auto-downloads both datasets on first run |
| internet (first run) | MedMCQA ≈ 183k rows + MedQA ≈ 10k rows land in the HF cache (few GB) |
| disk | + ~0.5 GB per run dir (tokenizer + checkpoint) |
| GPU (full run) | 1× A100 80GB fits batch 24 / seq 2048 / bf16; 16GB (T4): use the 20K recipe's `--batch-size 6 --grad-accum 8` (§3) |

No `mamba-ssm`, `flash-attn`, `triton`, `causal-conv1d` — pure PyTorch.

## 2. Install

```bash
cd 09-Code
pip install -e .          # package: python -m ekdant.train works
pip install -e .[dev]     # + pytest for the test suite
python -c "import ekdant; print('ok')"   # sanity check — must print ok
```

> [!important]
> `-e .` needs the **trailing dot**. `pip install -e` fails with
> `-e option requires 1 argument`, so nothing is installed and
> `python -m ekdant.train` raises
> `ModuleNotFoundError: No module named 'ekdant'`. If you see that error:
> you skipped the dot — re-run `pip install -e .` (from inside `09-Code/`).
>
> `pyproject.toml` pins `datasets>=2.19,<5` + `pyarrow>=12,<21` so a bare
> `pip install -e .` resolves a combo that works on both numpy 1.x
> (conda/Jupyter images) and numpy 2.x. An **older** clone without these
> pins will pull datasets 5.x / pyarrow 26 and crash with
> `pyarrow requires NumPy 2.0` — `git pull`, then re-run
> `pip install -e .` (it will downgrade them), or repair manually:
> `pip install "datasets==3.6.0" "pyarrow<21"`.

### Linux / Kaggle / WSL notes

```bash
git clone git@github.com:N0t-Harshil/Ekdant-LLM.git   # or the https URL
cd Ekdant-LLM/09-Code
pip install -e .        # uses the conda/venv python on PATH — that is the one that must print ok
python -c "import ekdant; print('ok')"
```

- Keep `pip` and `python` from the **same environment** (on conda/Kaggle:
  `/opt/conda/bin/pip` + `/opt/conda/bin/python`; a system pip installs into
  a python you are not running).
- PyTorch is preinstalled on most Kaggle images; `pip install -e .` reuses
  it (the `torch>=2.2` requirement is already satisfied).
- **`ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`** —
  conda/Jupyter pins numpy 1.x, but unpinned installs pull pyarrow ≥21
  (numpy-2 only) via datasets 5.x. **Fixed at the source**: `pyproject.toml`
  now caps `datasets<5` + `pyarrow<21`, so `git pull && pip install -e .`
  self-repairs. Manual repair for older clones:
  `pip install "datasets==3.6.0" "pyarrow<21"` (do **not** upgrade numpy on
  a conda image — it breaks prebuilt pandas/scipy).
- **Kaggle T4/P100 = 16 GB VRAM** — the default batch 24 assumes ~30–40 GB.
  Run `--batch-size 6 --grad-accum 8` (effective batch 48, same as the
  24×2 default), and `--no-compile` if compile is slow or crashes.
- No-install fallback (skip pip entirely, from `09-Code/`):

```bash
PYTHONPATH=src python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
```

## 3. Start training (both models)

### Full A/B (single GPU)

```bash
python -m ekdant.train --variant sandbox  --out runs/sandbox
python -m ekdant.train --variant baseline --out runs/baseline
```

### Recommended long run — the 20K-step A/B (T4, seq 2048)

The validated recipe for a decisive comparison. **~20,000 optimizer steps
total** (5 epochs × 3,809 MedMCQA steps + 5 × 213 MedQA steps at
`--batch-size 6 --grad-accum 8`), seq 2048 (shipped configs), correct
per-stage cosine schedule, full telemetry (§ "Watching training progress"):

```bash
cd 09-Code
nohup python -m ekdant.train --variant baseline --out runs/baseline_20k \
  --batch-size 6 --grad-accum 8 --epochs 5 > baseline_20k.log 2>&1 &
tail -f baseline_20k.log            # watch live; Ctrl-c stops watching only

# after it writes results.json, same command for the sandbox arm:
nohup python -m ekdant.train --variant sandbox --out runs/sandbox_20k \
  --batch-size 6 --grad-accum 8 --epochs 5 > sandbox_20k.log 2>&1 &
```

Budget on a T4: baseline ≈ 13 h, sandbox ≈ 18 h (~31 h sequential).
Epoch checkpoints + `nohup` make it resume-safe (add `--resume` after a
pod restart). Scale guide (everything stays linear in epochs at this
batch):

| `--epochs` | optimizer steps | T4 wall-clock (both arms) | note |
|---|---|---|---|
| 3 | ~12K | ~19 h | the recorded pipeline campaign (`runs/*_full`) |
| **5** | **~20K** | **~31 h** | **recommended — the 20K range** |
| 14 | ~50K | ~76 h | thesis recipe (50K steps) |
| 50 | ~200K | ~12 days | beyond the recipe; overfitting risk, not recommended |

### Watching training progress

Every `--log-every` optimizer steps (default 100) the run prints a
progress line; `tail -f <out>.log` shows the run advancing live:

```
[medmcqa] ep0 step 1200/3809 (32%) loss=4.0217 lr=2.91e-4 gnorm=1.32 tok/s=11812 eta=1h23m45s
[medmcqa] ep0 done loss=3.7104 gnorm=1.41 lr_end=2.30e-4 tok/s=11860 time=2h29m10s
```

How to read it:

| field | meaning / healthy signal |
|---|---|
| `step 1200/3809 (32%)` | optimizer steps into the stage and % of the **stage** budget — this is your "how far gone" number |
| `loss` | EMA of CE per token — should fall every epoch; a flat line at init lr≈0 means a schedule bug (§8) |
| `lr` | current learning rate — rises through warmup (default 200 steps) then cosine-decays to ~0 at the stage end |
| `gnorm` | global grad norm **before** clipping at 1.0 — sustained ≫1 means the clip is active every step; a sudden collapse toward 0 means training has stalled; expect ~0.5–3 in steady state |
| `tok/s` | throughput — drops when batches are long (2048-token rows) or the GPU is contended |
| `eta` | projected time to finish the **stage** from the current rate |
| `ep0 done …` | epoch summary: true sample-mean CE (comparable to eval loss), mean gnorm, LR at epoch end, wall time |

Epoch history (loss / lr / **gnorm** per epoch) is stored in
`last.pt → history` and written into `results.json`, so every run keeps
its training curve, not just its final numbers.

### Full A/B (multi-GPU, e.g. 4×A100)

```bash
torchrun --nproc_per_node=4 -m ekdant.train --variant sandbox  --out runs/sandbox
torchrun --nproc_per_node=4 -m ekdant.train --variant baseline --out runs/baseline
```

### Smoke first (verified on CPU, ~2 min per arm)

```bash
python -m ekdant.train --variant sandbox  --out /tmp/sbx  --limit 64 --epochs 1 --batch-size 4 --no-compile
python -m ekdant.train --variant baseline --out /tmp/base --limit 64 --epochs 1 --batch-size 4 --no-compile
```

(Low or near-zero test accuracy on a smoke run is normal — 64 rows × 1 epoch
trains almost nothing, so the model sits near chance. The point of a smoke
run is the pipeline, files, and resume paths. What is *not* normal is a
locked-in 0.00% on a real run: see §8 — that was a metric bug (T17), fixed.)

### What one invocation does

1. Loads config (`--config` → else `configs/<variant>_50m.json`; baseline
   additionally forces `use_sandbox=False`).
2. Trains a **fresh BPE tokenizer** (vocab 32k) on 20k MedMCQA prompts and
   saves `<out>/tokenizer.json` — or **reuses** it if already present. Both
   arms see the same sample with the same seed, so the vocabularies match;
   for a guaranteed-identical vocab, copy `runs/sandbox/tokenizer.json`
   into `runs/baseline/` before launching baseline.
3. Stage 1: **MedMCQA train** (182,822 rows) — `--epochs` epochs.
4. Stage 2: **MedQA train** (10,178 rows) — `--epochs` epochs.
5. After **every epoch** saves `<out>/last.pt` (weights + step + completed
   counter + history + config).
6. After training: evaluates both held-out splits (MedMCQA
   **validation** 4,183 — the public MedMCQA test set ships `cop=-1`,
   unlabelled, so validation is the standard eval split (bug T18) — and
   MedQA test 1,273) by letter-accuracy and writes
   `<out>/results.json`.

### Flags

| flag | default | meaning |
|---|---|---|
| `--variant` | required | `sandbox` or `baseline` |
| `--out` | required | run dir (checkpoints, tokenizer, results) |
| `--config` | `configs/<variant>_50m.json` | custom `EkdantConfig` JSON |
| `--batch-size` | 24 | **per GPU** |
| `--grad-accum` | 2 | effective batch = batch × accum × n_gpu (=192 on 4 GPUs) |
| `--lr` | 3e-4 | cosine schedule, `--warmup` steps |
| `--warmup` | 200 | warmup **optimizer steps**, then cosine (LR fix, §8) |
| `--epochs` | 3 | epochs **per stage** (MedMCQA and MedQA each) |
| `--limit` | none | cap rows per dataset (smoke/debug only) |
| `--seed` | 17 | torch seed + tokenizer sample + data sampling |
| `--no-compile` | off | skip `torch.compile` (use if compile crashes or first-run overhead hurts) |
| `--log-every` | 100 | progress line every N optimizer steps (`0` = epoch summaries only) |
| `--resume` | off | continue an interrupted run (§4) |
| `--eval-only` | off | skip training; evaluate existing checkpoint (§6/§7) |

**Fairness rule:** keep every flag identical between the two arms (only
`--variant` and `--out` change). Any other difference invalidates the A/B.

## 4. Resuming an interrupted run

```bash
python -m ekdant.train --variant sandbox --out runs/sandbox --resume
# same flags as the original run, plus --resume
```

`--resume` (also implied prerequisites for `--eval-only`):

| restored from `<out>/last.pt` | value |
|---|---|
| model weights | loaded into the fresh model before DDP/compile |
| optimizer-step counter `step` | continues the global count |
| completed (stage, epoch) pairs | already-finished pairs are **skipped** |
| loss/LR history | appended, not restarted |

| restored automatically from `<out>/` | note |
|---|---|
| `tokenizer.json` | reused, never retrained — vocab cannot drift |
| `config.json` | **exact config the run started with** takes precedence over `configs/` |

Not restored (unavoidable with `torch.save` per epoch): AdamW moments
(optimizer restarts cold) — expect a small loss bump on the first resumed
batch.

Semantics: checkpoints are saved after every epoch, tagged
`completed = stage_index × epochs + epoch + 1`. Resume with the **same
`--variant`, `--out`, and `--epochs`** as the original run; changing
`--epochs` changes the pair arithmetic. Examples:

- interrupted mid-stage → resumes at the next unfinished epoch of that stage
- interrupted between stages → skips the finished stage, starts the next
- finished run + `--resume` → skips all training, just re-evaluates (§7)

## 5. Testing

### Unit suite (CPU, ~10 s)

```bash
cd 09-Code
pip install -e .[dev]
python -m pytest            # 75 tests
```

Covers: config/param accounting, T block (causal+window mask exactness,
no-leakage, GQA mapping, RoPE, shapes), L2 sandbox (K trajectories,
debate rounds, descent, collapse floor, aux keys), end-to-end forward/loss
decrease for both variants, and the data layer (MedMCQA `opa..opd`/`cop`
0-indexed schema, MedQA `options`+`answer_idx` schema, rejection rules).

### Held-out eval (no training)

```bash
python -m ekdant.train --variant sandbox  --out runs/sandbox  --eval-only --no-compile
python -m ekdant.train --variant baseline --out runs/baseline --eval-only --no-compile
```

Loads `last.pt` + `tokenizer.json` from `--out`, skips training, runs both
test sets, rewrites `results.json`. Use it to re-score a checkpoint on a
different machine, or to re-run the benchmark after a `--resume` finish
(every training invocation already evaluates at the end — `--eval-only` is
for re-evaluation without training).

Errors are explicit: `--eval-only` into an empty/nonexistent `--out`
fails with `tokenizer.json not found` / `last.pt not found`.

## 6. Benchmarking

**Metric:** held-out **letter-accuracy** — the logits at the answer position
(`answer_pos - 1`, the position that predicts the answer), argmax over the
A/B/C/D token logits vs the letter **index** (0–3), on MedMCQA validation
(4,183) and MedQA test (1,273). MedMCQA's public test split has `cop = -1`
for every row (no released labels), so it must never be scored — see bug
T18. Not free-form generation.

### Full procedure

```bash
# 1. train both arms (identical flags, separate dirs)
python -m ekdant.train --variant sandbox  --out runs/sandbox
python -m ekdant.train --variant baseline --out runs/baseline

# 2. compare
python - <<'PY'
import json
for v in ("baseline", "sandbox"):
    r = json.load(open(f"runs/{v}/results.json"))
    print(f"{v:9s} params={r['params']/1e6:.2f}M  "
          f"medmcqa={r['test']['medmcqa']['accuracy']*100:.2f}%  "
          f"medqa={r['test']['medqa']['accuracy']*100:.2f}%")
PY
```

Produces one line per arm; any accuracy gap is attributable to L2 alone
(architecture-matched).

### Interpreting your numbers (reference magnitudes)

Full 3-epoch A/B on a T4 (`--batch-size 6 --grad-accum 8`, full data, one
seed, both arms run on the per-stage LR fix) — the reference for "what a
complete run looks like":

| arm | params | MedMCQA val (4,170) | MedQA test (1,273) |
|---|---|---|---|
| baseline | 43.53M | 33.31% | 25.69% |
| sandbox | 50.09M | 32.01% | 26.08% |

**No significant difference between arms** at this scale (MedMCQA gap 1.3
pts, noise σ≈0.7%; MedQA pinned at chance 25% — stage 2 is only ~638
optimizer steps). A tie is a publishable outcome (C24 framing, see
[11_Future_Work](../07-Management/11_Future_Work.md)). Note the mild
sequential-fine-tuning erosion of MedMCQA after the MedQA stage
(33.81→33.31, 34.15→32.01). These are pipeline-scale numbers, not thesis
results: 1 seed, 3 epochs, ~50M params, architecture-matched. Historical
note: the first 1-epoch × 60k checks predate the T18 metric fix and the
per-stage LR fix — don't mix them into comparisons.

### Reporting rules (thesis cost rule C8)

Publish **with every number**: total params, training tokens × K (the
sandbox arm runs **K=4 trajectories** per token, R=2 debate rounds,
S=8 descent steps), estimated FLOPs, wall-clock, and — for the sandbox arm
— the ρ (`spread_ratio`) trace. Always state that the comparison is
architecture-matched, not parameter-matched: the sandbox arm carries +6.56M
params (15%) and roughly 2–3× the per-token compute of the baseline's T
forward. `--limit` runs are pipeline checks, never publishable results.

## 7. Outputs (per run dir)

```
runs/<variant>/
  tokenizer.json   shared BPE vocab (reused on resume)
  config.json      exact EkdantConfig used (authoritative on resume)
  last.pt          {model, cfg, step, completed, history} — saved every epoch
  results.json     {variant, use_sandbox, params, history[stage,epoch,loss,lr,gnorm], test{medmcqa,medqa}}
```

## 8. Known issues & troubleshooting

| symptom / item | detail |
|---|---|
| **LR stuck tiny / stage 2 at lr=0** | **FIXED** (per-stage cosine): the old `lr_at()` was called once per *epoch* with a *batch* budget (warmup never completed), and the first fix used a single global `opt_step` that never reset between stages — so stage 2 (MedQA) inherited stage 1's spent budget and trained at lr=0 (flat losses ~20.82 — inflated ×batch-size, true ≈3.5 at batch 6). Now the LR counter **resets at each stage boundary**, each stage runs its own warmup+cosine, the cosine is clamped to `p ∈ [0,1]`, and `opt_step` is checkpointed for resume. Runs from before the fix used `--warmup 1` as a workaround — don't compare their lr curve to a fixed-schedule run. Regression: `tests/test_schedule.py`. |
| L2 = 6.56M vs thesis 2.8M | debate uses full H×H cross-projections, untied across rounds; decide (slim/tie L2, or update thesis table) before publishing. Does not affect the A/B — baseline simply omits L2. |
| sandbox diagnostics not logged | `anchor_energy` / `trajectory_variance` / `spread_ratio` are computed in `aux` but not written into `results.json`; pull them from a manual forward if you need the ρ trace. A pinned-at-floor ρ (<0.05) means trajectory collapse — itself a result. |
| OOM | lower `--batch-size`, raise `--grad-accum` to keep effective batch constant. |
| `torch.compile` crash/slow first epoch | `--no-compile`. |
| first run is slow | HF is downloading MedMCQA/MedQA into its cache — one-time. |
| `pyarrow requires NumPy 2.0` | Jupyter/conda + unpinned clone pulled datasets 5.x/pyarrow 26 against numpy 1.x. **Fixed at source** (pyproject caps `datasets<5`, `pyarrow<21`) — `git pull && pip install -e .` self-repairs; manual: `pip install "datasets==3.6.0" "pyarrow<21"`. Never upgrade numpy on a conda image. |
| **accuracy locked at 0.00%** on a real run | was metric bug **T17** (fixed): `answer_target` held a vocab id while `pred` is a letter index, and eval read `answer_pos` instead of `answer_pos-1`. CE loss looked healthy throughout. If you see exactly 0.00% on thousands of examples, you are on a pre-fix build — `git pull`. Near-zero on a *smoke* run is fine (chance). |
| **"accuracy" ≈ your model's A-prediction rate; all targets are A** | was bug **T18** (fixed): MedMCQA's public **test** split ships `cop=-1` (unlabelled) for all 6,150 rows and the loader silently mapped it to "A". Now `letter_from_answer` returns `None` (row dropped) and the runner evaluates MedMCQA **validation** (4,183 labelled rows). If you see a truth histogram of all-A, you are on a pre-fix build — `git pull` and re-run `--eval-only`. |
| OOM on 16 GB GPU (Kaggle T4) | default batch 24 needs ~30–40 GB; use `--batch-size 6 --grad-accum 8` (effective batch unchanged). |
| `torch.amp has no attribute GradScaler` | torch < 2.3 (Kaggle ships 2.2) — **fixed in code**: scaler construction falls back to `torch.cuda.amp.GradScaler`; pull latest. On T4/P100 (no bf16) the runner also auto-falls back to fp16 autocast + loss scaling and logs it. |
| CPU-only machine | smoke/eval-only work (`--no-compile`); full training needs CUDA. |
| both arms fair? | same flags, same seed, separate `--out`; copy tokenizer for guaranteed identical vocab. |
| dataset schema | loader handles current HF schemas: MedMCQA `opa..opd` + `cop` (0..3; `cop=-1` rows are dropped — the public test split is unlabelled), MedQA `options` dict + `answer_idx` (`tests/test_data.py` locks this in). |
| **old train losses were inflated ×batch-size** | pre-fix epoch loss divided Σ(loss×batch) by batch *count* — with `--batch-size 6` every reported train CE was 6× the true value (e.g. "22.28" ≈ true 3.7). Fixed (sample-mean now); eval losses were always correct. Comparisons between same-batch runs are unaffected — only absolute values changed. |

## 9. Quick reference

```bash
cd 09-Code && pip install -e .[dev]
python -m pytest                                        # 75 unit tests
# full A/B (one arm at a time)
torchrun --nproc_per_node=4 -m ekdant.train --variant sandbox  --out runs/sandbox
torchrun --nproc_per_node=4 -m ekdant.train --variant baseline --out runs/baseline
# after an interruption: same command + --resume
python -m ekdant.train --variant sandbox --out runs/sandbox --resume
# re-benchmark an existing checkpoint
python -m ekdant.train --variant sandbox --out runs/sandbox --eval-only
```
