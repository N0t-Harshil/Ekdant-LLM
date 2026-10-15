# run.md — Ekdant operations runbook

How to **start training, resume, test, and benchmark** both models. Every
command here was executed and verified (CPU smoke: 64 rows, 1 epoch, both
arms; full unit suite: 78 passed). All commands run from `09-Code/` unless
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
| GPU (full run) | batches pad to batch max (MCQ prompts ≈ 40–80 tokens), so memory stays far below the 2048 cap — measured 6.3 opt steps/s at batch 6×8 on a 16 GB T4; `max_seq_len` 2048 is only the truncation limit |

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
- **Kaggle T4/P100 = 16 GB VRAM** — MCQ batches pad to batch max (~50
  tokens), so memory is modest; the campaign runs `--batch-size 6
  --grad-accum 8` (effective batch 48, same as the 24×2 default) at a
  measured 6.3 opt steps/s. Use `--no-compile` if compile is slow or
  crashes, and shrink the batch only if a long-input batch OOMs.
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

### Recommended long run — the 100-epoch campaign (T4)

**Decision: 100 epochs per phase, sequential protocol.** One invocation
runs the two phases **in order, never trained jointly** (at
`--batch-size 6 --grad-accum 8`, effective batch 48):

1. **Phase 1 — MedQA:** train on the **full** MedQA train split
   (10,178 rows → 212 opt/epoch), then record the **MedQA test**
   result (`phase1` in `results.json`).
2. **Phase 2 — MedMCQA fine-tune:** fine-tune on **60% of MedMCQA
   train** (109,497 of 182,496 labelled rows → 2,282 opt/epoch; the
   326 unlabelled `cop=-1` rows carry no answer and are the only rows
   dropped at load). The 60/40 partition is a `--seed`-derived
   permutation (`split_frac()` in `train.py`, locked by a test) —
   **identical for both A/B arms**.
3. **Final benchmark:** score **the complementary 40% of MedMCQA**
   (72,999 rows — never trained on) exactly once, plus MedQA test
   again (retention after the fine-tune) and the selection-set
   validation score.

100 epochs × (212 + 2,282) = **249,400 optimizer steps** total. Steps
per epoch cannot be raised without shrinking the effective batch (same
data, lr 3e-4 tuned at 48) — total steps grow by training longer, which
is what this recipe does.

```bash
cd 09-Code
nohup python -m ekdant.train --variant baseline --out runs/baseline_100ep \
  --batch-size 6 --grad-accum 8 --epochs 100 > baseline_100ep.log 2>&1 &
tail -f baseline_100ep.log            # watch live; Ctrl-c stops watching only

# after it writes results.json (≈11.5 h), same command for the sandbox arm:
nohup python -m ekdant.train --variant sandbox --out runs/sandbox_100ep \
  --batch-size 6 --grad-accum 8 --epochs 100 > sandbox_100ep.log 2>&1 &
```

**Measured budget (live telemetry, baseline arm):** the collator pads
batches to *batch max* (`data.py::collate`) and MedMCQA/MedQA prompts
are only ~40–80 tokens — `max_seq_len` 2048 is a truncation cap, never
padded to — so a 16 GB T4 sustains **≈6.3 optimizer steps/s ≈ 48 min
per MedMCQA fine-tune epoch** (fp16 fallback on T4, ~11.9K real tok/s).
Baseline arm ≈ **11.5 h** (phase 1 ≈ 56 min + phase 2 ≈ 10 h + 100
validation evals ≈ 35 min + final evals). Sandbox arm does K=4
trajectories + debate + descent per token — expect roughly 1.5× (read
its first ETA line) ≈ **17 h**. Full campaign **≈ 1.2 days sequential**.
Epoch checkpoints + `nohup` make it resume-safe (add `--resume` after a
pod restart). Scale guide (projected from the measured 6.3 steps/s;
sandbox ≈ 1.5× baseline; steps = epochs × 2,494):

| `--epochs` | optimizer steps | T4 wall-clock (both arms) | note |
|---|---|---|---|
| 3 | ~7.5K | ~1 h | quick protocol check |
| 14 | ~35K | ~4 h | thesis-class (~50K steps at 20 ep) |
| 40 | ~100K | ~12 h | earlier campaign draft |
| **100** | **~249K** | **~28 h (b ≈11.5 h, sbx ≈17 h)** | **the campaign — sequential MedQA→MedMCQA 60/40** |

> **Reading epoch-0 ETA lines:** the first lines are inflated because
> `torch.compile` warmup (~150 s, step-100 `tok/s`=1.1K vs 11.8K steady)
> gets averaged in — your ETA will keep falling (69 h → 37 h → … → 9.7 h
> at step 2000) and converge. The window resets every epoch, so from
> ep1 on the first line is accurate. A falling ETA is warmup washout,
> not a misconfiguration. Phase 1's ETA counts only the 212 steps/epoch
> MedQA stage — it is *not* a bug that phase 1 is fast.

### Benchmark protocol: MedQA → test → MedMCQA 60% fine-tune → 40% benchmark

The model is **trained and tested on MedQA first**, then **fine-tuned
on 60% of MedMCQA**, and only then benchmarked on **the rest** — the
protocol you asked for, in one invocation, strictly sequential:

| key | data | role | when the model sees it |
|---|---|---|---|
| `phase1.medqa` | MedQA test 1,273 | **phase-1 test** (trained on all of MedQA) | once, right after phase 1 |
| `medmcqa_val` | MedMCQA validation 4,183 (untouched by training) | **checkpoint selection** — `best.pt` tracking + final better-of | every epoch of phase 2 (`--eval-every`), then at selection |
| `medmcqa_bench` | **40% of MedMCQA train** (72,999 rows) | **the benchmark** — the number you quote | **exactly once**, at the end, on the selected checkpoint |
| `medqa` | MedQA test 1,273 | **retention** after the fine-tune | once at the end (compare against `phase1.medqa`) |

**Selection (`best.pt` vs `last.pt` — "change the best with last if it
is good"):** during phase 2, `best.pt` is refreshed whenever the
untouched validation accuracy improves (the 40% benchmark is never
looked at during training). At the end the runner **evaluates both
`best.pt` and `last.pt` on `medmcqa_val` and loads whichever is better**
(ties keep `best.pt`); `results.json` records
`selected: {checkpoint, selected_on: "medmcqa_val", best_val_accuracy,
last_val_accuracy}` — **quote `medmcqa_bench` from the selected
checkpoint**, and quote `phase1.medqa` vs final `medqa` as the
forgetting comparison. The fairness rule does the rest: same `--seed`
→ the identical 60/40 partition and the identical benchmark rows for
both arms. Note: every per-epoch eval loads data from the local HF
cache offline (module default in `data.py` — hub etag checks are
skipped, so evals can never stall on the network).

### Watching training progress

Every `--log-every` optimizer steps (default 100) the run prints a
progress line; `tail -f <out>.log` shows the run advancing live:

```
=== stage medqa  n=10178 (epoch 0+) | 212 opt/epoch x 100 epochs = 21200 steps, lr 0.0003 cosine, warmup 200 ===
[medqa] ep0 step 100/212 (47%) loss=5.1042 lr=2.85e-4 gnorm=1.14 tok/s=11850 eta=0m19s
[medqa] ep0 done loss=4.6110 gnorm=1.22 lr_end=2.30e-4 tok/s=11860 time=0m34s
[phase1 test medqa] acc=31.20% loss=2.8744 n=1273  (after MedQA training, before MedMCQA fine-tune)
=== stage medmcqa_ft  n=109497 (epoch 0+) | 2282 opt/epoch x 100 epochs = 228200 steps, lr 0.0003 cosine, warmup 200 ===
[medmcqa_ft] ep0 step 1200/2282 (53%) loss=3.2105 lr=2.91e-4 gnorm=1.32 tok/s=11812 eta=9m36s
[medmcqa_ft] ep0 done loss=2.9114 gnorm=1.41 lr_end=2.30e-4 tok/s=11860 time=48m10s
[medmcqa_ft] ep0 eval medmcqa_val acc=31.42% loss=3.3910 n=4183
[medmcqa_ft] ep0 new best medmcqa_val 31.42% -> best.pt
selection on medmcqa_val: last.pt=34.01%  best.pt=34.52%
final benchmark uses best.pt (medmcqa_val 34.52%)
[test medmcqa_bench] acc=33.90% loss=3.1022 n=72999
[test medqa] acc=30.80% loss=3.0114 n=1273
```

(The phase-1 result appears once, between the phases; the 40%
benchmark (`[test medmcqa_bench]`) appears **only once, at the very
end**.)

How to read it:

| field | meaning / healthy signal |
|---|---|
| `step 1200/2282 (53%)` | optimizer steps into the phase and % of the **phase** budget — this is your "how far gone" number |
| `loss` | EMA of CE per token — should fall every epoch; a flat line at init lr≈0 means a schedule bug (§8) |
| `lr` | current learning rate — rises through warmup (default 200 steps) then cosine-decays to ~0 at the phase end (each phase gets its own cycle) |
| `gnorm` | global grad norm **before** clipping at 1.0 — sustained ≫1 means the clip is active every step; a sudden collapse toward 0 means training has stalled; expect ~0.5–3 in steady state |
| `tok/s` | throughput — drops when batches are long (2048-token rows) or the GPU is contended |
| `eta` | projected time to finish the **phase** from the current rate |
| `[phase1 test medqa] …` | the phase-1 result (MedQA test, recorded once after phase 1, before the fine-tune) — compare it to the final `[test medqa]` for forgetting |
| `ep0 done …` | epoch summary: true sample-mean CE (comparable to eval loss), mean gnorm, LR at epoch end, wall time |
| `eval medmcqa_val …` / `new best …` | phase-2 selection eval (untouched MedMCQA validation, `--eval-every 1`); `new best … -> best.pt` means `best.pt` was refreshed on the selection set — the 40% benchmark is never evaluated here |
| `selection on medmcqa_val: …` | the final better-of comparison: both checkpoints' validation accuracy, then `final benchmark uses <ckpt>` names the winner (ties keep `best.pt`) |

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
3. **Phase 1 — MedQA:** trains on the **full** MedQA train split
   (10,178 rows — header prints the real `n=`) for `--epochs` epochs,
   then records the **MedQA test** result once into
   `results["phase1"]["medqa"]`.
4. **Phase 2 — MedMCQA fine-tune:** trains on the **60% fine-tune
   partition** of MedMCQA train (109,497 of 182,496 labelled rows —
   182,822 raw minus the 326 unlabelled `cop=-1` rows, dropped by the
   T18 rule) for `--epochs` epochs. The partition is the head of a
   `--seed` permutation (`split_frac`, `MCQA_FT_FRAC = 0.6`) — the
   identical rows for both arms; the complementary 40% is never trained
   on.
5. During **every phase-2 epoch** (`--eval-every 1`): saves
   `<out>/last.pt` (weights + step + completed counter + opt step +
   history + best-val + config), evaluates the **untouched MedMCQA
   validation** split (4,183 — the public test set ships `cop=-1`,
   unlabelled, T18), appends it to the epoch's history entry, rewrites
   `<out>/results.json` (crash-safe), and refreshes `<out>/best.pt` on
   a new `medmcqa_val` best. Phase 1 has no per-epoch eval.
6. After training: **selects the checkpoint** — evaluates both
   `best.pt` and `last.pt` on `medmcqa_val` and loads the better one
   (ties keep `best.pt`; logged as `selection on medmcqa_val: …`), then
   runs the final evaluation by letter-accuracy: the **40% benchmark
   partition** (`medmcqa_bench`, scored exactly once), MedQA test
   (retention), and the selection-set score — written to
   `<out>/results.json` under `test{medmcqa_bench, medmcqa_val, medqa}`
   + `phase1` + `selected`.

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
| `--epochs` | 3 | epochs **per phase** (MedQA phase 1 and MedMCQA fine-tune phase 2 each) |
| `--limit` | none | cap rows per dataset (smoke/debug only) |
| `--seed` | 17 | torch seed + tokenizer sample + data sampling + **the 60/40 partition** (identical for both arms) |
| `--no-compile` | off | skip `torch.compile` (use if compile crashes or first-run overhead hurts) |
| `--log-every` | 100 | progress line every N optimizer steps (`0` = epoch summaries only) |
| `--eval-every` | 1 | during phase 2: evaluate the **untouched MedMCQA validation** every N epochs → history + incremental `results.json` + `best.pt` on `medmcqa_val` (`0` = no per-epoch eval; final selection falls back to `last.pt`) |
| `--resume` | off | continue an interrupted run (§4) |
| `--eval-only` | off | skip training; run the final protocol (selection + benchmark) (§5/§7) |

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
| per-phase LR counter `opt_step` | continues inside the current phase's cosine cycle |
| completed (stage, epoch) pairs | already-finished pairs are **skipped** |
| loss/LR/gnorm/eval history | appended, not restarted |
| `best_medmcqa` | `best.pt` (validation-based) tracking continues from the recorded best |

| restored automatically from `<out>/` | note |
|---|---|
| `tokenizer.json` | reused, never retrained — vocab cannot drift |
| `config.json` | **exact config the run started with** takes precedence over `configs/` |
| `results.json` → `phase1` | the phase-1 MedQA-test record survives a resume/eval-only rewrite |

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
python -m pytest            # 78 tests
```

Covers: config/param accounting, T block (causal+window mask exactness,
no-leakage, GQA mapping, RoPE, shapes), L2 sandbox (K trajectories,
debate rounds, descent, collapse floor, aux keys), end-to-end forward/loss
decrease for both variants, the data layer (MedMCQA `opa..opd`/`cop`
0-indexed schema, MedQA `options`+`answer_idx` schema, rejection rules),
and the **60/40 protocol partition** (`test_eval_split.py`: the seeded
`split_frac` partition is deterministic, disjoint, complete, and locked
to `MCQA_FT_FRAC = 0.6`).

### Held-out eval (no training)

```bash
python -m ekdant.train --variant sandbox  --out runs/sandbox  --eval-only --no-compile
python -m ekdant.train --variant baseline --out runs/baseline --eval-only --no-compile
```

Loads `last.pt` + `tokenizer.json` from `--out`, skips training, then
runs the **same final protocol as a normal run**: evaluate `best.pt` vs
`last.pt` on `medmcqa_val`, load the better, score the 40% benchmark
partition + MedQA test, rewrite `results.json` (a recorded `phase1` from
`results.json` is preserved). Use it to re-score a checkpoint on a
different machine, or to re-run the benchmark after a `--resume` finish
(every training invocation already evaluates at the end — `--eval-only` is
for re-evaluation without training).

Errors are explicit: `--eval-only` into an empty/nonexistent `--out`
fails with `tokenizer.json not found` / `last.pt not found`.

## 6. Benchmarking

**Metric:** held-out **letter-accuracy** — the logits at the answer position
(`answer_pos - 1`, the position that predicts the answer), argmax over the
A/B/C/D token logits vs the letter **index** (0–3), on the **40% benchmark
partition of MedMCQA train** (72,999 rows), MedQA test (1,273), and the
untouched MedMCQA validation (4,183, selection set only).
MedMCQA's public test split has `cop = -1`
for every row (no released labels), so it must never be scored — see bug
T18. Not free-form generation.

**Which key to quote:** `test.medmcqa_bench` (rows never trained on,
scored once at the end on the selected checkpoint) is the defensible test
number; `phase1.medqa` vs final `test.medqa` is the forgetting comparison
(MedQA before vs after the MedMCQA fine-tune); `test.medmcqa_val` is the
selection metric (never quote it as a result — it chose the checkpoint);
`selected` says which checkpoint ran the benchmark.

### Full procedure

```bash
# 1. train both arms (identical flags — incl. --seed, which locks the
#    60/40 partition — separate dirs)
python -m ekdant.train --variant sandbox  --out runs/sandbox  --epochs 100
python -m ekdant.train --variant baseline --out runs/baseline --epochs 100

# 2. compare
python - <<'PY'
import json
for v in ("baseline", "sandbox"):
    r = json.load(open(f"runs/{v}/results.json"))
    t, s = r["test"], r["selected"]
    p1 = r.get("phase1", {}).get("medqa", {}).get("accuracy")
    print(f"{v:9s} params={r['params']/1e6:.2f}M ckpt={s['checkpoint']}  "
          f"bench(medmcqa40)={t['medmcqa_bench']['accuracy']*100:.2f}%  "
          f"medqa phase1={(p1 or 0)*100:.2f}% -> final="
          f"{t['medqa']['accuracy']*100:.2f}%")
PY
```

Produces one line per arm; any accuracy gap is attributable to L2 alone
(architecture-matched, same `--seed` → identical 60/40 partition and
identical benchmark rows).

### Interpreting your numbers (reference magnitudes)

Full 3-epoch A/B on a T4 (`--batch-size 6 --grad-accum 8`, full data, one
seed, both arms run on the per-stage LR fix) — the reference for "what a
complete run looks like":

| arm | params | MedMCQA val (4,170) | MedQA test (1,273) |
|---|---|---|---|
| baseline | 43.53M | 33.31% | 25.69% |
| sandbox | 50.09M | 32.01% | 26.08% |

*(Pre-protocol reference numbers: these 3-epoch runs predate the
sequential 60/40 protocol — they trained jointly (MedMCQA→MedQA) and
their "MedMCQA val" column is the full validation split, comparable
today to `test.medmcqa_val`; there is **no** old counterpart to
`test.medmcqa_bench`. Never compare an old joint-run number against a
new `medmcqa_bench`.)*

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
  last.pt          {model, cfg, step, completed, opt_step, history, best_medmcqa} — every epoch (resume artifact)
  best.pt          same payload at the peak medmcqa_val accuracy (phase 2) — candidate checkpoint
  results.json     {variant, use_sandbox, params,
                    history[stage,epoch,loss,lr,gnorm,test{medmcqa_val} (phase-2 epochs)],
                    phase1{medqa},            ← MedQA test right after phase 1
                    test{medmcqa_bench, medmcqa_val, medqa},
                    selected{checkpoint: best.pt|last.pt, selected_on: medmcqa_val,
                             best_val_accuracy, last_val_accuracy}}
```

The final protocol loads whichever checkpoint `selected.checkpoint`
names — that is the checkpoint the `medmcqa_bench` number belongs to.

## 8. Known issues & troubleshooting

| symptom / item | detail |
|---|---|
| **LR stuck tiny / stage 2 at lr=0** | **FIXED** (per-stage cosine): the old `lr_at()` was called once per *epoch* with a *batch* budget (warmup never completed), and the first fix used a single global `opt_step` that never reset between stages — so stage 2 (MedQA) inherited stage 1's spent budget and trained at lr=0 (flat losses ~20.82 — inflated ×batch-size, true ≈3.5 at batch 6). Now the LR counter **resets at each stage boundary**, each stage runs its own warmup+cosine, the cosine is clamped to `p ∈ [0,1]`, and `opt_step` is checkpointed for resume. Runs from before the fix used `--warmup 1` as a workaround — don't compare their lr curve to a fixed-schedule run. Regression: `tests/test_schedule.py`. |
| L2 = 6.56M vs thesis 2.8M | debate uses full H×H cross-projections, untied across rounds; decide (slim/tie L2, or update thesis table) before publishing. Does not affect the A/B — baseline simply omits L2. |
| sandbox diagnostics not logged | `anchor_energy` / `trajectory_variance` / `spread_ratio` are computed in `aux` but not written into `results.json`; pull them from a manual forward if you need the ρ trace. A pinned-at-floor ρ (<0.05) means trajectory collapse — itself a result. |
| OOM | lower `--batch-size`, raise `--grad-accum` to keep effective batch constant. |
| `torch.compile` crash/slow first epoch | `--no-compile`. |
| first run is slow | HF is downloading MedMCQA/MedQA into its cache — one-time. Loads are **offline-first** by default (`data.py` sets `HF_HUB_OFFLINE`): after the cache is warm nothing touches the hub. |
| **hangs silently right before/inside an eval (no log line for minutes)** | was a **hub etag stall**: every `load_dataset` did a network HEAD to check for dataset updates, and on a dead/IPv6 route that request never completes — training looked frozen with no error. **Fixed at source**: `data.py` defaults to offline loads (cache-only) and only goes online for the single first-run download. Pull latest; `HF_HUB_OFFLINE=0` forces hub checks if you deliberately want them. |
| `pyarrow requires NumPy 2.0` | Jupyter/conda + unpinned clone pulled datasets 5.x/pyarrow 26 against numpy 1.x. **Fixed at source** (pyproject caps `datasets<5`, `pyarrow<21`) — `git pull && pip install -e .` self-repairs; manual: `pip install "datasets==3.6.0" "pyarrow<21"`. Never upgrade numpy on a conda image. |
| **accuracy locked at 0.00%** on a real run | was metric bug **T17** (fixed): `answer_target` held a vocab id while `pred` is a letter index, and eval read `answer_pos` instead of `answer_pos-1`. CE loss looked healthy throughout. If you see exactly 0.00% on thousands of examples, you are on a pre-fix build — `git pull`. Near-zero on a *smoke* run is fine (chance). |
| **"accuracy" ≈ your model's A-prediction rate; all targets are A** | was bug **T18** (fixed): MedMCQA's public **test** split ships `cop=-1` (unlabelled) for all 6,150 rows and the loader silently mapped it to "A". Now `letter_from_answer` returns `None` (row dropped) and the runner evaluates MedMCQA **validation** (4,183 labelled rows). If you see a truth histogram of all-A, you are on a pre-fix build — `git pull` and re-run `--eval-only`. |
| OOM on 16 GB GPU (Kaggle T4) | batch-max padding means MCQ batches are small (measured: batch 6×8 ≈ 6.3 steps/s); OOM should only come from long inputs or bigger batches — if it happens, lower `--batch-size`, raise `--grad-accum` to keep effective batch 48. |
| `torch.amp has no attribute GradScaler` | torch < 2.3 (Kaggle ships 2.2) — **fixed in code**: scaler construction falls back to `torch.cuda.amp.GradScaler`; pull latest. On T4/P100 (no bf16) the runner also auto-falls back to fp16 autocast + loss scaling and logs it. |
| CPU-only machine | smoke/eval-only work (`--no-compile`); full training needs CUDA. |
| both arms fair? | same flags, same seed, separate `--out`; copy tokenizer for guaranteed identical vocab. |
| dataset schema | loader handles current HF schemas: MedMCQA `opa..opd` + `cop` (0..3; `cop=-1` rows are dropped — the public test split is unlabelled), MedQA `options` dict + `answer_idx` (`tests/test_data.py` locks this in). |
| **old train losses were inflated ×batch-size** | pre-fix epoch loss divided Σ(loss×batch) by batch *count* — with `--batch-size 6` every reported train CE was 6× the true value (e.g. "22.28" ≈ true 3.7). Fixed (sample-mean now); eval losses were always correct. Comparisons between same-batch runs are unaffected — only absolute values changed. |

## 9. Quick reference

```bash
cd 09-Code && pip install -e .[dev]
python -m pytest                                        # 78 unit tests
# full A/B (one arm at a time)
torchrun --nproc_per_node=4 -m ekdant.train --variant sandbox  --out runs/sandbox
torchrun --nproc_per_node=4 -m ekdant.train --variant baseline --out runs/baseline
# after an interruption: same command + --resume
python -m ekdant.train --variant sandbox --out runs/sandbox --resume
# re-benchmark an existing checkpoint
python -m ekdant.train --variant sandbox --out runs/sandbox --eval-only
```
