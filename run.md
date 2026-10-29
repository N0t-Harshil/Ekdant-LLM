# run.md — Ekdant operations runbook

How to **install, train, resume, test and benchmark** Ekdant, from a clean
machine to a number you can quote. All commands run from `09-Code/` unless
stated otherwise.

There are two experiments in the code:

| | **v2 — current** (Part A) | v1 — from-scratch pilot (Part B) |
|---|---|---|
| backbone | pretrained **SmolLM2-135M** | 43.53M model trained from scratch |
| arms | `baseline`, `sandbox` (+19.59M L2) | `baseline`, `sandbox` |
| data | PubMedQA → MedQuAD (write the answer) | MedMCQA → MedQA (pick A–D) |
| main number | reference-answer perplexity on the MedQuAD benchmark half | letter accuracy on `medqa_bench` |
| start with | `python -m ekdant.train --task gen …` | `python -m ekdant.train …` (`--task mcq` is the default) |

v1 is kept so its results can be reproduced: every v1 result sat at the
majority-letter rate (T27), which is why v2 exists — see
[30_Design_v2](07-Management/30_Design_v2_Pretrained_Backbone.md).

---

# Part A — v2: pretrained backbone, generative medical QA

## A1. Quick start

```bash
git clone https://github.com/N0t-Harshil/Ekdant-LLM.git
cd Ekdant-LLM/09-Code
pip install -e .[dev]               # trailing dot matters; installs transformers too
python -m pytest                    # all tests must pass (CPU, no downloads)
# alternative, dependencies only: pip install -r requirements.txt
```

**Requirements** (`09-Code/requirements.txt`, identical to `pyproject.toml`):

| package | pin | why |
|---|---|---|
| `torch` | `>=2.2` | the pod runs **2.2.2+cu121** (CUDA 12.2 driver); newer works too |
| `transformers` | `>=4.45,<5` | **5.x needs PyTorch ≥ 2.6** and silently disables its models on 2.2.2 (T30); 4.57.6 is what the pin installs |
| `datasets` | `>=2.19,<5` | 5.x needs pyarrow ≥ 24 → numpy ≥ 2, which breaks numpy-1.x conda images |
| `pyarrow` | `>=12,<21` | same reason |
| `tokenizers` | `>=0.19` | v1 BPE tokenizer and SmolLM2's |
| `pytest` | `>=8` | tests (`[dev]` extra) |

The same list is in `09-Code/requirements.txt` and `pyproject.toml`
(`tests/test_requirements.py` fails if they drift apart). Tested combinations,
all tests green plus a real-SmolLM2 smoke run: **PyTorch 2.2.2 + transformers
4.57.6** (datasets 4.0.0, pyarrow 20.0.0, tokenizers 0.22.2, numpy 1.26.4 — a
mirror of the pod) and PyTorch 2.9.0 on the development machine.

The first real run downloads SmolLM2-135M (~270 MB), PubMedQA `pqa_artificial`
(~233 MB) and MedQuAD (~9 MB) into the Hugging Face cache; later runs read the
cache offline.

**Config files:** `configs/v2_baseline.json` and `configs/v2_sandbox.json`,
identical except `"addon"` (`none` / `sandbox`; a test enforces this). Each run
loads `configs/v2_<variant>.json` automatically; `--config` overrides it, and a
config whose `addon` does not match `--variant` is refused.

**Hugging Face token (optional).** SmolLM2-135M, PubMedQA and MedQuAD are
public, so no token is needed; one only avoids anonymous download rate limits.
Both configs point to a token **file**, never the token itself:

```bash
echo "hf_xxxxxxxxxxxxxxxx" > configs/hf_token.txt   # from 09-Code/; git-ignored
```

The run reads the first line, exports it as `HF_TOKEN` and logs only where it
came from (`hugging face token: file`). An `HF_TOKEN` environment variable
takes priority. The token is never written to the log, `last.pt` or
`results.json` (tested), and `.gitignore` keeps `hf_token.txt` out of git —
never put a token in a `.json` config, because the config is saved into every
checkpoint and results file and would be pushed to GitHub.

## A2. What one run does

```
zero-shot      score the untrained arm on every evaluation set (the floor)
phase 1        PubMedQA: question + abstract → write the conclusion
               (209,269 rows; 2,000 held out, scored every 25 % of the epoch)
phase 2        MedQuAD fine-tune half: question → write the answer
               (10 % of its questions held back to pick the best checkpoint)
final          better of best.pt / last.pt on that 10 % → score the MedQuAD
               benchmark half ONCE + the PubMedQA held-out rows again
```

MedQuAD is split by **unique question** (16,407 rows, 14,345 questions): every
row sharing a question lands in the same half, so no benchmark answer leaks
into training. The split, the hold-out, the data order and the starting
weights are identical for both arms; L2 starts as a no-op (its gate is 0),
so at step 0 both arms give the backbone's exact output.

| arm (`--variant`) | added on top of SmolLM2 | added params |
|---|---|---|
| `baseline` | nothing | 0 |
| `sandbox` | L2: 4 trajectories, 4 debate rounds (4× MLP), 8 anchored descent steps, gated residual | 19.59M |

The comparison is architecture-matched, not parameter-matched: the sandbox arm
has 19.59M more parameters and does more work per token. Say so next to every
number.

## A3e. Forced L2: L2 must be used (T42) — **run this next**

In every run so far L2 was **optional**: each gate starts at 0 and the model
opened it only as far as it found L2 useful (v25: end gate 0.076, mix 0.120).
These two runs **force** L2 to be used: `--gate-floor 0.5 --gate-force`.

- After 200 steps (L2 learns first) a minimum gate rises smoothly from 0 to
  **0.5** over 2,000 steps and is then held. The model can open the gate
  further (up to 1.0) but **never below 0.5** — L2 is always at least as
  important as the stream it joins.
- At the end: the final state is at least half L2's output.
- Inside the network (mix): L2's output is scaled to the size of the
  residual stream, so 0.5 means L2 adds half as much as the stream holds
  (without this, L2's output would be ~100× smaller than the stream at
  layer 20 and forcing would mean nothing).
- L2's output scale is frozen, so the model cannot hide L2 by shrinking it.
- **Fast, then controlled (T43):** the minimum starts rising after 100 steps
  and reaches 0.5 in **500 steps** (4× faster than before). Every 100 steps the
  trainer keeps a copy of the last good state in memory. **If L2 bursts, it
  goes back to that copy** (weights and optimiser), then re-opens **half as
  fast** and holds the minimum **0.05 lower** (never below 0.4 = 0.8 × 0.5);
  the brake also halves L2's learning rate. Each burst slows and lowers it a
  bit more, so the gate settles at the highest opening L2 can handle.

The comparison then has **five models**, all seed 17, 3 PubMedQA + 5 MedQuAD epochs:

| # | run | L2 | status |
|---|---|---|---|
| 1 | `v25_baseline` | none | done (A7e) |
| 2 | `v25_end` | at the end, optional (free gate) | done (A7e) |
| 3 | `v25_mix` | layers 10, 15, 20–30 + end, optional | done (A7e) |
| 4 | `v26_end_forced` | at the end, **forced** (gate ≥ 0.5) | run now |
| 5 | `v26_mix_forced` | layers 10, 15, 20–30 + end, **forced** | run now |

Same flags as A3d (3 PubMedQA + 5 MedQuAD epochs, seed 17), so both runs
compare directly with `runs/v25_baseline`, `v25_end` and `v25_mix`.

```bash
cd ~/work/Ekdant-LLM && git pull && cd 09-Code
python -m pytest -q                          # all tests must pass
nvidia-smi                                   # two GPUs with >= 30 GB free (not GPU 0)
FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 3 --ft-epochs 5 --min-free-gb 30"
FORCE="--gate-floor 0.5 --gate-force --gate-floor-start 100 --gate-floor-steps 500 --gate-rollback"
CUDA_VISIBLE_DEVICES=1 nohup python -m ekdant.train $FLAGS $FORCE --variant sandbox --config configs/v2_sandbox_final.json --out runs/v26_end_forced > v26_end_forced.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m ekdant.train $FLAGS $FORCE --variant sandbox --config configs/v2_sandbox_mix.json   --out runs/v26_mix_forced > v26_mix_forced.log 2>&1 &

# afterwards:
python -m ekdant.gen_eval compare runs/v25_baseline runs/v25_end runs/v25_mix runs/v26_end_forced runs/v26_mix_forced
```

Time (from the v25 logs): end ~4 h, mix ~17 h (the snapshots add well under
1 %; they need ~2 GB of CPU RAM). Watch `|g|=` reach ≥ 0.50 after ~600 steps
and stay there; a `rollback:` line in the log means a burst was undone (the
log shows the new target and ramp length). If you already started the A3e
runs without `--gate-rollback`, stop them and start again with these flags
in a new `--out` folder. Expect the loss to rise when
the floor comes up — that is the experiment: can the model learn to make good
use of L2 when it cannot ignore it? Copy both `results.json` files and logs back.

## A3d. More epochs: the three models again (T41) — done, results in A7e

The whole model is trained, as before (no LoRA): every SmolLM2 weight and
all of L2. The only change, applied **identically to all three models**, is
more training: **3 PubMedQA epochs** (was 1) and **5 MedQuAD epochs** (was 3).
Because of the longer training, the old `v22_*` runs are **not** comparable —
run all three fresh:

| # | run | config | trained params |
|---|---|---|---|
| 1 | `v25_baseline` | `configs/v2_baseline.json` | 134,515,008 (SmolLM2) |
| 2 | `v25_end` | `configs/v2_sandbox_final.json` | 154,109,952 (SmolLM2 + L2) |
| 3 | `v25_mix` | `configs/v2_sandbox_mix.json` | 154,123,776 (SmolLM2 + shared L2 at 9 sites) |

```bash
cd ~/work/Ekdant-LLM && git pull && cd 09-Code
python -m pytest -q                          # all tests must pass
nvidia-smi                                   # pick GPUs with >= 30 GB free (not GPU 0)
FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 3 --ft-epochs 5 --min-free-gb 30"
```

**Fastest — three GPUs at once** (wall-clock = the mix, the slowest run):

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -m ekdant.train $FLAGS --variant baseline --config configs/v2_baseline.json     --out runs/v25_baseline > v25_baseline.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_final.json --out runs/v25_end      > v25_end.log      2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_mix.json   --out runs/v25_mix      > v25_mix.log      2>&1 &
```

**Or one GPU, one after another:**

```bash
export CUDA_VISIBLE_DEVICES=2
nohup bash -c "
python -m ekdant.train $FLAGS --variant baseline --config configs/v2_baseline.json     --out runs/v25_baseline > v25_baseline.log 2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_final.json --out runs/v25_end      > v25_end.log      2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_mix.json   --out runs/v25_mix      > v25_mix.log      2>&1
" > v25_all.log 2>&1 &
```

Time: three times the PubMedQA data, so roughly **3× the A3 times** (A3 baseline
took 1 h 20 min) — a rough guess, not measured: baseline ~4 h, end ~5 h,
mix ~9–10 h. A run can be resumed with the **same** flags plus `--resume`.

Afterwards:

```bash
python -m ekdant.gen_eval compare runs/v25_baseline runs/v25_end runs/v25_mix
```

Copy the three `results.json` files and logs back for the thesis.

## A3. The v2.2 experiment: four runs, one after another

Four models with **identical flags** (seed 17, 1 PubMedQA epoch, 3 MedQuAD
epochs, gate LR 1e-3, L2 matrices LR 1e-4), differing **only** in where L2
sits:

| # | run (`--out`) | config | L2 |
|---|---|---|---|
| 1 | `runs/v22_baseline` | `configs/v2_baseline.json` | none — SmolLM2 alone |
| 2 | `runs/v22_mid` | `configs/v2_sandbox.json` | once, after layer 20 of 30 (T35) |
| 3 | `runs/v22_multi` | `configs/v2_sandbox_multi.json` | **the same L2** after layers 10, 20 and 30, each site with its own gate and norms (T36) |
| 4 | `runs/v22_end` | `configs/v2_sandbox_final.json` | once, after the final norm (the v2.0 placement) |

Parameters: baseline 134,515,008; run 2 +19,595,520; run 3 +19,598,976
(one shared L2 — only 3,456 more than run 2); run 4 +19,594,944.

**Gate control (v2.3, T37) — all sandbox runs:** (1) **boost** — the gate LR is 10× higher at the start of phase 1 (1e-2), fading smoothly to 1e-3 over the first 1,000 steps (`--gate-boost 10 --gate-boost-steps 1000`); (2) **soft cap** — the effective gate is `1.0 · tanh(θ / 1.0)`: 0 at init, the same slope there, never above 1 (`"gate_cap": 1.0` in the configs); (3) **brake** — if L2's gradient norm jumps above 3× its running average or L2's size `l2` exceeds 50 (`--brake-spike 3 --brake-l2-max 50`), every gate is multiplied by 0.9 and the L2 and gate LRs are halved; each healthy step they recover ×1.01 (floor 0.05). Brake events are logged (`brake: …`), counted in `cost.brake_events`, and the brake state is saved in `last.pt`. The baseline is unaffected.

Why 1 epoch: with 2 PubMedQA epochs both models got worse on the benchmark
(v2.1: 3.914 vs 3.782 with 1 epoch), so the 1-epoch recipe is kept.

```bash
cd ~/work/Ekdant-LLM && git pull && cd 09-Code
python -m pytest -q                          # all tests must pass
nvidia-smi                                   # pick ONE GPU with >= 40 GB free (run 3 needs the most)
export CUDA_VISIBLE_DEVICES=2                # <- the GPU you picked
FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 1 --ft-epochs 3 --min-free-gb 40"

nohup bash -c "
python -m ekdant.train $FLAGS --variant baseline --config configs/v2_baseline.json       --out runs/v22_baseline > v22_baseline.log 2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox.json        --out runs/v22_mid      > v22_mid.log      2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_multi.json  --out runs/v22_multi    > v22_multi.log    2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_final.json  --out runs/v22_end      > v22_end.log      2>&1
" > v22_all.log 2>&1 &

tail -f v22_baseline.log                     # then v22_mid.log, v22_multi.log, v22_end.log
```

The four runs start one after another; if one fails, the next one still
starts (`;`). Each refuses to start with less than 40 GB free on its GPU. If
you have two free GPUs, you can instead split the four commands over them —
the results are the same, only the wall-clock changes.

**Expected time** (from the measured runs; run 3 is an estimate): baseline
~1 h 20 min, run 2 ~1 h 45 min, run 3 ~2 h 30 min (L2 runs three times per
token), run 4 ~1 h 40 min — about **7 hours** in total. Peak memory: 16.6 GB
(baseline), 22.9 GB (one L2), run 3 estimated ~35 GB.

**If run 3 runs out of memory:** all four must use the same batch, so re-run
all four with `--batch-size 4 --grad-accum 8` (still 32 sequences). If run 3
is unstable (many `warning: non-finite` lines), set `"addon_shared": false` in
`configs/v2_sandbox_multi.json` (three separate L2 copies, +58.8M parameters)
and re-run only run 3 — say so next to its result.

When all four have finished:

```bash
python -m ekdant.gen_eval compare runs/v22_baseline runs/v22_mid runs/v22_multi runs/v22_end
```

It compares every L2 run with the baseline (A6). Copy the four
`runs/v22_*/results.json` files and the four logs back for the thesis.

**Settings (all runs):** backbone LR 5e-5, L2 matrices 1e-4 (`--addon-lr`,
T34), every gate 1e-3 (`--gate-lr`, T35), cosine per phase with 200 warm-up
steps, effective batch 32 sequences, bf16 on A100, gradient clip 1.0 applied
**separately** to the backbone and the add-on (so the add-on can never shrink
the backbone's update), sequences ≤ 1,024 tokens, answers ≤ 512 tokens.

**Earlier experiments, for reference:** the v2.0 tie (L2 at the end, gate LR
1e-4, 1 epoch) is A7a; v2.1 (L2 after layer 20, 2 epochs) is A7c.

## A3b. Next experiments (after A3): confirm the win, then one lever at a time

All runs use the same flags as A3 (1 PubMedQA epoch). Runs 1–2 repeat the
winning comparison with a **new seed**; runs 3–6 each change **one thing**
against the end placement and are compared with the existing seed-17 baseline
`runs/v22_baseline` (so no new baseline is needed for them). One GPU,
one after another — about 9–10 hours for all six; run 2 and run 4 are the most
important if you must choose.

```bash
cd ~/work/Ekdant-LLM && git pull && cd 09-Code
python -m pytest -q                          # all tests must pass
nvidia-smi                                   # ONE GPU with >= 30 GB free
export CUDA_VISIBLE_DEVICES=2                # <- the GPU you picked
FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 1 --ft-epochs 3 --min-free-gb 30"
END="--variant sandbox --config configs/v2_sandbox_final.json"

nohup bash -c "
python -m ekdant.train $FLAGS --variant baseline --config configs/v2_baseline.json --seed 23 --out runs/v23_baseline_s23 > v23_baseline_s23.log 2>&1;
python -m ekdant.train $FLAGS $END --seed 23                                         --out runs/v23_end_s23      > v23_end_s23.log      2>&1;
python -m ekdant.train $FLAGS $END --addon-lr 3e-4                                    --out runs/v23_end_lr3e4    > v23_end_lr3e4.log    2>&1;
python -m ekdant.train $FLAGS $END --gate-floor 0.7                                   --out runs/v23_end_floor07  > v23_end_floor07.log  2>&1;
python -m ekdant.train $FLAGS --variant sandbox --config configs/v2_sandbox_final_k8.json     --out runs/v23_end_k8     > v23_end_k8.log     2>&1;
python -m ekdant.train $FLAGS --variant sandbox --config configs/v2_sandbox_final_lowest.json --out runs/v23_end_lowest > v23_end_lowest.log 2>&1
" > v23_all.log 2>&1 &
```

| # | run | what changes | question |
|---|---|---|---|
| 1 | `v23_baseline_s23` | seed 23 | baseline for run 2 |
| 2 | `v23_end_s23` | seed 23 | does the end-placement win repeat with new data order and split? |
| 3 | `v23_end_lr3e4` | L2 learning rate 1e-4 → 3e-4 | does L2 gain more if it learns faster? |
| 4 | `v23_end_floor07` | gate forced open to 0.7 (T39) | does L2 help when it must contribute a lot? |
| 5 | `v23_end_k8` | 8 hypotheses instead of 4 | do more hypotheses help? |
| 6 | `v23_end_lowest` | pick the hypothesis closest to the anchor instead of averaging | is picking better than averaging? |

**Run 4 — the forced-open gate (`--gate-floor 0.7`).** For the first 200
steps the gate behaves as before, so L2 learns first; then a minimum opening
rises smoothly from 0 to 0.7 over 2,000 steps (`--gate-floor-start`,
`--gate-floor-steps`) and is held. The network can still move the gate
between 0.4 and 1.0 around that floor. If L2 starts bursting, the brake fires
and **moves the ramp back** by 200 steps, so an unstable L2 is never pushed
open — the gate opens fast, then slows down instead of exploding. Watch
`|g|=` climb towards 0.7 and `brake=` stay near 1.00. Caution: at the end,
`g = 0.7` replaces 70 % of SmolLM2's final state with L2's, so the loss may
rise at first — that is the experiment.

Compare after all six:

```bash
python -m ekdant.gen_eval compare runs/v23_baseline_s23 runs/v23_end_s23
python -m ekdant.gen_eval compare runs/v22_baseline runs/v22_end runs/v23_end_lr3e4 runs/v23_end_floor07 runs/v23_end_k8 runs/v23_end_lowest
```

Copy the six `results.json` files and logs back for the thesis.

## A3c. The "mix": L2 inside and at the end (T40)

*Superseded by A3d (the same three models with more epochs).*

Three models, identical flags (seed 17, 1 PubMedQA epoch, gate control on):

| # | run | config | L2 |
|---|---|---|---|
| 1 | baseline | `configs/v2_baseline.json` | none |
| 2 | end | `configs/v2_sandbox_final.json` | once, after the final norm |
| 3 | **mix** | `configs/v2_sandbox_mix.json` | **the same L2** after layers 10, 15, 20, 22, 24, 26, 28, 30 **and** at the end — 9 sites, each with its own gate and norms |

The mix shares one set of L2 weights: 19,608,768 parameters (+14,976 over a
single L2). It recomputes L2 in the backward pass instead of storing it
(`"sandbox_checkpoint": true`) — identical numbers (tested), far less memory —
because 9 stored sites would need ~70 GB. Expect it to take **~3–3.5 h** (L2
runs 9 times per token).

**Runs 1 and 2 already exist** with exactly these settings (`runs/v22_baseline`,
`runs/v22_end`, A3/A7d) — the quickest path is to run only the mix:

```bash
cd ~/work/Ekdant-LLM && git pull && cd 09-Code
python -m pytest -q                          # all tests must pass
nvidia-smi                                   # ONE GPU with >= 30 GB free
export CUDA_VISIBLE_DEVICES=2
FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 1 --ft-epochs 3 --min-free-gb 30"
nohup python -m ekdant.train $FLAGS --variant sandbox --config configs/v2_sandbox_mix.json --out runs/v24_mix > v24_mix.log 2>&1 &

# afterwards:
python -m ekdant.gen_eval compare runs/v22_baseline runs/v22_end runs/v24_mix
```

**Or all three fresh** (~6 h, one after another):

```bash
nohup bash -c "
python -m ekdant.train $FLAGS --variant baseline --config configs/v2_baseline.json     --out runs/v24_baseline > v24_baseline.log 2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_final.json --out runs/v24_end      > v24_end.log      2>&1;
python -m ekdant.train $FLAGS --variant sandbox  --config configs/v2_sandbox_mix.json   --out runs/v24_mix      > v24_mix.log      2>&1
" > v24_all.log 2>&1 &
python -m ekdant.gen_eval compare runs/v24_baseline runs/v24_end runs/v24_mix
```

Watch `|g|=` (the mean of the 9 gates) and `brake=`; `results.json`
`test.gate_per_site` lists the 9 gates in order (layers 10 … 30, then the end).

## A4. Watching a run

```
variant=sandbox addon=sandbox backbone=HuggingFaceTB/SmolLM2-135M params total=… addon=19.59M device=cuda:0
rows pubmedqa=…(-dropped) pubmedqa_held=2000(-0) medquad_ft=… medquad_val=… medquad_bench=…
zero-shot ppl pubmedqa_held=… medquad_val=… medquad_bench=…
[pubmedqa] ep0 step 50/… loss=… lr=… gnorm=<backbone>/<add-on> |g|=… l2=… brake=1.00 eta=…
[pubmedqa] ep0 25% val pubmedqa_held ppl=…
[medquad_ft] ep0 done … val medquad_val ppl=…
[medquad_ft] ep0 new best medquad_val -> best.pt
final benchmark uses best.pt (medquad_val nll …)
[test medquad_bench] ppl=… rouge_l=… token_f1=… n=…
[test pubmedqa_held] ppl=… rouge_l=… token_f1=…
wrote runs/v2_sandbox/results.json
```

- **ppl** (perplexity) — how surprised the model is by the reference answer;
  **lower is better**. It is the main number.
- **l2** (sandbox arm only) — typical size (RMS) of L2's output before its
  normalisation. It should stay roughly flat (order 1–10). A steady climb by
  ×2 every few hundred steps is the drift behind T32–T34 — report it.
- **brake** (sandbox arm) — the L2 learning-rate factor of the brake (1.00 =
  no braking). A `brake: L2 burst at step …` line means the brake fired (gates
  ×0.9, L2 LRs halved). A few are fine; many in a row means L2 is unstable —
  report it.
- **|g|** — mean size of the add-on's gate. 0 means the model is not using the
  add-on at all; watch whether it grows.
- `(-dropped)` — rows that could not fit 1,024 tokens even with the abstract
  trimmed (expected to be 0 or near 0).

## A5. Resume

Re-run the exact same command with `--resume` added. `last.pt` is written at
every phase-1 validation point (each 25 % of the epoch) and every epoch end,
and stores the optimiser state and the position inside the epoch, so a resumed
run continues **exactly** where it stopped (tested). The checkpoint also stores
the flags that fix the data order and schedule; a resume with a different
`--variant`, `--batch-size`, `--grad-accum`, LR, epochs, `--eval-fraction`,
`--limit` or `--seed` is refused. A fresh run refuses to start in a folder that
already has `last.pt`. Wall-clock in `cost` adds up across resumes.

`--eval-only` (with an existing `--out`) skips training and re-runs the final
protocol.

## A6. Compare runs

```bash
python -m ekdant.gen_eval compare runs/v2_baseline runs/v2_sandbox
```

It prints, per arm, the benchmark perplexity, ROUGE-L, token F1, gate size and
PubMedQA held-out perplexity. Then, for the sandbox against the baseline, it gives
the **per-token NLL difference with a 95 % confidence interval** (a paired
bootstrap that resamples whole questions, weighted by answer tokens exactly
like the headline perplexity). **Negative = better than the
baseline. A difference is real only if the interval does not contain 0;
otherwise report a tie.**

How to read it:

| outcome | meaning |
|---|---|
| interval entirely below 0 | the sandbox arm predicts the reference answers better — but it also has 19.59M more parameters, so this does not by itself show the debate mechanism is the cause |
| interval entirely above 0 | the sandbox arm is worse |
| interval contains 0 | a tie — report it (the thesis pre-committed to this) |

## A7. `results.json` (v2)

| key | what |
|---|---|
| `zero_shot.{pubmedqa_held,medquad_val,medquad_bench}` | untrained scores (`nll`, `ppl`, `tokens`, `n`); written only by a fresh run, never from loaded weights |
| `history[]` | every validation point: stage, epoch, fraction of epoch, `val`, train loss, gate size |
| `phase1` | PubMedQA held-out + MedQuAD val after phase 1 |
| `selected` | which checkpoint ran the benchmark, and both validation NLLs |
| `test.medquad_bench` | **the benchmark**: `ppl`, `by_type` (per question type), `rouge_l`, `token_f1` (1,000 generated answers), per-row lists for the bootstrap |
| `test.pubmedqa_held` | retention of phase 1 after the fine-tune (+ ROUGE-L/F1 on 500 rows) |
| `test.gate_magnitude` | final mean `|g|` |
| `cost` | params, tokens seen, wall-clock, peak GPU memory — report these with every number |

`addon` inside an evaluation holds the sandbox diagnostics (`anchor_energy`,
`trajectory_variance`, `spread_ratio`, over answer tokens) for the sandbox arm.

## A7a. Paired results (2026-10-24, the thesis numbers)

Both arms with the A3 flags, seed 17, one A100 each (`v2_baseline-part3.log`,
`v2_sandbox-part3.log`; results in `09-Code/Runs/`, git-ignored).

| | baseline | sandbox |
|---|---|---|
| params | 134,515,008 | 154,109,952 (+19,594,944) |
| MedQuAD benchmark ppl (8,258 answers) | 3.7817 | **3.7795** |
| MedQuAD validation ppl | 3.958 | 3.956 |
| PubMedQA held-out ppl, after phase 1 → final | 7.536 → 8.191 | 7.534 → 8.188 |
| ROUGE-L / token F1 (1,000 generated) | 0.3127 / 0.3770 | 0.3137 / 0.3789 |
| gate `|g|` / L2 size `l2` | — | 0.0054 / 0.7–5.2 |
| wall-clock, peak GPU memory | 4,612 s, 16.6 GB | 5,807 s (+26 %), 22.8 GB |
| skipped steps | 0 | 0 |

`python -m ekdant.gen_eval compare`: per-token NLL **−0.0006, 95 % CI
[−0.0007, −0.0005]** — the interval excludes 0, but the effect is 0.06 % of
perplexity. Three same-seed baseline runs gave benchmark ppl 3.782 each and
ROUGE-L 0.3127–0.3155 / F1 0.3770–0.3797, so the generation scores tie.
**Verdict: a practical tie;** the network kept L2's gate almost closed.

## A7e. v25 results: 3 PubMedQA + 5 MedQuAD epochs (2026-10-29)

Seed 17, L2 optional (gates free). MedQuAD benchmark: 8,258 questions, 1,000
generated answers. NLL difference vs baseline, paired bootstrap 95 % CI.

| run | bench ppl | ROUGE-L | F1 | PubMedQA held ppl | final `|g|` | Δ NLL vs baseline [95 % CI] | time |
|---|---|---|---|---|---|---|---|
| `v25_baseline` | 3.925 | 0.3087 | 0.3762 | 8.916 | — | — | ~2 h 45 min |
| `v25_end` | 3.921 | **0.3137** | **0.3810** | 8.913 | 0.0746 | −0.0011 [−0.0015, −0.0008] | ~3 h 50 min |
| `v25_mix` | **3.918** | 0.3092 | 0.3762 | **8.887** | 0.1192 | **−0.0019 [−0.0022, −0.0015]** | ~16 h 40 min |

- Both L2 placements beat the baseline (CIs exclude 0); the mix gains most in
  perplexity. ROUGE-L/F1 differences are within same-seed noise (~0.003).
- More epochs hurt every arm: PubMedQA val ppl is best after epoch 1 (7.68)
  and rises in epochs 2–3 (7.78); benchmark ppl 3.93 vs 3.78 at 1 + 3 epochs.
- The gates were free: both first closed (end 0.023 → 0.015, mix 0.062 →
  0.052 by step 6,500), then opened as L2 became useful, fastest in MedQuAD.
- Stable: no skipped steps; 2 brake events in the mix (steps 6,040/6,043, l2 3.1).

## A7d. v2.2 four-run results (2026-10-27, the current thesis numbers)

`v22_*.log`; identical flags; no skipped steps, **no brake events**.

| run (seed 17, 1 epoch, gate control) | benchmark ppl | per-token NLL vs baseline (95 % CI) | ROUGE-L / F1 | PubMedQA ppl | \|g\| | peak mem |
|---|---|---|---|---|---|---|
| baseline | 3.7826 | — | 0.3144 / 0.3788 | 8.205 | — | 16.6 GB |
| L2 after layer 20 | 3.7815 | −0.0003 [−0.0003, −0.0002] | 0.3152 / 0.3792 | 8.208 | 0.063 | 22.9 GB |
| L2 at layers 10/20/30 (shared) | 3.7792 | −0.0009 [−0.0010, −0.0008] | 0.3171 / 0.3820 | 8.196 | 0.071 (0.037 / 0.050 / 0.127) | 34.9 GB |
| **L2 at the end** | **3.7761** | **−0.0017 [−0.0019, −0.0016]** | **0.3202 / 0.3852** | **8.176** | 0.028 | 22.8 GB |

Every L2 run beats the baseline beyond its interval; **L2 at the end wins on every
measure** (−0.17 % perplexity; ROUGE-L / F1 +0.006, above the 0.3127–0.3155 /
0.3770–0.3797 spread of three same-seed baselines; less PubMedQA forgetting). The
three-depth run opened its deepest gate most. Wall-clock: baseline 4,613 s, end
5,274 s (+14 %); the two inner runs shared their GPU with other jobs, so their
times (≈ 4 h) are not comparable. One seed — confirm with `--seed 23`.

## A7c. v2.1 results (L2 after layer 20, 2 PubMedQA epochs, 2026-10-24)

`v21_baseline.log`, `v21_sandbox.log`; seed 17, gate LR 1e-3.

| | baseline | L2 after layer 20 |
|---|---|---|
| MedQuAD benchmark ppl | 3.914 | **3.913** |
| per-token NLL vs baseline (95 % CI) | — | −0.0002 [−0.0003, −0.0001] |
| ROUGE-L / token F1 | 0.3111 / 0.3726 | 0.3065 / 0.3682 |
| PubMedQA held-out ppl, after phase 1 → final | 7.593 → 8.340 | 7.592 → 8.333 |
| gate `|g|` / L2 size `l2` | — | 0.029 (still rising) / 0.3–16 |
| wall-clock, peak memory | 8,053 s, 16.6 GB | 10,491 s (+30 %), 22.9 GB |

A tie again, and the **second PubMedQA epoch hurt both** (benchmark 3.914 vs
3.782 with 1 epoch; MedQuAD validation 4.106 vs 3.958) — hence the 1-epoch
recipe of A3.

## A7b. Reference numbers (first baseline run, 2026-10-23)

One seed, one shared A100 (`CUDA_VISIBLE_DEVICES=1`), the flags in A3. Phase 1
took 1 h 02 m 48 s; each MedQuAD epoch about 2 min 10 s; about 1 h 15 min end
to end including generation.

| measured on | before training | after PubMedQA | final |
|---|---|---|---|
| MedQuAD benchmark half (8,258 answers) — **ppl** | 7.578 | — | **3.782** |
| MedQuAD validation questions — ppl | 7.508 | — | 3.958 |
| PubMedQA held-out (2,000) — ppl | 9.703 | 7.534 | 8.192 |

Phase-1 validation curve (PubMedQA held-out ppl at 25/50/75/100 %): 7.811,
7.660, 7.567, 7.534. MedQuAD validation per epoch: 4.314, 3.986, 3.958 (all
three new bests). Generated answers (1,000 benchmark questions): ROUGE-L
0.316, token F1 0.380. Training loss stayed near 2.0 with backbone gradient
norm 1.4–2.0 throughout; no skipped steps.

The first sandbox run diverged at about step 400 (T32) and is being repeated
with the fixed code. Re-run both arms into new folders (for example
`runs/v2b_baseline`, `runs/v2b_sandbox`) so this run stays as a reference for
run-to-run variation.

## A8. v2 troubleshooting

| symptom | fix |
|---|---|
| `error: GPU n has only x GB free` | other users' jobs fill that GPU — pick another with `CUDA_VISIBLE_DEVICES` |
| `error: CUDA is not available` | the GPU/driver is broken in this container; fix it (the run will not silently use the CPU). `--gpu cpu` only for smoke runs |
| `error: this run was started with different settings` | resume with the original flags (the message lists them) |
| `CUDA out of memory` mid-run | both arms to `--batch-size 4 --grad-accum 8`, then `--resume` |
| `expected m1 and m2 to have the same dtype … BFloat16` | old code: transformers 5 loads SmolLM2 in bf16; fixed (the backbone is cast to float32) — `git pull` |
| sandbox log: `gnorm` jumps to huge values or `inf`, loss in the thousands, then `OverflowError: math range error` | old code (T32, fixed): the L2 output reached the logits unnormalised. `git pull`, delete `runs/v2_sandbox`, re-run the sandbox arm only (the baseline is unaffected) |
| sandbox stops with `20 consecutive non-finite steps` around step 200 (gnorm 10¹¹+) | old code (T33, fixed): the debate rounds compounded their size; `git pull` (debate pre-norm), delete that run folder, re-run the sandbox arm |
| many `brake: L2 burst` lines and `brake=` falling towards 0.05 | L2 keeps spiking; the brake holds it, but the gate cannot open. Report the log; a gentler `--brake-spike 4` or a lower `--gate-boost` are the knobs (T37) |
| `warning: non-finite loss/gradient at step … - update skipped` | the guard dropped one bad update (T32). A few are harmless; after `--max-bad-steps` (20) in a row the run stops — report it |
| `|g|` stays ≈ 0 for the whole run | the model chose not to use the add-on; that is a result — report it |
| `ModuleNotFoundError: transformers` | `pip install -e .` again (v2 added the dependency) |
| `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files` on the first v2 run | old code (T31, fixed): the one-time download for an empty cache was blocked inside `transformers` 4.x. `git pull`; on old code, put `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0` in front of the first run |
| tests fail with `LlamaForCausalLM requires the PyTorch library`; log says `PyTorch >= 2.6 is required but found 2.2.2` | `transformers` 5.x was installed; it needs PyTorch ≥ 2.6. `git pull && pip install -e .` installs the pinned `transformers<5` (T30) |

---

# Part B — v1: from-scratch MCQ pilot (kept for reproducibility)

Every v1 command below still works unchanged (`--task mcq` is the default).

## Quick start (v1)

```bash
python -m ekdant.train --variant baseline --out /tmp/bas --limit 64 --epochs 1 --ft-epochs 1 --batch-size 4 --no-compile
python -m ekdant.train --variant sandbox  --out /tmp/sbx --limit 64 --epochs 1 --ft-epochs 1 --batch-size 4 --no-compile
# real training + benchmark: §3.3 (one model at a time, GPU)
```

---

## 0. What you are running

The code (`09-Code/src/ekdant/`) is **T + L2**: `embed → T (windowed dense
Transformer) → L2 (sandbox) → head`. The thesis's L1 SSM layer is specified
but not implemented, and is not in this package.

The experiment is an **architecture-matched A/B**: two models identical in
every setting except `use_sandbox`.

| variant | total params | embeddings | T (12 blocks) | L2 sandbox | config |
|---|---|---|---|---|---|
| `baseline` | **43.53M** | 16.78M | 26.75M | — | `configs/baseline_50m.json` |
| `sandbox` | **50.09M** | 16.78M | 26.75M | 6.56M | `configs/sandbox_50m.json` |

The embedding counts include all 32,768 allocated rows; the tokenizer learns
26,766 entries, so 6,002 rows (3.07M parameters, identical in both models) are
never used. Any accuracy difference between the two models comes from L2 —
but the sandbox model is also 15 % larger and does more work per token, so
always report it as *architecture-matched, not parameter-matched*.

**The protocol in one picture** (one command runs all of it, in order):

```
MedMCQA train (182,496 rows) ──► phase 1: train
MedMCQA validation (4,170 usable) ──► score after phase 1 · pick the checkpoint · measure forgetting
MedQA train (10,201) ── seed-locked split ──┬─ 60 % (6,106) ──► phase 2: fine-tune
                                            └─ 40 % (4,072) ──► THE BENCHMARK — never trained on, scored once
MedQA test (1,273) ──► second untouched number
```

The model is never trained on both datasets at once, never sees the 40 %
benchmark during training, and never picks a checkpoint with it.

---

## 1. Prerequisites

| need | notes |
|---|---|
| Python 3.10+ | verified on 3.11 and 3.12 |
| PyTorch 2.2+ | CUDA GPU for real runs; CPU is fine for tests and smoke runs. Exact pins and tested versions: Part A1 and `09-Code/requirements.txt` |
| internet on the first run | MedMCQA (~183k rows) and MedQA (~11k rows) download into the Hugging Face cache once; after that everything runs offline |
| disk | a few GB for the dataset cache + ~0.5 GB per run folder |
| GPU memory | modest: batches are padded only to the longest row in the batch (MedMCQA p50/p99 = 34/118 tokens, MedQA 173/364, max 857). A 16 GB T4 runs batch 6 × accumulation 8 comfortably |

Pure PyTorch — no `flash-attn`, `mamba-ssm`, `triton` or `causal-conv1d`.

---

## 2. Install

```bash
cd 09-Code
pip install -e .[dev]                     # package + pytest
python -c "import ekdant; print('ok')"    # must print ok
```

> [!important]
> `pip install -e .` needs the **trailing dot**. Without it pip prints
> `-e option requires 1 argument`, nothing is installed, and training later
> fails with `ModuleNotFoundError: No module named 'ekdant'`.

`pyproject.toml` pins `datasets>=2.19,<5` and `pyarrow>=12,<21`, so the
install works on both numpy 1.x (conda/Jupyter images) and numpy 2.x.

### Linux / Kaggle / Jupyter notes

- Use `pip` and `python` from the **same** environment (on conda images:
  `/opt/conda/bin/pip` and `/opt/conda/bin/python`).
- PyTorch is preinstalled on most Kaggle/Jupyter images; `pip install -e .`
  reuses it.
- Do **not** upgrade numpy on a conda image — it breaks the prebuilt
  pandas/scipy. If an old clone pulled a pyarrow that needs numpy 2:
  `pip install "datasets==3.6.0" "pyarrow<21"`.
- No-install fallback (from `09-Code/`):
  `PYTHONPATH=src python -m ekdant.train ...`

### First run and the dataset cache

Datasets are downloaded once, then always read from the local cache — the
code runs in offline mode by default so a flaky network can never stall
training (T19). On an empty cache it goes online exactly once to download,
then switches back. If a fresh machine fails with
`Couldn't reach 'openlifescienceai/medmcqa' on the Hub (OfflineModeIsEnabled)`,
you are on code from before that fix (T26): `git pull`, or put
`HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0` in front of the first command.

---

## 3. Train

### 3.1 Smoke test first (~2 min per model, CPU is fine)

```bash
python -m ekdant.train --variant baseline --out /tmp/bas --limit 64 --epochs 1 --ft-epochs 1 --batch-size 4 --no-compile
python -m ekdant.train --variant sandbox  --out /tmp/sbx --limit 64 --epochs 1 --ft-epochs 1 --batch-size 4 --no-compile
```

`--limit 64` uses 64 rows per dataset. It checks the whole pipeline — tokenizer,
both phases, checkpoints, the benchmark, `results.json` — in a couple of
minutes. Accuracy will be near chance; that is expected. Smoke numbers are
never results.

### 3.2 The recipe

| setting | value | why |
|---|---|---|
| batch | `--batch-size 6 --grad-accum 8` (effective 48) | fits a 16 GB T4; same effective batch as the 24 × 2 default |
| phase 1 | all of MedMCQA, `--epochs N`, lr 3e-4 cosine, 200 warmup steps | see the warning below for choosing N |
| phase 2 | 60 % of MedQA, `--ft-epochs 10 --ft-lr 5e-5` | only 6,106 rows — a longer or hotter fine-tune memorises them (T21) |
| answer weight | `--answer-weight 10` | the answer letter is 1 token in ~34–173; without this, almost none of the gradient trains the actual decision (T22) |
| seed | `--seed 17` (default) | also fixes the 60/40 MedQA split — identical for both models |

> [!warning] **Choosing `--epochs` (phase 1) — read before a long run (T25).**
> A 100-epoch baseline run reached 33.33 % MedMCQA validation after phase 1 —
> the same as a 3-epoch run (33.31 %) — with training loss 0.58 against
> validation loss 4.61. Phase 1 memorises MedMCQA long before epoch 100. Every
> phase-1 epoch now prints a `monitor medmcqa_val` line: run once, see where
> that curve stops rising, and use roughly that many epochs for both models.
> With the current recipe the curve peaked at epoch 1 (20-epoch run: 27.36 →
> 33.33 → 32.16 → 32.13 %; 2-epoch run: 32.57 → 33.91 %), so **use `--epochs 2`**.

Step counts at effective batch 48: phase 1 = **3,802 optimizer steps per
epoch**; phase 2 = **128 per epoch** (1,280 for 10 epochs). For example
`--epochs 2` gives 7,604 + 1,280 = **8,884** steps; `--epochs 100` gave 381,480.

### 3.3 Full A/B — commands

Run **one model at a time**; each uses the whole GPU. Keep every flag
identical between the two runs (only `--variant` and `--out` change) — any
other difference invalidates the comparison.

```bash
cd 09-Code
FLAGS="--batch-size 6 --grad-accum 8 --epochs 2 --ft-lr 5e-5 --ft-epochs 10 --answer-weight 10"

# 1) baseline first
nohup python -m ekdant.train --variant baseline --out runs/baseline $FLAGS > baseline.log 2>&1 &
tail -f baseline.log                  # Ctrl-C stops watching, not training

# 2) after baseline.log ends with "wrote runs/baseline/results.json":
#    give the sandbox model the identical vocabulary, then start it
mkdir -p runs/sandbox && cp runs/baseline/tokenizer.json runs/sandbox/
nohup python -m ekdant.train --variant sandbox --out runs/sandbox $FLAGS > sandbox.log 2>&1 &
tail -f sandbox.log
```

`--epochs 2` is where the phase-1 validation curve peaked (§3.2, §6.3). Multi-GPU: replace
`python -m` with `torchrun --nproc_per_node=4 -m` (still one model at a
time; `--batch-size` is per GPU, so the effective batch grows with GPUs).

Notes:
- **Use a new `--out` for each new run.** The script refuses to start in a
  folder that already holds `last.pt`, so it can never overwrite a finished
  or paused run by accident. Use `--resume` (§4) to continue one.
- The tokenizer copy is optional — both runs train the same tokenizer from the
  same seeded sample (both logged runs learned 26,766 entries) — but copying
  guarantees an identical vocabulary.

### 3.4 How long it takes (16 GB T4)

Measured on the baseline: **~6.3 optimizer steps/s ≈ 10 min per MedMCQA epoch**
(fp16, ~11.9k tokens/s); one phase-2 epoch ≈ 25 s; each MedMCQA-validation
check adds a few seconds per epoch. The sandbox model does K = 4 trajectories,
2 debate rounds and 8 descent steps per token — **estimated** at ~1.5× the
baseline on a T4 (its first `eta=` line tells you).

**Measured on a shared 80 GB GPU** (2-epoch recipe, batch 6 × 8): baseline
18m42s + 15m08s for the two MedMCQA epochs (~7.9k tokens/s), ~31 s per
fine-tune epoch, ~45 min end to end. Sandbox 23m15s + 18m25s (~6.5k tokens/s,
**~1.2× the baseline**). The first epoch includes `torch.compile` warm-up.

| `--epochs` | optimizer steps | baseline | sandbox (est.) |
|---|---|---|---|
| 3 | ~12.7K | ~35 min | ~55 min |
| 14 | ~54.5K | ~2.5 h | ~3.8 h |
| 40 | ~153K | ~6.8 h | ~10 h |
| 100 | ~381K | ~17 h | ~26 h |

The first `eta=` lines of a run are inflated by `torch.compile` warm-up
(~150 s) and settle within a few hundred steps.

### 3.5 Watching progress

Real lines from a finished baseline run (note: that run used the older
recipe, so its phase 2 shows `100 epochs ... lr 0.0003` instead of the
recipe's `10 epochs = 1280 steps, lr 5e-05`):

```
=== stage medmcqa  n=182496 (epoch 0+) | 3802 opt/epoch x 100 epochs = 380200 steps, lr 0.0003 cosine, warmup 200 ===
[medmcqa] ep0 step 100/380200 (0%) loss=6.8995 lr=1.50e-04 gnorm=4.09 tok/s=1206 eta=165h39m20s
[medmcqa] ep0 done loss=3.7137 gnorm=1.55 lr_end=3.00e-04 tok/s=9631 time=12m25s
[phase1 test medmcqa_val] acc=33.33% loss=4.6092 n=4170  (after MedMCQA training, before MedQA fine-tune)
=== stage medqa_ft  n=6106 (epoch 0+) | 128 opt/epoch x 100 epochs = 12800 steps, lr 0.0003 cosine, warmup 200 ===
[medqa_ft] ep0 done loss=3.4885 gnorm=4.76 lr_end=1.90e-04 tok/s=44108 time=0m24s
[medqa_ft] ep0 eval medmcqa_val acc=31.80% loss=3.7552 n=4170
[medqa_ft] ep0 new best medmcqa_val 31.80% -> best.pt
selection on medmcqa_val: last.pt=27.58%  best.pt=31.80%
final benchmark uses best.pt (medmcqa_val 31.80%)
[test medqa_bench] acc=26.89% loss=2.7601 n=4072
[test medqa_test] acc=27.02% loss=2.7900 n=1273
```

Two line types were added after that run (format shown from a smoke run; its
numbers mean nothing):

```
[medmcqa] ep0 sandbox anchor_energy=247.9 trajectory_variance=0.2805 spread_ratio=0.281
[medmcqa] ep0 monitor medmcqa_val acc=32.81% loss=8.9922 n=64
```

| line / field | meaning — and what healthy looks like |
|---|---|
| `step X/Y (%)` | optimizer steps into the current phase — your "how far along" number |
| `loss` | running average of the training loss; should fall |
| `lr` | rises during the 200 warm-up steps, then decays (cosine) to ~0 at the end of each phase |
| `gnorm` | gradient norm before clipping at 1.0; ~0.5–3 is normal, a collapse toward 0 means training stalled |
| `tok/s`, `eta` | throughput and time left in the **phase** |
| `epN done …` | epoch summary: mean training loss, mean gnorm, lr at epoch end, wall time |
| `epN monitor medmcqa_val …` | phase 1 only: validation accuracy each epoch, **monitoring only**. When it stops rising, more phase-1 epochs are memorisation (T25) |
| `epN sandbox …` | sandbox model only: average `anchor_energy`, `trajectory_variance`, `spread_ratio` (ρ) over real tokens. ρ stuck below 0.05 means the trajectories collapsed into one and L2 is doing nothing |
| `[phase1 test medmcqa_val]` | MedMCQA validation after phase 1 — the "before" number for forgetting |
| `epN eval medmcqa_val …` / `new best … -> best.pt` | phase 2: validation each epoch; a new best saves `best.pt` |
| `selection on medmcqa_val …` | the final pick: whichever of `best.pt` / `last.pt` scores higher on validation (ties keep `best.pt`) |
| `[test medqa_bench]` | **the benchmark** — appears once, at the very end |

### 3.6 What one command does, step by step

1. Loads the config (`--config`, else `configs/<variant>_50m.json`; `baseline`
   always forces `use_sandbox=False`).
2. Trains a BPE tokenizer on 20,000 MedMCQA prompts (budget 32,768; it learns
   26,766) and saves `<out>/tokenizer.json` — or reuses one already there.
3. **Phase 1:** trains on all 182,496 labelled MedMCQA rows for `--epochs`
   epochs (the 326 rows without an answer are dropped), checking MedMCQA
   validation every `--eval-every` epochs for monitoring, then records the
   phase-1 result in `results.json → phase1`.
4. **Phase 2:** fine-tunes on the 60 % MedQA partition (6,106 rows) for
   `--ft-epochs` epochs at `--ft-lr`, with its own warm-up and cosine. After
   each epoch it saves `last.pt`, scores MedMCQA validation, and saves `best.pt`
   on a new best.
5. **Final protocol:** picks the better of `best.pt` / `last.pt` on MedMCQA
   validation, then scores the 40 % MedQA benchmark (once) and the official
   MedQA test, and writes everything to `results.json`.

### 3.7 All flags

| flag | default | meaning |
|---|---|---|
| `--variant` | required | `baseline` or `sandbox` |
| `--out` | required | run folder (checkpoints, tokenizer, results) — new folder per run |
| `--config` | `configs/<variant>_50m.json` | custom model config JSON |
| `--batch-size` | 24 | per GPU |
| `--grad-accum` | 2 | effective batch = batch × accum × number of GPUs |
| `--lr` | 3e-4 | phase-1 peak learning rate (cosine) |
| `--warmup` | 200 | warm-up optimizer steps, per phase |
| `--epochs` | 3 | phase-1 (MedMCQA) epochs |
| `--ft-lr` | = `--lr` | phase-2 (MedQA) peak learning rate — use `5e-5` |
| `--ft-epochs` | = `--epochs` | phase-2 epochs — use `10` |
| `--answer-weight` | 0.0 | extra loss on the answer letter — use `10` (`0` = plain language-model loss) |
| `--seed` | 17 | randomness **and** the 60/40 MedQA split — keep identical across models |
| `--eval-every` | 1 | score MedMCQA validation every N epochs (phase 1: monitoring; phase 2: also `best.pt`); `0` = only at the end |
| `--log-every` | 100 | progress line every N optimizer steps (`0` = epoch summaries only) |
| `--limit` | none | rows per dataset — smoke tests only |
| `--no-compile` | off | skip `torch.compile` (use on CPU, or if compile crashes) |
| `--resume` | off | continue an interrupted run (§4) |
| `--eval-only` | off | no training; re-run the final protocol on a finished run (§5.2) |

---

## 4. Resuming an interrupted run

Re-run the **exact same command** with `--resume` added:

```bash
nohup python -m ekdant.train --variant sandbox --out runs/sandbox $FLAGS --resume >> sandbox.log 2>&1 &
```

A checkpoint (`last.pt`) is written after every epoch, so at most one epoch is
lost. On resume:

| restored | from |
|---|---|
| model weights, finished (phase, epoch) pairs — those are skipped | `last.pt` |
| learning-rate position inside the current phase, epoch counter, history, best validation score | `last.pt` |
| tokenizer (never retrained) and the exact model config | `<out>/tokenizer.json`, `<out>/config.json` |
| the phase-1 result | `<out>/results.json` |

Not restored: the optimizer's internal state (expect a small loss bump on the
first resumed batch) and the shuffle order (rows after the break may come in a
different order — the 60/40 split itself never changes). Keep `--epochs`,
`--ft-epochs`, `--ft-lr` and `--answer-weight` the same as the original run;
changing them shifts which epochs count as finished.

---

## 5. Testing

### 5.1 Unit suite (CPU, ~10 s)

```bash
python -m pytest            # 201 tests
```

Covers: parameter accounting and config consistency; layer T (causal + window
mask, no future leakage, GQA, RoPE); the sandbox (K trajectories, debate,
descent moves the readout, collapse floor, per-position `lowest_energy`
readout, padding-free diagnostics); end-to-end loss decrease for both models;
the data layer (both dataset schemas, rejection rules, offline/online toggle);
the evaluator (answer position, sample-weighted loss); the learning-rate
schedule and gradient-accumulation flush; and the seed-locked 60/40 split.

### 5.2 Re-score a finished run (no training)

```bash
python -m ekdant.train --variant sandbox --out runs/sandbox --eval-only --no-compile
```

Needs `last.pt` and `tokenizer.json` in `--out`. Runs the same final protocol
as training (pick `best.pt` vs `last.pt` on validation → benchmark) and
rewrites `results.json`, keeping the recorded phase-1 result. Useful for
re-scoring on another machine (CPU works, just slower).

---

## 6. Benchmarking

### 6.1 What is measured

**Letter accuracy:** at the position right before the answer letter, the
model's scores for the four letter tokens (A–D) are compared and the highest
one is its answer. No free-form generation.

**Chance is not 25 %.** The answer letters are not balanced, so a model that always outputs the most common letter scores the *majority rate*: MedMCQA validation **32.11 %** (always A), `medqa_bench` **25.69 %** (always B), `medqa_test` **27.73 %** (always A). Compare every accuracy with these, and check `pred_counts` — a model piled on one letter has learned only the label prior (T27).

Every evaluation in `results.json` also stores `pred_counts` (how often the
model chose each letter), `gold_counts` and `majority_accuracy`, and the log
line ends with `pred A/B/C/D=… majority=…%`.

| key in `results.json` | rows | role |
|---|---|---|
| `test.medqa_bench` | 4,072 | **the benchmark — the number you quote.** Never trained on, never used to pick a checkpoint, scored once |
| `test.medqa_test` | 1,273 | official MedQA test — a second clean number |
| `phase1.medmcqa_val` → `test.medmcqa_val` | 4,170 | MedMCQA before → after the MedQA fine-tune: how much it forgot. Report the change, not the final value alone (validation also chose the checkpoint) |
| `selected` | — | which checkpoint (`best.pt` / `last.pt`) produced the benchmark numbers |

MedMCQA's public **test** split has no answers (`cop = -1` on every row) and is
never scored (T18).

### 6.2 Compare the two models

```bash
python - <<'PY'
import json
for v in ("baseline", "sandbox"):
    r = json.load(open(f"runs/{v}/results.json"))
    t, s = r["test"], r["selected"]
    p1 = r.get("phase1", {}).get("medmcqa_val", {}).get("accuracy")
    print(f"{v:9s} params={r['params']/1e6:.2f}M ckpt={s['checkpoint']}  "
          f"medqa_bench={t['medqa_bench']['accuracy']*100:.2f}%  "
          f"medqa_test={t['medqa_test']['accuracy']*100:.2f}%  "
          f"medmcqa_val {(p1 or 0)*100:.2f}% -> {t['medmcqa_val']['accuracy']*100:.2f}%")
PY
```

**Is the difference real?** With one seed and ~4,000 benchmark rows, the
noise on an accuracy is about ±0.7 points. A gap smaller than ~2 points is a
tie — report it as one. A tie is a publishable result for this thesis.

### 6.3 Reference numbers (what a real run looks like)

**2-epoch baseline, current recipe** (`--epochs 2 --ft-lr 5e-5 --ft-epochs 10
--answer-weight 10`, batch 6 × 8, one seed, shared 80 GB GPU): MedMCQA
validation 32.57 % after epoch 0 and **33.91 %** after phase 1 (majority 32.11 %);
best fine-tune checkpoint 34.39 % (fine-tune epoch 1); `medqa_bench` **25.44 %**
(majority 25.69 %); `medqa_test` **29.62 %** (majority 27.73 %). Only the MedQA
test number clears its majority rate, by 1.9 points (noise ±1.3). The sandbox
run with the same flags reached 32.23 % after phase 1, then stopped with a CUDA
out-of-memory error at the start of phase 2 because other processes held the
shared GPU. It was not re-run: v1 was superseded by v2 (Part A, T28).

A 20-epoch baseline with the same recipe was stopped early: its monitor line
peaked at epoch 1 (33.33 %) and fell after, while training loss kept dropping —
the reason the recipe now uses `--epochs 2`.

**3-epoch A/B** (T4, batch 6 × 8, one seed). It predates the 60/40 protocol —
it fine-tuned on all of MedQA and has no `medqa_bench` — so never compare it
with a new `medqa_bench`:

| model | params | MedMCQA val (4,170) | MedQA test (1,273) |
|---|---|---|---|
| baseline | 43.53M | 33.31 % | 25.69 % |
| sandbox | 50.09M | 32.01 % | 26.08 % |

A tie: the MedMCQA gap (1.3 points) is within noise, and MedQA sat at chance
because its fine-tune was only ~638 optimizer steps.

**100-epoch baseline, older recipe** (no `--ft-lr`/`--ft-epochs`, so phase 2
ran 100 epochs at 3e-4): MedMCQA validation **33.33 %** after phase 1
(training loss 0.58 vs validation loss 4.61), selected checkpoint 31.80 %,
`medqa_bench` **26.89 %**, `medqa_test` **27.02 %**. Ninety-seven extra
phase-1 epochs bought nothing over the 3-epoch run (T25).

**40-epoch pilot** (older still: fine-tuned on all 10,178 MedQA rows and
watched the MedQA test every epoch): MedQA test 26.79 % after phase 1, best
28.44 % at fine-tune epoch 4; fine-tune training loss fell to 0.05 while
MedMCQA validation dropped from 34.75 % to 28.80 % — the memorisation and
forgetting that the `--ft-*` and `--answer-weight` flags address (T21/T22).

### 6.4 What to report with every number (cost rule C8)

Total parameters; training tokens × K (the sandbox runs K = 4 trajectories,
2 debate rounds and 8 descent steps per token); estimated FLOPs; wall-clock;
and for the sandbox model the `spread_ratio` (ρ) trace from
`history[*].sandbox`. Always state that the comparison is architecture-matched,
not parameter-matched (+6.56M parameters and roughly 2–3× the per-token compute
for the sandbox model). `--limit` runs are never results.

---

## 7. Outputs (per run folder)

```
runs/<variant>/
  tokenizer.json   BPE vocabulary (reused on resume)
  config.json      exact model config of this run (used on resume)
  last.pt          checkpoint after every epoch — the resume point
  best.pt          checkpoint at the best phase-2 validation score
  results.json     variant, use_sandbox, params,
                   history   one entry per epoch: stage, epoch, loss, lr, gnorm,
                             test{medmcqa_val} when scored,
                             sandbox{anchor_energy, trajectory_variance, spread_ratio} (sandbox model),
                   phase1    {medmcqa_val}  — after phase 1,
                   test      {medmcqa_val, medqa_bench, medqa_test} — each {loss, accuracy, n[, sandbox]},
                   selected  {checkpoint, selected_on, best_val_accuracy, last_val_accuracy}
```

`results.json` is rewritten during training, so a crash never loses the
numbers already measured. The benchmark belongs to the checkpoint named in
`selected.checkpoint`.

---

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'ekdant'` | install missed the trailing dot: `pip install -e .` from inside `09-Code/` |
| `Couldn't reach 'openlifescienceai/medmcqa' on the Hub (OfflineModeIsEnabled)` on a fresh machine | old code (T26, fixed): the one-time download for an empty cache was blocked. `git pull`, or run the first command once with `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0` in front |
| first run is slow | the datasets are downloading into the cache — one time only |
| run refuses to start: `last.pt exists but --resume was not passed` | by design: pick a new `--out`, or add `--resume` to continue that run |
| hangs right before an evaluation, no output | old code (T19, fixed): a network check stalled. `git pull` — loading is now offline-first |
| `pyarrow requires NumPy 2.0` | old clone without version caps: `git pull && pip install -e .`, or `pip install "datasets==3.6.0" "pyarrow<21"`; never upgrade numpy on a conda image |
| `torch.amp has no attribute GradScaler` | torch < 2.3 (old Kaggle images): fixed in code — `git pull`. On T4/P100 (no bf16) the run switches to fp16 automatically and says so |
| out of GPU memory | first run `nvidia-smi`: if other processes hold the GPU (the error lists them, e.g. "Process … has 54.82 GiB memory in use" while PyTorch itself holds ~4 GiB), wait or move to a free GPU — the batch is not the problem. Otherwise lower `--batch-size` and raise `--grad-accum` to keep the effective batch at 48. Phase 2 (MedQA, ~5× longer prompts) needs more memory than phase 1 |
| accuracy ≈ the majority rate, `pred_counts` piled on one letter | the model learned only the label prior (T27) — not a 25 %-chance result. Report it against the majority rate |
| `torch.compile` crashes or is slow | add `--no-compile` |
| phase-1 training loss keeps falling but validation stops improving | memorisation (T25) — use fewer `--epochs` |
| fine-tune loss near 0 but benchmark near the majority rate | MedQA memorised (T21) — use `--ft-lr 5e-5 --ft-epochs 10` |
| sandbox `spread_ratio` stuck below 0.05 | the K trajectories collapsed (T8); report it — it is a result. Raising `rho_min` is the first knob to try |
| accuracy exactly 0.00 % on thousands of rows | very old code (T17, fixed): `git pull` |
| "accuracy" equals how often the model says A | very old code scoring the unlabelled MedMCQA test split (T18, fixed): `git pull` |
| answer tokens are lowercase `a b c d` | expected: the tokenizer lower-cases text, and `a` is also the English article. Both models share this |
| `lowest_energy` readout gave batch-dependent outputs | old code (T24, fixed); the default readout is `mean`, so no logged run was affected |
| CPU-only machine | tests, smoke runs and `--eval-only` work (`--no-compile`); real training needs a GPU |

More: [06-Training-Eval/13_Debugging_Guide.md](06-Training-Eval/13_Debugging_Guide.md) ·
full issue log: [07-Management/10_Known_Issues.md](07-Management/10_Known_Issues.md) ·
next steps: [07-Management/11_Future_Work.md](07-Management/11_Future_Work.md).

---

## 9. Checklist for a publishable A/B

- [ ] `python -m pytest` passes on the machine you train on
- [ ] smoke test of both models passes
- [ ] phase-1 epoch count chosen from the `monitor medmcqa_val` curve
- [ ] both models run with **identical** flags (only `--variant` / `--out` differ), same `--seed`
- [ ] sandbox run uses the baseline's `tokenizer.json`
- [ ] both `results.json` files exist; quote `test.medqa_bench`, plus `medqa_test` and the MedMCQA before → after change
- [ ] cost report (§6.4) and the sandbox ρ trace attached
- [ ] stated: one seed, architecture-matched, differences under ~2 points are a tie
