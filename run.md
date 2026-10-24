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

## A3. Train both arms (shared 4×A100 pod)

The GPUs are shared with other users. **Check free memory first** — the v1
sandbox run crashed because PyTorch defaults to GPU 0, which was nearly full.

```bash
nvidia-smi                                  # pick GPUs with >= 20 GB free
# optional, once: echo "hf_..." > 09-Code/configs/hf_token.txt   (see below)
cd 09-Code
FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 2 --ft-epochs 3"   # v2.1

CUDA_VISIBLE_DEVICES=1 nohup python -m ekdant.train $FLAGS --variant baseline    --out runs/v2_baseline > v2_baseline.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m ekdant.train $FLAGS --variant sandbox     --out runs/v2_sandbox  > v2_sandbox.log  2>&1 &

tail -f v2_sandbox.log                      # Ctrl-C stops watching, not training
```

Replace `1/2` with whichever GPUs `nvidia-smi` shows as free. One arm per GPU
means both finish in about the time of one. Keep `$FLAGS` identical for both;
only `--variant` and `--out` change.

Each run checks its GPU before loading anything and refuses to start with less
than 20 GB free (`--min-free-gb`), or when CUDA is not available at all — it
never falls back to CPU unless you pass `--gpu cpu` (smoke runs only). Without
`CUDA_VISIBLE_DEVICES`, `--gpu auto` (the default) picks the visible GPU with
the most free memory.

**Settings (both arms):** v2.1 puts L2 after decoder layer 20 (`addon_layer` in the configs, T35); backbone LR 5e-5, L2 matrices 1e-4 (`--addon-lr`; was 1e-3, T34), the gate alone 1e-3 (`--gate-lr`), 2 PubMedQA epochs. The v2.0 tie used `--epochs 1` and L2 after the final norm: reproduce it with `--config configs/v2_sandbox_final.json --epochs 1 --gate-lr 1e-4`.
cosine per phase with 200 warm-up steps, effective batch 32 sequences, bf16 on
A100, gradient clip 1.0 applied **separately** to the backbone and the add-on
(so the add-on can never shrink the backbone's update), sequences ≤ 1,024 tokens, answers ≤ 512 tokens. If
the sandbox arm runs out of memory, move **both** to
`--batch-size 4 --grad-accum 8` (still 32).

**Time:** not yet measured on the A100s. One PubMedQA epoch is ~85M tokens;
the first `eta=` lines of each log give the real figure. The sandbox arm is
slower than the other two (4 debate rounds × 4 trajectories per token).

## A4. Watching a run

```
variant=sandbox addon=sandbox backbone=HuggingFaceTB/SmolLM2-135M params total=… addon=19.59M device=cuda:0
rows pubmedqa=…(-dropped) pubmedqa_held=2000(-0) medquad_ft=… medquad_val=… medquad_bench=…
zero-shot ppl pubmedqa_held=… medquad_val=… medquad_bench=…
[pubmedqa] ep0 step 50/… loss=… lr=… gnorm=<backbone>/<add-on> |g|=… l2=… eta=…
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

## A6. Compare the two arms

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
MedQA train (10,178) ── seed-locked split ──┬─ 60 % (6,106) ──► phase 2: fine-tune
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
python -m pytest            # 153 tests
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
