# Ekdant — sandbox (L2) experiments

> **v2 (current):** pretrained SmolLM2-135M + the L2 sandbox as a gated add-on (or nothing, for the baseline), trained on PubMedQA then half of MedQuAD and benchmarked on the other half by perplexity. Start it with `python -m ekdant.train --task gen …` — full commands in [`../run.md`](../run.md) Part A, design in [`30_Design_v2`](../07-Management/30_Design_v2_Pretrained_Backbone.md).
>
> | v2 arm | added | total params |
> |---|---|---|
> | `baseline` | — | 134.52M |
> | `sandbox` | L2 19.59M (K=4, R=4, 4× MLP, S=8) | 154.11M |
>
> v2 modules: `backbone.py`, `gen_data.py`, `gen_eval.py`, `train_gen.py`; `model.py` adds `V2Config`, `GatedAddOn`, `EkdantLM`.
>
> **Everything below describes v1** (`--task mcq`, the default): the from-scratch 50M MCQ pilot, kept so its results reproduce.

The model is **embed → T → L2**: a windowed dense Transformer backbone with
the L2 sandbox layer on top. The codebase is T + L2 (no L1) — the L1 SSM storage
layer lives in the thesis vault (`04-Model-DeepDive/`) but is not part of
this package.

```
embed -> T (windowed dense) -> L2 (sandbox) -> head
```

## The A/B you are running now

Two models, **identical architecture**, differing **only** in L2:

| variant | total | embed | T (depth) | L2 |
|---|---|---|---|---|
| `sandbox` | **50.09M** | 16.78M | 26.75M (N_T=**12**) | 6.56M |
| `baseline`| **43.53M** | 16.78M | 26.75M (N_T=**12**) | 0 |

Every knob — width, depth, heads, window, FFN, vocab — is the same. The
baseline is exactly the sandbox model with `use_sandbox: false`; the parameter
delta is exactly L2's 6.56M (13%). The experiment isolates the sandbox layer
alone.

**Trade-off to report honestly:** this is *architecture-matched*, not
parameter-matched — the sandbox arm has 15% more parameters *and* runs
K=4 trajectories × R=2 debate rounds × S=8 descent steps on top of its T
blocks. Per the thesis's cost rule (C8), always publish tokens×K and FLOPs
alongside accuracy; a parameter-matched control (spending L2's budget on
baseline depth) is the follow-up.

### Deviations from the thesis parameter table

The thesis now reports the measured numbers below (`thesis.tex`
§Parameters, "as built") and keeps the old closed-form estimate for the full
three-stage design alongside. Measured against that estimate at M scale:

| component | thesis | this code | status |
|---|---|---|---|
| Embeddings (tied) | 16.8M | 16.777M | matches |
| T per block | 2.23M | 2,229,248 | **exact match** to `2.5H²+3HI` |
| T depth | N_T=6 | 12 | intentional — larger backbone for the 50M run |
| L2 | 2.8M | 6.56M | **over — see note** |

> **Open item: L2 is 2.3× the thesis estimate (6.56M vs 2.8M).** The estimate
> budgets ~0.965M for debate across R=2 rounds; the build spends 4.72M because
> it uses full `q,k,v` H×H cross-projections plus an H→2H→H MLP, untied across
> rounds. The thesis now states the measured 6.56M and this explanation; the
> remaining decision is whether to slim/tie L2 in a future build. It does not affect
> the current A/B: the baseline simply omits L2, so the delta between arms is
> the *measured* 6.56M.

## Install

```bash
cd 09-Code
pip install -e .                      # or: pip install -r requirements.txt
```

`requirements.txt` and `pyproject.toml` list the same pins
(`tests/test_requirements.py` checks it): `torch>=2.2`, `datasets>=2.19,<5`,
`pyarrow>=12,<21`, `tokenizers>=0.19`, **`transformers>=4.45,<5`**, `pytest>=8`.
`transformers` 5.x needs PyTorch ≥ 2.6 and silently disables its models on the
training pod's PyTorch 2.2.2 (T30). Tested: PyTorch 2.2.2 + transformers 4.57.6
(pod mirror) and PyTorch 2.9.0.

v2 configs: `configs/v2_baseline.json` and `configs/v2_sandbox.json` (identical
except `addon`). An optional Hugging Face token goes in the git-ignored
`configs/hf_token.txt` (first line), never in a JSON config — see
[`../run.md`](../run.md) A1.

> [!important]
> The trailing **`.` matters** — `pip install -e .` (with the dot) tells pip
> "install the package in this folder". `pip install -e` alone fails with
> `-e option requires 1 argument`, nothing gets installed, and
> `python -m ekdant.train` then dies with
> `ModuleNotFoundError: No module named 'ekdant'`. Verify the install:
> `python -c "import ekdant; print('ok')"`.

No-install fallback (works without pip; from `09-Code/`):

```bash
PYTHONPATH=src python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
```

Requires PyTorch 2.x; real runs need a CUDA GPU (CPU works for tests and
smoke runs). No `mamba-ssm`, `flash-attn`, `triton` or `causal-conv1d` —
every kernel is pure PyTorch, so it runs on any CUDA build.

Full operations runbook (train / resume / test / benchmark, both arms):
[`../run.md`](../run.md).

## Train

Run **one variant at a time** (they use the full node).

```bash
# 4x A100 80GB
torchrun --nproc_per_node=4 -m ekdant.train --variant sandbox  --out runs/sandbox
torchrun --nproc_per_node=4 -m ekdant.train --variant baseline --out runs/baseline
```

Single GPU / debugging:

```bash
python -m ekdant.train --variant sandbox --out runs/sandbox --batch-size 8
```

Both phases run inside one invocation, strictly sequential: train+test on
all of MedMCQA, then fine-tune on 60% of MedQA, then benchmark the
untouched 40%. Each run directory trains its tokenizer once on the same
seeded MedMCQA sample (both logged runs learned 26,766 entries), so the two
variants see the **same vocabulary** — required for a fair comparison. Copy
`tokenizer.json` from the first arm's `--out` to guarantee it.

### Useful flags

| flag | default | notes |
|---|---|---|
| `--batch-size` | 24 | per GPU |
| `--grad-accum` | 2 | effective batch = 24 × 2 × n_gpu |
| `--lr` | 3e-4 | cosine, `--warmup` optimizer steps |
| `--epochs` | 3 | phase 1 (MedMCQA) epochs |
| `--ft-lr` | `--lr` | **phase-2 (MedQA fine-tune) cosine base LR** — anti-overfit knob (e.g. `5e-5`); phase 1 never uses it |
| `--ft-epochs` | `--epochs` | **phase-2 epochs** — cap the fine-tune (e.g. `10`) instead of replaying `--epochs` over 6,106 rows |
| `--answer-weight` | 0.0 | adds `weight × CE(logits[answer_pos−1], answer_id)` to the LM loss — the letter is 1 token in ~34/173, so plain CE dilutes the decision ~97–99% (`10` recommended; `0` = legacy) |
| `--limit` | none | cap rows per dataset for a smoke run |
| `--no-compile` | off | skip `torch.compile` if it misbehaves |
| `--log-every` | 100 | progress line every N optimizer steps: `step X/Y (%) loss lr gnorm tok/s eta` (`0` = summaries only) |
| `--eval-every` | 1 | untouched MedMCQA validation every N epochs — phase 1: monitoring only (history); phase 2: also `best.pt` selection (`0` = final only) |
| `--seed` | 17 | also **locks the 60/40 MedQA partition** — identical for both arms |
| `--resume` | off | continue `<out>/last.pt`; skips finished stage/epochs |
| `--eval-only` | off | no training; re-run the final protocol (selection + 40% benchmark) |

Smoke test first (~2 min):

```bash
python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
```

### Resume / re-evaluate

```bash
python -m ekdant.train --variant sandbox --out runs/sandbox --resume        # after interruption
python -m ekdant.train --variant sandbox --out runs/sandbox --eval-only    # re-score checkpoint
```

`--resume` restores weights, step, epoch/stage position, and history from
`last.pt`; the tokenizer and the exact `config.json` in `--out` are reused.
Optimizer moments restart cold (small loss bump expected), and the
train-shuffle generator restarts (row order after the break may differ —
never the seed-locked 60/40 partition). Keep the same
flags (esp. `--epochs` and `--ft-epochs`/`--ft-lr`/`--answer-weight`, if
used) as the
original run. See [`../run.md`](../run.md) §4.

## Outputs

```
runs/<variant>/
  tokenizer.json     shared vocabulary
  config.json        exact EkdantConfig used
  last.pt            every-epoch checkpoint (resume artifact; incl. history)
  best.pt            peak medmcqa_val during the fine-tune — candidate checkpoint
  results.json       history (loss/lr/gnorm per epoch, MedMCQA-val curve for both phases,
                     sandbox diagnostics for the sandbox arm)
                     + phase1{medmcqa_val} + test{medmcqa_val, medqa_bench, medqa_test} + selected{...}
```

**Protocol (sequential, never jointly trained):** train on all of
MedMCQA → record the **MedMCQA validation** result (`phase1`) → fine-tune
on a seed-locked **60% of MedQA train** → benchmark the complementary
**40%** (`medqa_bench`, never trained on, scored once) + official MedQA
test (`medqa_test`, also untouched). Checkpoint = **better of best.pt vs
last.pt** on the untouched MedMCQA validation (same `--seed` → identical
partition and identical benchmark rows for both arms). Compare the two
`results.json` files:

```bash
python - <<'PY'
import json
for v in ("baseline", "sandbox"):
    r = json.load(open(f"runs/{v}/results.json"))
    t, s = r["test"], r["selected"]
    p1 = r.get("phase1", {}).get("medmcqa_val", {}).get("accuracy")
    print(f"{v:9s} params={r['params']/1e6:.2f}M ckpt={s['checkpoint']} "
          f"bench={t['medqa_bench']['accuracy']*100:.2f}% "
          f"medqa_test={t['medqa_test']['accuracy']*100:.2f}% "
          f"medmcqa {(p1 or 0)*100:.2f}%->{t['medmcqa_val']['accuracy']*100:.2f}%")
PY
```

## What the sandbox logs

The sandbox variant emits three **separately named** diagnostics in `aux`
(only `spread_ratio` enters the loss, through the floor):

- `anchor_energy` — mean ½‖T_k − a_t‖², how far trajectories sit from the
  external anchor
- `trajectory_variance` — diversity across the K trajectories
- `spread_ratio` — ρ, the health ratio; a collapse floor fires when ρ < 0.05

All three are computed over real tokens only and averaged per epoch into `history[*].sandbox` and every `test.*` result in `results.json` (printed as `[stage] epN sandbox …`). Watch `spread_ratio` during training. If it pins at the floor, the trajectories
have converged and the sandbox is doing nothing — that is itself a result.

## Expected memory (per GPU)

The collator pads to **batch max** (`data.py::collate`); measured with
the real tokenizer on all four splits: MedMCQA p50=34 / p99=118 tokens,
MedQA p50=173 / p99=364 (max 857 — **0 of 198k rows exceed the 2048
cap**, so nothing is truncated or dropped). Training therefore uses a
few hundred to a few thousand tokens per micro-batch: measured **6.3
optimizer steps/s at
batch 6 × accum 8 on a 16 GB T4** (fp16 fallback — T4 has no bf16).
Memory only becomes a concern with genuinely long inputs (near 2048) or
much larger batches — then lower `--batch-size` and raise
`--grad-accum` to keep effective batch 48. The sandbox's K=4
trajectories add a `[K,B,T,dim]` scratch tensor — the main activation
cost of the sandbox arm (wall-clock estimated at ≈1.5× the baseline).

## Tests

```bash
pip install -e .[dev]
pytest -q          # 138 tests (model/sandbox/transformer/data/schedule/split), CPU, ~10s
```
