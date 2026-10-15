# Ekdant — 50M sandbox-first training

The model is **embed → T → L2**: a windowed dense Transformer backbone with
the L2 sandbox layer on top. The codebase is L2-only — the L1 SSM storage
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

The thesis table (`thesis.tex` §Parameters) is a *closed-form estimate* that it
itself says "must be confirmed with `estimate_model_size` before any results
table is published." Measured against it at M scale:

| component | thesis | this code | status |
|---|---|---|---|
| Embeddings (tied) | 16.8M | 16.777M | matches |
| T per block | 2.23M | 2,229,248 | **exact match** to `2.5H²+3HI` |
| T depth | N_T=6 | 12 | intentional — larger backbone for the 50M run |
| L2 | 2.8M | 6.56M | **over — see note** |

> **Open item: L2 is 2.3× the thesis estimate (6.56M vs 2.8M).** The thesis
> budgets ~0.965M for debate across R=2 rounds; the build spends 4.72M because
> it uses full `q,k,v` H×H cross-projections plus an H→2H→H MLP, untied across
> rounds. Fixing this means either slimming/tieing L2, or updating the thesis
> table — **decide before any results table is published.** It does not affect
> the current A/B: the baseline simply omits L2, so the delta between arms is
> the *measured* 6.56M.

## Install

```bash
cd 09-Code
pip install -e .
```

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

Requires PyTorch 2.x + CUDA 12.x. No `mamba-ssm`, `flash-attn`, `triton` or
`causal-conv1d` — every kernel is pure PyTorch, so it runs on any CUDA build.

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

Both stages (MedMCQA then MedQA) run inside one invocation. The tokenizer is
trained once on the MedMCQA sample and reused, so the two variants see the
**same vocabulary** — required for a fair comparison.

### Useful flags

| flag | default | notes |
|---|---|---|
| `--batch-size` | 24 | per GPU |
| `--grad-accum` | 2 | effective batch = 24 × 2 × n_gpu |
| `--lr` | 3e-4 | cosine, 200 warmup |
| `--epochs` | 3 | per stage |
| `--limit` | none | cap rows per dataset for a smoke run |
| `--no-compile` | off | skip `torch.compile` if it misbehaves |
| `--resume` | off | continue `<out>/last.pt`; skips finished stage/epochs |
| `--eval-only` | off | no training; evaluate `<out>/last.pt` on both test sets |

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
Optimizer moments restart cold (small loss bump expected). Keep the same
flags (esp. `--epochs`) as the original run. See [`../run.md`](../run.md) §4.

## Outputs

```
runs/<variant>/
  tokenizer.json     shared vocabulary
  config.json        exact EkdantConfig used
  last.pt            checkpoint (model state + config)
  results.json       history + per-dataset test accuracy
```

Compare the two `results.json` files:

```bash
python - <<'PY'
import json
for v in ("baseline", "sandbox"):
    r = json.load(open(f"runs/{v}/results.json"))
    print(f"{v:9s} params={r['params']/1e6:.2f}M "
          f"medmcqa={r['test']['medmcqa']['accuracy']*100:.2f}% "
          f"medqa={r['test']['medqa']['accuracy']*100:.2f}%")
PY
```

## What the sandbox logs

The sandbox variant emits three **separately named** diagnostics in `aux`
(and per-step in the history via the loss):

- `anchor_energy` — mean ‖T_k − a_t‖², how far trajectories sit from the
  external anchor
- `trajectory_variance` — diversity across the K trajectories
- `spread_ratio` — ρ, the health ratio; a collapse floor fires when ρ < 0.05

Watch `spread_ratio` during training. If it pins at the floor, the trajectories
have converged and the sandbox is doing nothing — that is itself a result.

## Expected memory (per A100 80GB)

At batch 24, seq 1024, bf16, `torch.compile`, expect roughly 30–40 GB with
room to raise `--batch-size`. The sandbox's K=4 trajectories add a
`[K,B,T,dim]` scratch tensor — the main activation cost of the sandbox arm.

## Tests

```bash
pip install -e .[dev]
pytest -q          # 70 tests (model/sandbox/transformer/data), CPU, ~10s
```
