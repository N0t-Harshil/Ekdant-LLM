> How the evaluated build is validated: a CPU unit suite, a smoke run per arm, and a leakage-free held-out evaluation protocol. Model details live in `02_System_Architecture.md`; training in `07_Training_Pipeline.md`.
>
> An earlier design described an 8-phase corpus-validation harness for a 54-dataset pre-training mixture (registry verification, doc scraping, token-distribution and packing checks). That corpus was dropped when the project narrowed to MedMCQA → MedQA, and the harness scripts were never part of this repo.

# Validation System

Three layers, cheapest first. Nothing below needs a GPU except the real runs.

## Layer 1 — Unit suite (CPU, ~10 s)

```bash
cd 09-Code && pip install -e .[dev] && pytest -q      # 93 tests
```

| File | Tests | What it pins down |
|---|---|---|
| `test_model.py` | 26 | forward shapes and aux outputs for both arms, tied head adds no parameters, `use_sandbox` is the only A/B difference, shipped configs consistent and depth-matched, floor penalty zero when healthy / fires when collapsed, gradients reach the backbone, answer-weighted loss (scaling, `0` = legacy, gradient reaches the answer position), config validation (incl. every config-valid `readout` builds a model), loss decreases, no blocked imports |
| `test_sandbox.py` | 21 | shapes for K=1/4, three separately named diagnostics, orthogonal init and spread gains, descent moves the readout while the peer-mean target is a no-op (the T12 regression), per-trajectory gates, spread floor and ρ at init, `lowest_energy` readout is per-position and causal, diagnostics count only real tokens (T24), loss decreases |
| `test_transformer.py` | 14 | causal and windowed mask, future tokens never change the past, RoPE round-trip, GQA head repetition, RMSNorm, SwiGLU, gradients through every block |
| `test_data.py` | 19 | MedMCQA flat schema and 0-indexed `cop`, MedQA dict schema, rejection of short/blank/unknown rows, letter resolution, unlabelled rows dropped, prompt template, letter-id binding, collate, `evaluate` scores the answer position, is sample-weighted, reports sandbox diagnostics only for the sandbox arm, and the offline toggle reaches every flag `datasets` reads (T26) |
| `test_schedule.py` | 10 | warmup + cosine, per-stage LR reset, `--ft-lr` / `--ft-epochs` overrides, resume offsets, trailing grad-accumulation flush leaks no grads (T23), seeded shuffle generator |
| `test_eval_split.py` | 3 | the 60/40 MedQA partition is disjoint, deterministic for a seed, and sized as the protocol constant says |

## Layer 2 — Smoke run per arm (~2 min)

```bash
python -m ekdant.train --variant sandbox  --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
python -m ekdant.train --variant baseline --out /tmp/bas --limit 400 --epochs 1 --batch-size 4
```

Exercises tokenizer training, both phases, checkpointing, `--resume`, and the final benchmark end to end. `--limit` runs are pipeline checks and are **never** publishable results.

## Layer 3 — Held-out evaluation protocol

All scoring is letter-accuracy (argmax over the four letter logits at the answer position; chance 25 %).

| Key in `results.json` | Rows | Role |
|---|---|---|
| `phase1.medmcqa_val` | 4,170 | accuracy after phase 1 (before the MedQA fine-tune) |
| per-epoch history | 4,170 | MedMCQA validation every `--eval-every` epochs — phase 1: monitoring only (T25); phase 2: drives `best.pt` |
| `selected` | — | which checkpoint ran the benchmark: the better of `best.pt` / `last.pt` on MedMCQA validation |
| `test.medqa_bench` | 4,072 | **decision gate** — the 40 % of MedQA train never trained on, scored once |
| `test.medqa_test` | 1,273 | official MedQA test, also untouched — a second clean number |
| `test.medmcqa_val` | 4,170 | retention after the fine-tune (report `phase1` → final, not the raw final: it also chose the checkpoint) |

Leakage rules baked into the code (T20): the benchmark rows are never used for selection or per-epoch monitoring; selection uses only MedMCQA validation; MedMCQA's unlabelled public test split is never scored (T18); the 60/40 partition is fixed by `--seed`, so both arms see identical train and benchmark rows.

## Reporting rules (cost rule C8)

Publish with every number: total parameters, training tokens × K, estimated FLOPs, wall-clock, and — for the sandbox arm — the `spread_ratio` trace. State that the A/B is architecture-matched, not parameter-matched (+6.56 M, +15 % parameters, extra compute for the sandbox arm). One seed means differences smaller than seed variance (≈ 0.7 points on MedMCQA validation in the pilot) should not be interpreted.

## Sandbox health (what to watch)

The sandbox arm emits `anchor_energy`, `trajectory_variance` and `spread_ratio` (ρ) in `aux`. If ρ pins below `rho_min` (0.05) the trajectories have collapsed and L2 is doing nothing — a result in itself. Per-epoch means are in `history[*].sandbox` and each `evaluate()` result carries a `sandbox` block (T24). RQ3 still needs a per-example agreement/accuracy dump on top of these batch means.

## Open validation work

- Per-example agreement export for the RQ3 calibration plot.
- Depth-/parameter-matched control (no L2) for the RQ2 claim.
- Multi-seed runs to put an error bar on the A/B gap.
- CI that runs `pytest -q` on push.

Connections: prev [07_Training_Pipeline](07_Training_Pipeline.md) · next [13_Debugging_Guide](13_Debugging_Guide.md)
