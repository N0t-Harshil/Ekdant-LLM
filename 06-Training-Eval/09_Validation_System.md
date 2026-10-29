> How the evaluated build is validated: a CPU unit suite, a smoke run per arm, and a leakage-free held-out evaluation protocol. Model details live in `02_System_Architecture.md`; training in `07_Training_Pipeline.md`.
>
> An earlier design described an 8-phase corpus-validation harness for a 54-dataset pre-training mixture (registry verification, doc scraping, token-distribution and packing checks). That corpus was dropped when the project narrowed to MedMCQA → MedQA, and the harness scripts were never part of this repo.

# Validation System

Three layers, cheapest first. Nothing below needs a GPU except the real runs.

## Layer 1 — Unit suite (CPU, ~10 s)

```bash
cd 09-Code && pip install -e .[dev] && pytest -q      # 195 tests
```

| File | Tests | What it pins down |
|---|---|---|
| `test_model.py` | 26 | forward shapes and aux outputs for both arms, tied head adds no parameters, `use_sandbox` is the only A/B difference, shipped configs consistent and depth-matched, floor penalty zero when healthy / fires when collapsed, gradients reach the backbone, answer-weighted loss (scaling, `0` = legacy, gradient reaches the answer position), config validation (incl. every config-valid `readout` builds a model), loss decreases, no blocked imports |
| `test_sandbox.py` | 22 | shapes for K=1/4, three separately named diagnostics, orthogonal init and spread gains, descent moves the readout while the peer-mean target is a no-op (the T12 regression), per-trajectory gates, spread floor and ρ at init, `lowest_energy` readout is per-position and causal, diagnostics count only real tokens (T24), loss decreases, and the v2 debate pre-norm stops size from compounding across rounds (T33) |
| `test_transformer.py` | 14 | causal and windowed mask, future tokens never change the past, RoPE round-trip, GQA head repetition, RMSNorm, SwiGLU, gradients through every block |
| `test_data.py` | 20 | MedMCQA flat schema and 0-indexed `cop`, MedQA dict schema, rejection of short/blank/unknown rows, letter resolution, unlabelled rows dropped, prompt template, letter-id binding, collate, `evaluate` scores the answer position, is sample-weighted, reports sandbox diagnostics only for the sandbox arm, and the offline toggle reaches every flag `datasets` reads (T26), and evaluation reports letter counts + the majority rate (T27) |
| `test_schedule.py` | 10 | warmup + cosine, per-stage LR reset, `--ft-lr` / `--ft-epochs` overrides, resume offsets, trailing grad-accumulation flush leaks no grads (T23), seeded shuffle generator |
| `test_eval_split.py` | 3 | the 60/40 MedQA partition is disjoint, deterministic for a seed, and sized as the protocol constant says |
| `test_addon.py` (v2) | 5 | `mlp_ratio` sets the debate width (19.59M at R=4, 4×; 8.30M at the old R=2, 2×), the add-on is the identity at init, the zero gate still receives gradient, and L2's contribution stays bounded however large its output grows (T32), and it reports its internal size `consensus_rms` (T34) |
| `test_backbone.py` (v2) | 10 | both arms give identical logits at init, loss counts answer tokens only, the add-on's diagnostics/floor mask covers exactly the loss positions, a KV-cache step equals the full forward with the sandbox on, LR groups (backbone 5e-5 / add-on 1e-3, no decay on the gate), config round trip, the two shipped configs differ only in `addon` and hold a token *file* path, never a token; `HF_TOKEN` from the environment wins over the file, the backbone is cast to float32 (T28), the offline toggle reaches `transformers` 4.x's cached flag (T31) |
| `test_gen_data.py` (v2) | 9 | exact prompts, prompt labels −100, answers capped at 512 without EOS, the context (not the answer) is trimmed to fit 1,024, empty rows dropped, the MedQuAD split never puts one question in two sets and is stable, the PubMedQA hold-out, collate padding, rows stored as int32 arrays (host RAM) |
| `test_gen_eval.py` (v2) | 6 | ROUGE-L / token F1 known values and empty cases, `answer_nll` equals the model loss, batched left-padded generation equals one-at-a-time, the bootstrap CI covers a known shift and its token-weighted form matches the headline perplexity, per-type perplexity |
| `test_train_gen.py` (v2) | 22 | GPU pre-flight (`auto` = most free, refuses below `--min-free-gb`, never falls back to CPU unless `--gpu cpu`), zero-shot is never measured on loaded weights, resume refuses changed flags or variant, phase-1 checkpoints never fall on the epoch end, backbone and add-on gradients are clipped separately, wall-clock accumulates across resumes, a full tiny run writes every `results.json` key, a mid-epoch stop + `--resume` ends bit-identical to an uninterrupted run, the fresh-run guard, `--task` dispatch and v1 `--help` (T29), only the `baseline` / `sandbox` arms exist, a config whose `addon` contradicts `--variant` is refused, and a token from the token file is used but never written to any output file, non-finite steps are skipped and counted, and too many in a row stop the run (T32), the add-on LR defaults to 1e-4 and history records L2's size (T34) |
| `test_gate_control.py` (v2.3) | 14 | the capped gate starts at 0, never exceeds the cap and moves like the uncapped one at the start; the boost starts at 10× and fades to 1×; the brake fires on a gradient spike and on a large L2 size, halves the LR factor, recovers slowly, never goes below its floor, ignores non-finite steps, round-trips its state; param groups carry their kind; the shipped configs cap the gate; the gate floor (T39): floor 0 keeps the identity, floor f sets the gate to f bounded in [2f−1, 1], the ramp opens smoothly and holds, backs off on a burst and round-trips its state, is off by default; each lever config changes exactly one key (+ `test_brake_fires_on_forced_spike_and_is_recorded`, `test_gate_floor_run_opens_the_gate`) |
| `test_gate_force.py` (T42) | 9 | forced floor is a hard minimum but can open further; the unforced floor can still be undone; the gate still gets gradient below the floor; the output scale is frozen only when forced; a forced mid site's addition is sized to the stream (half its rms at g = 0.5); a forced mix is the identity until the floor rises; off by default; `--gate-force` without a floor is refused; a forced run keeps every gate ≥ 0.5 and `--eval-only` reproduces it with the checkpoint's floor |
| `test_multisite.py` (v2.2) | 15 | the same L2 at several depths: identity at init, one shared Sandbox (and the separate-copies switch), every site's gate gets gradient, KV-cache step = full forward, all gates in one LR group, per-site gate magnitudes, the shipped multi config; the mix (T40): identity at init, an end site sharing the one Sandbox, the end site acts on the output, KV-cache equality, L2 checkpointing gives identical loss and gradients, `addon_final` needs `addon_layers`, the shipped mix config (+ `test_multisite_run_end_to_end`, `test_mix_run_end_to_end` in `test_train_gen.py`: weights stay shared after checkpointing) |
| `test_midlayer.py` (v2.1) | 8 | L2 after decoder layer k: identity at init, its change at token t reaches later tokens' predictions and never earlier ones while the final placement reaches none, KV-cache step = full forward, bounded against massive activations, layer validation, separate gate LR group and `--gate-lr 1e-3`, shipped configs (layer 20; `v2_sandbox_final.json` = −1) |
| `test_requirements.py` | 2 | `requirements.txt` lists exactly the `pyproject.toml` dependencies, and `transformers` stays pinned `<5` for PyTorch 2.2 (T30) |

## Layer 2 — Smoke run per arm (~2 min)

```bash
python -m ekdant.train --variant sandbox  --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4
python -m ekdant.train --variant baseline --out /tmp/bas --limit 400 --epochs 1 --batch-size 4
```

Exercises tokenizer training, both phases, checkpointing, `--resume`, and the final benchmark end to end. `--limit` runs are pipeline checks and are **never** publishable results.

## Layer 3 — Held-out evaluation protocol

All scoring is letter-accuracy (argmax over the four letter logits at the answer position). The floor is the **majority rate**, not 25 %: MedMCQA val 32.11 %, `medqa_bench` 25.69 %, `medqa_test` 27.73 %. Every evaluation records `pred_counts`, `gold_counts` and `majority_accuracy` (T27).

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
