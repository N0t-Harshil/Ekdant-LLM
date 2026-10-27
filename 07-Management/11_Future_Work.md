# Future Work (thesis-ordered)

## Critical (must finish for thesis defense)

- **Run v2, both arms** — `run.md` Part A: SmolLM2-135M `baseline` / `sandbox`, one per free A100, then `python -m ekdant.gen_eval compare`. Fills the v2 results tables (benchmark perplexity with 95 % CI, ROUGE-L / F1, gate size, cost). A tie is reported as a tie.
- ~~Run v2, both arms~~ **DONE 2026-10-24:** a practical tie (benchmark ppl 3.7817 vs 3.7795, CI excludes 0 but −0.06 %; gate |g| = 0.005) — `run.md` A7a.
- ~~Run v2.1~~ **DONE:** a tie (3.914 vs 3.913); the 2nd PubMedQA epoch hurt both arms.
- ~~v2.2 four-run comparison~~ **DONE 2026-10-27:** L2 at the end wins (−0.0017 per token, ROUGE-L/F1 +0.006).
- **Run `run.md` A3b:** seed-23 repeat of baseline + L2 at the end, then one lever at a time (L2 LR 3e-4, gate floor 0.7 (T39), K=8, `lowest_energy`).
- **Run the v2.2 four-run comparison** (baseline, L2 after layer 20, shared L2 at 10/20/30, L2 at the end; 1 epoch) — `run.md` A3, ~7 h on one GPU.
- **Second seed for both arms** — the CI excludes 0 but covers only question sampling; a different `--seed` (new data order and split) shows whether the 0.0006-nat gain survives.
- **Why the gate stays closed** — log per-step `|g|` and the gate gradient; test a forced contribution (fixed g) as a separate question.
- **Measure v2 time and memory** (done: 4,612 s / 16.6 GB vs 5,807 s / 22.8 GB) on the A100s and write them into `run.md` A3 (currently not measured).
- **Parameter-matched control** — a same-size non-sandbox block (removed from v2 on 2026-10-23) would show whether a gain comes from the mechanism or from +19.59M parameters.
- **Multi-seed v2** if any CI excludes 0 — one seed cannot rule out a lucky split of training order.

**v1 items below are superseded by v2** (T28) and kept for the record:

- **100-epoch campaign, both arms** — `run.md` §3: phase 1 = MedMCQA (100 epochs), phase 2 = 60 % MedQA (`--ft-lr 5e-5 --ft-epochs 10`), `--answer-weight 10`, 381,480 steps (~17 h baseline + ~26 h sandbox on a T4). Fills the Ch. 6 tables (`medqa_bench`, `medqa_test`, MedMCQA phase-1 → final). If the arms tie, report the tie (C24 framing).
- **Choose phase-1 epochs from the validation curve (T25)** — the legacy 100-epoch baseline memorised MedMCQA (train CE 0.58 vs val CE 4.61) and its validation accuracy after 100 epochs (33.33 %) equalled the 3-epoch run (33.31 %). `--eval-every` now records the phase-1 curve; set `--epochs` near its peak before the ~26 h sandbox run. (Sandbox diagnostics are now persisted — T24.)
- **Depth-/parameter-matched control (forced by C24)** — a plain Transformer with the same parameter count (or block-pass budget) and no L2. The shipped A/B is architecture-matched only (50.09M vs 43.53M; the 6.56M delta is L2), so it cannot separate a gain from L2 from a gain from spending the same compute on more untied depth, which is what C24 (arXiv:2603.08082) found winning. C24 also reports **no consistent benefit** for the full Autoregressive TRM, so a tie is publishable.
- **Sandbox ablations** — K = 1/4/8, rounds 0/2/3, steps 0/4/8, `rho_min` ∈ {0.01, 0.05, 0.2}, and anchored descent vs. the discarded peer-mean pull (expect: peer-mean leaves the logits bit-identical).
- **Multi-seed** — one seed (17) cannot put an error bar on the gap; the pilot's noise σ was ≈ 0.7 pt on MedMCQA validation.
- **Resolve the L2 size overage** — measured 6.56M vs. 2.8M in the thesis estimate: slim/tie L2, or restate the estimate. The thesis now reports the measured number and explains the discrepancy; decide before publishing the final tables.

## High (should finish)

- **L1 integration** — add `ssm_storage.py` (selective SSM, three stores) before T, the L1 → T shape assertion (T9), the O(1)-state proof (same `compressed [B, 3·d_state]` shape and flat peak memory for T = 512 / 2048 / 4096). Unlocks RQ1 and the layer-addition ladder (backbone-only → +L2 → +L1+L2). Until then the thesis states L1 as specified-not-built.
- **CI** — `pytest -q` on every push in `09-Code/` (178 tests, CPU, ~10 s).
- **Cost report** — tokens × K, FLOPs, wall-clock per arm (C8); the runner currently logs tok/s only.
- **Rebuild `thesis.pdf`** from the updated `thesis.tex` (`pdflatex` three times); the committed PDF predates the T + L2 alignment.

## Medium (if time)

- Scale-up to a 500M-class model if the 50M campaign separates the arms (needs more than a T4).
- Long-context RoPE extension test (needs L1 or a longer window to be meaningful).

## Low / Nice-to-have (post-thesis)

- Demo UI, REST API, quantization, LoRA, hyper-parameter search, Docker — all deferred. Do not add before defense.
- Scope boundary: tool use, multimodality, mixture-of-experts, SFT/DPO, general pre-training corpora and distilled/internalized debate stay future work; each needs its own research question before entering the build.

## Research ideas (thesis discussion chapter)

- SSM vs. Transformer backbone at fixed params/tokens (perplexity, throughput, memory vs. length) — once L1 exists.
- Internalized debate (Yi et al. 2026): distill the K-path sandbox into a single path, compare tokens vs. accuracy.
- Energy vs. autoregressive reasoning in the sandbox (accuracy, diversity, cost).
- Cross-attention debate vs. mean-pool revision at fixed K (accuracy, cost).
