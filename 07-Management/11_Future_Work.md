# Future Work (thesis-ordered)

## Critical (must finish for thesis defense)

- **Build Ekdant L1+T+L2 + 10-step loss-decrease test** — **PARTIALLY DONE:** T+L2 built and shipped L2-only at `09-Code/src/ekdant/{model,transformer_stack,sandbox,output_head,data,train}.py`; 67 tests pass in `09-Code/tests/{test_model,test_sandbox,test_transformer,test_data}.py` (forward shapes, grads through T+L2, loss decrease, data-schema regression). **Remains:** L1 (`ssm_storage`) integration into `model.py` — not in the code (L1 stays in the vault/thesis design docs for now); the full L1+T+L2 grad/loss check lands with it.
- **Internal-only L1+T+L2** — **mostly done for the shipped package:** pure-PyTorch scan/debate, zero third-party kernels (no `mamba-ssm`/`flash-attn`/`triton`), allowlist enforced by import-guard tests (`test_model_has_no_blocked_imports`, `test_no_blocked_imports_in_ekdant_package`); L1 half pending (see above), per [23](../02-Architecture/02_System_Architecture.md).
- **O(1) memory proof** — same `compressed[B,3·d_state]` triple shape + flat peak memory for T=512/2048/4096. Table in thesis. **Blocked on L1** (no `ssm_storage` in the code yet).
- **Sandbox ablation** - K=1 vs. 4 vs. 8 and rounds 0 vs. 2 on the XS/S-runnable set (perplexity, ARC-easy/HellaSwag/LAMBADA, synthetic known-answer multi-step). GSM8K/HumanEval are ==scaling-held-out== — at 12–46M params they sit at the floor and cannot decide the claim ([29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) §3.3). If K=1 ties, report it honestly.
- **Depth-matched control (new, forced by C24)** — a plain Transformer with the **same parameter count and the same block-pass budget but no L2**. **Still open:** the shipped A/B is architecture-matched only (sandbox 50.09M vs baseline 43.53M — the 6.56M delta is L2 itself), so it cannot separate a gain from L2 against a gain from spending the same compute on more untied depth, which is exactly what C24 (arXiv:2603.08082) found winning. C24 also reports the full Autoregressive TRM giving **no consistent benefit**, so a tie is a publishable outcome and must be reported if it is what we measure.
- **Energy-mechanism ablation (new)** - anchored gated descent vs. the discarded peer-mean pull (expect: the peer-mean pull leaves logits bit-identical), `sandbox_steps` ∈ {0,4,8}, and a `sandbox_rho_min` sweep ∈ {0.01, 0.05, 0.2} to show the floor trading collapse-risk against forced agreement.
- **Transformer ablation** — T blocks (XS: 3, S: 5, M: 6), window 256/512, XS→S→M ladder, on recall probe + reasoning samples.
- **Layer-addition ladder (the core proof)** — backbone-only → +L1 → +L1+L2 at matched params/FLOPs on recall probe + GSM8K/HumanEval samples; every gain must be attributable to the added layer. **Needs L1 integration first** (current code is T+L2 only).
- **50K foundation runs + checkpoints** — end-to-end on the runnable configs `09-Code/configs/{sandbox,baseline}_50m.json` (the `config_ekdant_xs/s/m.yaml` sizes are thesis-documented specs), Phase 8 PASS each, 3 saved checkpoints. **Training runs themselves have not been executed yet.**
- **Fix doc scraping gaps for corpus chapter** — PyTorch mirror, PostgreSQL sitemap (keep from old list; data chapter needs full corpus).

## High (should finish)

- SFT mini-run (DPO optional), 5-benchmark eval (HumanEval/GSM8K/MMLU/HellaSwag/ARC), packing-quality numbers, eval-vs-baseline table (Transformer tiny at equal params if time).
- Tidy the tree so the examiner sees only thesis files (`09-Code/src/ekdant/` 7 modules — `model`, `transformer_stack`, `sandbox`, `output_head`, `data`, `train`, `__init__` — runnable configs, `papers/`, vault).
- CI: `pytest` on PR in `09-Code/` (67 shipped model/sandbox/transformer/data tests).

## Medium (if time)

- Triton selective-scan kernel, long-context RoPE extension test (4K→16K), custom code tokenizer.
- More languages in registry, AST filter for Ruby/PHP/Kotlin.

## Low / Nice-to-have (post-thesis)

- Demo UI, REST API, quantization, LoRA, synthetic data, hyperparam search, Docker — all deferred. Do not add before defense.
- Scope boundary: tool use, multimodality, mixture-of-experts, and distilled/internalized debate stay future work (Ch. 7); each needs its own research question before entering the build.

## Research ideas (thesis discussion chapter)

- SSM vs. Transformer backbone at fixed params/tokens (perplexity, throughput, memory vs. length).
- SSM vs. Transformer-tiny backbone at fixed params/tokens (perplexity, throughput, memory vs. length).
- Internalized debate (Yi et al. 2026): distill K-path sandbox into single path, compare tokens vs. accuracy.
- Energy vs. autoregressive reasoning in sandbox (accuracy, diversity, cost).
- Cross-attention debate vs. mean-pool revision at fixed K (accuracy, cost).
