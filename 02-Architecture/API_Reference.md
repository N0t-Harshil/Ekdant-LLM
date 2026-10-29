# API Reference — shipped package (`09-Code/src/ekdant/`, T + L2, no L1)

> Complete API surface of the **shipped** 7-module package, read from source. Forward: `embed → T → L2 → head`; the A/B differs only in `use_sandbox`. L1 (`ssm_storage`) and the HF-style wrappers are **DESIGN-ONLY** (final section). Canonical spec: [02](02_System_Architecture.md); knobs: [12](12_Configuration_Guide.md).

## v2 API (`--task gen`)

**`ekdant.model`** — `V2Config(backbone, addon="none"|"sandbox", n_sandbox=4, debate_rounds=4, sandbox_mlp_ratio=4.0, descent_steps=8, eta=0.0625, rho_min=0.05, lambda_sbx=1.0, readout="mean", max_seq_len=1024, max_answer=512, hf_token_file="", addon_layer=-1)` (`addon_layer` k ≥ 1 hooks L2 after decoder layer k) with `validate`, `sandbox_kwargs`, `to_dict`, `from_dict`. `MultiAddOn(dim, sandbox_kwargs, n_sites, shared=True, gate_cap=0.0, final=False, checkpoint=False)` (`final_site`: the same L2 after the final norm, T40) (v2.2: `sites` = one `GatedAddOn` per depth sharing one `Sandbox`; `gates()`, `site_gate_magnitudes()`, `gate_magnitude()` = mean). `V2Config.addon_layers: list`, `addon_shared: bool`, `sites()`. `GatedAddOn(dim, sandbox_kwargs, mode="final"|"mid")` — `mid` (v2.1) returns `h + gate*out_norm(consensus(in_norm(h)))`; `final`: `forward(h, mask) -> (h + gate*(out_norm(consensus(h))-h), aux)` (`out_norm` = RMSNorm, T32), `gate` zeros `[dim]`, `gate_magnitude()`. `EkdantLM(cfg, lm)`: `forward(input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, position_ids=None) -> ModelOutput` (answer-only CE + spread floor; `aux["past_key_values"]` when caching), `addon_params()`, `param_groups(backbone_lr, addon_lr, wd=0.1, gate_lr=None)` (the gate in its own group when `gate_lr` is given).

**`ekdant.sandbox.Sandbox`** gains `mlp_ratio: float = 2.0` (debate MLP width) and `norm_inputs: bool = False` (pre-norm in every `DebateRound`, T33); both defaults keep v1 unchanged. `V2Config.sandbox_prenorm = True` turns the pre-norm on.

**`ekdant.backbone`** — `apply_hf_token(path) -> "env"|"file"|"none"` (exports the token file's first line as `HF_TOKEN` unless already set; never logs it), `load_backbone(model_id)` (float32), `load_tokenizer(model_id)`, `tiny_llama(vocab=64, dim=32, layers=2)`.

**`ekdant.gen_data`** — `GenExample(question, answer, context="", qtype="")`; `load_pubmedqa(limit=None, seed=17)`, `load_medquad(limit=None, seed=17)` (empty rows dropped); `format_gen_prompt(ex)`; `GenDataset(examples, tokenizer, max_len=1024, max_answer=512)` with items `{input_ids, labels, idx}`, `prompt_ids(i)`, `example(i)`, `dropped`; `gen_collate(batch, pad_id)`; `question_key(q)`; `split_by_question(examples, seed=17, bench_frac=0.5, val_frac=0.1) -> (ft, val, bench)`; `holdout(examples, n=2000, seed=17) -> (train, held)`.

**`ekdant.gen_eval`** — `answer_nll(model, loader, device, autocast) -> {nll, ppl, tokens, n, row_nll, row_tokens, row_idx[, addon]}`; `by_type(result, examples)`; `normalise_text`, `rouge_l(pred, ref)`, `token_f1(pred, ref)`, `score_generations(preds, refs)`; `generate_greedy(model, prompts: List[List[int]], max_new, device, eos_id, pad_id, batch_size=16, autocast=None) -> List[List[int]]`; `paired_bootstrap(a, b, groups, n=1000, seed=17, weights=None) -> {diff, lo, hi}` (with per-row NLL sums and token weights: the per-token difference); CLI `python -m ekdant.gen_eval compare <run> <run> …`.

**`ekdant.train_gen` gate floor (T39)** — `GateRamp(target=0.0, start=200, steps=2000, backoff=0.1)` with `floor()`, `step(burst)`, `pos`, state dict; `GatedAddOn.gate_floor` / `set_gate_floor(f)` and `MultiAddOn.set_gate_floor(f)`: effective gate `f + (1 − f) · tanh(θ)`.

**Rollback (T43)** — `GateRamp(target, start, steps, backoff=0.1, slowdown=2.0, step_down=0.05, min_target=None)` with `rollback(pos)` and state `{pos, target, steps, rollbacks}`; `take_snapshot(model, opt, pos) -> dict` (CPU copy), `restore_snapshot(model, opt, snap)`. CLI: `--gate-rollback`, `--snapshot-every`, `--floor-slowdown`, `--floor-step-down`, `--gate-floor-min`.

**Forced L2 (T42)** — `GatedAddOn(..., force=False)` / `MultiAddOn(..., force=False)` / `V2Config.gate_force`: with a floor set, `effective_gate()` never falls below it (straight-through clamp), `out_norm.weight` is frozen, and a mid site adds `g · rms(h) · out_norm(c)`. CLI: `--gate-force` (requires `--gate-floor > 0`).

**`ekdant.train_gen` gate control (T37)** — `gate_boost(step, boost, steps) -> float`; `GateBrake(spike=3.0, l2_max=50.0, cut=0.5, recover=1.01, min_factor=0.05, warmup=20)` with `update(gn_addon, l2_rms) -> bool` (burst), `factor`, `events`, `state_dict()` / `load_state_dict()`. `GatedAddOn(..., gate_cap=0.0)` + `effective_gate()`; every param group carries `kind` (`backbone` / `addon` / `gate`).

**`ekdant.train_gen`** — `pick_device(gpu, min_free_gb)` (`gpu` = `auto`, an index or `cpu`; no silent CPU fallback), `phase1_eval_points(nb, eval_fraction, grad_accum)`, `clip_grads(model, max_norm) -> (backbone_norm, addon_norm)`, `save_checkpoint(state, path)`, `build_parser()`, `main(argv=None)`; `last.pt = {model, optimizer, cfg, stage, epoch, batches_done, opt_step, history, best_val_nll, tokens_seen, wall_clock_s, run_args}`. **`ekdant.train.main(argv=None)`** dispatches `--task gen` to it.

The sections below are the v1 API.

## `ekdant.model`

### `EkdantConfig` — plain dataclass (no HF `PretrainedConfig`)

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 32768 | embedding rows (the trained tokenizer uses 26,766 of them) |
| `dim` | 512 | width H |
| `n_t` | 9 | T dense blocks (shipped A/B: 12) |
| `n_heads` / `n_kv_heads` | 8 / 2 | GQA split (`dim % n_heads`, `n_heads % n_kv_heads` enforced) |
| `window` | 512 | attention window W (KV bounded by W, never T) |
| `ffn_hidden` | 1024 | SwiGLU inner width |
| `max_seq_len` | 1024 | longest example kept (shipped configs: 2048) |
| `pad_id` | 0 | pad token (overwritten from the trained tokenizer at runtime) |
| `use_sandbox` | true | **the only A/B switch** |
| `n_sandbox` | 4 | K trajectories |
| `debate_rounds` | 2 | R cross-read debate rounds |
| `descent_steps` | 8 | S anchored-gated descent steps |
| `eta` | None → `0.5 / descent_steps` | descent step size |
| `rho_min` | 0.05 | spread-floor threshold (`RHO_MIN_DEFAULT`) |
| `lambda_sbx` | 1.0 | spread-floor loss weight (shipped; thesis design uses 0.1) |
| `readout` | `"mean"` | config-valid values: `mean` / `lowest_energy` (same as `Sandbox`) |
| `dropout` | 0.0 | embedding dropout (shipped configs: 0.1) |

Methods: `validate()` (divisibility + readout + eta default), `to_dict()`, `from_dict(d)` (constructs + validates).

### `EkdantModel(nn.Module)` / `ModelOutput`

- `forward(input_ids[B,T], labels=None, return_aux=True, answer_pos=None, answer_ids=None, answer_weight=0.0, attention_mask=None) -> ModelOutput` — embed (+drop) → T → optional Sandbox (replaces hT with `consensus`, copies tensor aux) → `ln_f` → tied head. `loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)` + `answer_weight · CE(logits[b, answer_pos−1], answer_ids)` (when `answer_weight > 0` and both answer tensors are given) + `_floor_penalty` (sandbox on, `return_aux=True`). `attention_mask` only restricts the sandbox statistics and floor to real tokens; logits never depend on it.
- `_floor_penalty(aux)` — `lambda_sbx · mean(relu(rho_min − spread_ratio)²)`; zero when no `spread_ratio` in aux.
- `score_options(input_ids[B·n,L], option_mask[B·n,L], n_options=4) -> [B, n_options]` — masked mean log-probability per MCQ option.
- `num_parameters(trainable_only=True) -> int`.
- `ModelOutput`: `logits[B,T,V]`, `loss` (optional), `aux: Dict[str, Tensor]`.

### Module helpers

`count_parameters(model, trainable_only=True)` · `estimate_model_size(cfg) -> {total, embedding_tied, t, sandbox, ln_f}` (instantiates the model to count) · `baseline_config(cfg)` (copy with `use_sandbox=False`) · `RHO_MIN_DEFAULT = 0.05`.

## `ekdant.transformer_stack` — layer T

| API | Signature / shapes |
|---|---|
| `RMSNorm(dim, eps=1e-6)` | `forward(x[B,T,H]) -> [B,T,H]` |
| `rope_cache(dim, length, base=1e5)` | `-> (cos, sin)` |
| `rotate_half(x)` / `apply_rope(x, cos, sin)` | RoPE injection |
| `window_mask(length, window)` | `[T,T]` causal bool mask, distance < W |
| `repeat_kv(x, n_rep)` | GQA KV expansion |
| `WindowedGQA(dim, n_heads, n_kv_heads, window)` | bias-free q/k/v/o `Linear`, `head_dim = dim/n_heads`; `forward(x, cos, sin, mask)` via `F.scaled_dot_product_attention` |
| `SwiGLU(dim, hidden)` | `down(SiLU(gate(x)) * up(x))` |
| `TransformerBlock(...)` | pre-norm attn + residual → pre-norm FFN + residual |
| `TransformerStack(dim, n_layers, n_heads=8, n_kv_heads=2, window=512, ffn_hidden=None→2·dim, rope_base=1e5)` | `forward(x[B,T,H]) -> hT[B,T,H]`; final `norm_out`; `o_proj`/`down` zero-initialised |

## `ekdant.sandbox` — layer L2

| API | Signature / notes |
|---|---|
| `SandboxOutput` | NamedTuple: `consensus[B,T,H]`, `trajectories[K,B,T,H]`, `aux` |
| `mutual_orthogonal(K, n)` / `spread_gains(K, lo=0.8, hi=1.25)` | init helpers (Frobenius-orthogonal unit matrices; log-spaced gains) |
| `DiverseInit(dim, K, gain_lo=0.8, gain_hi=1.25)` | `forward(hT[B,T,H]) -> [K,B,T,H]` |
| `CrossRead(dim)` | `forward(traj[K,B,T,H])` — each trajectory attends the other K−1, self masked |
| `DebateRound(dim, mlp_ratio=2.0, gate_bias=-2.0)` | cross-read + sigmoid repair gate + SiLU MLP |
| `AnchoredDescent(dim)` | `forward(traj, hT, steps, eta, mask=None) -> (traj, a_t, energy[K,B,T], retention[K])` |
| `Sandbox(dim, K=4, rounds=2, steps=8, eta=None→0.5/steps, rho_min=0.05, readout="mean"\|"lowest_energy", gain_lo=0.8, gain_hi=1.25)` | `forward(hT[B,T,H], mask=None)`: init → debate×R → descent×S → consensus (mean, or per-position lowest-anchor-energy pick); aux statistics over real tokens |

`aux` keys: `anchor_energy` (mean ½‖T_k − a_t‖²), `trajectory_variance`, `spread_ratio` (ρ = variance / mean(hT²)), `per_trajectory_retention[K]`, `anchor_distance[K]` — all means over real tokens when a mask is given. Helper: `masked_mean_bt(x, mask)`. Extras: `spread_floor(rho)`, `@no_grad post_round_state(hT) -> List[[K,B,T,H]]`.

## `ekdant.output_head`

`OutputHead(embed)` — reuses the embedding weight (tied); `forward(x[B,T,H]) -> logits[B,T,V]`; `num_parameters == 0`.

## `ekdant.data` — MedMCQA/MedQA MCQ pipeline

| API | Notes |
|---|---|
| `format_prompt(question, options) -> str` | question + `A.`–`D.` lines + `\nAnswer:` |
| `normalise_options(raw)` / `letter_from_answer(answer, options)` | dict/list containers; int/str index → letter |
| `MCQExample` | dataclass: `question`, `options`, `answer` |
| `load_examples(name, split, limit=None, seed=17)` | `name ∈ {"medmcqa", "medqa"}` → `List[MCQExample]` |
| `Tokenizer` | `train(texts, vocab_size, path, min_freq=2)`, `load(path)`, `encode(text, add_eos=True)`; exposes `vocab_size/bos_id/eos_id/pad_id` |
| `MCQDataset(examples, tokenizer, max_len=1024)` | rows `{input_ids, answer_pos, answer_id, letter, letter_idx}`; drops over-length rows (`train.py` passes `max_seq_len`, 2048) |
| `collate(batch, pad_id=0)` | `{input_ids, attention_mask, labels (−100 on pad), answer_pos, answer_target (letter index 0–3), answer_ids (letter vocab id)}` |
| `bind_letter_ids(tokenizer)` / `LETTER_IDS` | resolves the A–D token ids for letter-token eval |
| `answer_accuracy(model, batch, n_options=4)` | argmax over letter ids at `answer_pos − 1` (the position that predicts the letter) |

## `ekdant.train` — entrypoint

`python -m ekdant.train --variant sandbox|baseline --out runs/<variant>` (multi-GPU: `torchrun --nproc_per_node=N -m ekdant.train ...`). Loads `configs/{variant}_50m.json` (or `--config`), forces `use_sandbox=False` for baseline, trains/reuses a BPE tokenizer on a MedMCQA sample (32,768 budget; it learns 26,766), then runs the **sequential two-phase protocol** (never trained jointly; AdamW wd 0.1, per-phase cosine LR + warmup on optimizer steps, grad-clip 1.0 with logged pre-clip `gnorm`, bf16 autocast (fp16 + GradScaler where bf16 is unsupported), optional `torch.compile`, DDP via `torchrun`): **phase 1** trains on the full MedMCQA train split (182,496 labelled rows) and records the MedMCQA validation result once (`results["phase1"]["medmcqa_val"]` — the public test split is unlabelled, T18), scoring it every `--eval-every` epochs on the way for monitoring only (T25); **phase 2** fine-tunes on the **60% partition** of MedQA train (`split_frac`, `FT_FRAC = 0.6`, `--seed`-locked → identical rows for both A/B arms) while `--eval-every 1` evaluates the untouched MedMCQA validation set after each epoch (history + incremental `results.json` + `best.pt` on `medmcqa_val`); the **final protocol** evaluates `best.pt` vs `last.pt` on `medmcqa_val`, loads the better (ties keep `best.pt`), and scores the complementary **40% MedQA benchmark partition exactly once** plus the official MedQA test (also untouched) into `results.json` (`test{medmcqa_val, medqa_bench, medqa_test}` + `phase1` + `selected{checkpoint, selected_on, best_val_accuracy, last_val_accuracy}`). Progress line every `--log-every` optimizer steps (`step X/Y (%) loss lr gnorm tok/s eta`); `train_stage` returns `(loss, ntok, opt_step, last_lr, avg_gnorm, sandbox_diag)` and `evaluate` returns `{loss, accuracy, n, pred_counts, gold_counts, majority_accuracy[, sandbox]}` (letter histograms + always-majority-letter accuracy, T27) — sandbox diagnostics land in `history[*].sandbox` and `test.*.sandbox` (T24). Flags: `--batch-size 24`, `--grad-accum 2`, `--lr 3e-4`, `--warmup 200`, `--epochs 3` (phase 1), `--ft-lr`, `--ft-epochs` (phase-2 schedule; default to `--lr`/`--epochs`, `5e-5`/`10` recommended for the 6,106-row fine-tune), `--answer-weight` (adds a re-weighted answer-letter CE to the LM loss — the letter is 1 token in ~34/173, so plain CE dilutes the decision ~97–99%; default `0` = legacy, `10` recommended), `--limit`, `--no-compile`, `--log-every 100`, `--eval-every 1`, `--seed 17`, `--resume`, `--eval-only`. Shipped tests: `tests/test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py`, `test_eval_split.py` (201).

## DESIGN-ONLY APIs (not in the current package)

Thesis-design surface from the pre-restructure docs — none of these are importable from the shipped package:

| Design API | What it would be |
|---|---|
| `ssm_storage.SSMStorageStack` / `SelectiveSSMBlock` / `SSMStorageConfig` | L1: embed + RoPE + selective-SSM stack; `forward(ids, mask) -> (h[B,T,H], compressed[B,3·d_state])`; O(1) state — see [26a](../04-Model-DeepDive/26a_Internal_SSM_Build.md), [21](../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) |
| `EkdantForCausalLM`, `generate()`, HF `PretrainedConfig`/`PreTrainedModel` | HF Trainer-compatible wrapper + autoregressive generation |
| `src.models.factory` (`create_model`, `load_model`, `get_fsdp_layer_cls`) | one-arch factory |
