# API Reference — shipped package (`09-Code/src/ekdant/`, L2-only)

> Complete API surface of the **shipped** 7-module package, read from source. Forward: `embed → T → L2 → head`; the A/B differs only in `use_sandbox`. L1 (`ssm_storage`) and the HF-style wrappers are **DESIGN-ONLY** (final section). Canonical spec: [02](02_System_Architecture.md); knobs: [12](12_Configuration_Guide.md).

## `ekdant.model`

### `EkdantConfig` — plain dataclass (no HF `PretrainedConfig`)

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 32768 | embedding rows |
| `dim` | 512 | width H |
| `n_t` | 9 | T dense blocks (shipped A/B: 12) |
| `n_heads` / `n_kv_heads` | 8 / 2 | GQA split (`dim % n_heads`, `n_heads % n_kv_heads` enforced) |
| `window` | 512 | attention window W (KV bounded by W, never T) |
| `ffn_hidden` | 1024 | SwiGLU inner width |
| `max_seq_len` | 2048 | dataset packing length (shipped configs) |
| `pad_id` | 0 | pad token (overwritten from the trained tokenizer at runtime) |
| `use_sandbox` | true | **the only A/B switch** |
| `n_sandbox` | 4 | K trajectories |
| `debate_rounds` | 2 | R cross-read debate rounds |
| `descent_steps` | 8 | S anchored-gated descent steps |
| `eta` | None → `0.5 / descent_steps` | descent step size |
| `rho_min` | 0.05 | spread-floor threshold (`RHO_MIN_DEFAULT`) |
| `lambda_sbx` | 1.0 | spread-floor loss weight (shipped; thesis design uses 0.1) |
| `readout` | `"mean"` | config-valid values: `mean` / `argmin` |
| `dropout` | 0.0 | embedding dropout (shipped configs: 0.1) |

Methods: `validate()` (divisibility + readout + eta default), `to_dict()`, `from_dict(d)` (constructs + validates).

### `EkdantModel(nn.Module)` / `ModelOutput`

- `forward(input_ids[B,T], labels=None, return_aux=True) -> ModelOutput` — embed (+drop) → T → optional Sandbox (replaces hT with `consensus`, copies tensor aux) → `ln_f` → tied head. `loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)` plus `_floor_penalty` when sandbox is on and labels are given.
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
| `AnchoredDescent(dim)` | `forward(traj, hT, steps, eta) -> (traj, a_t, energy[B], retention[B])` |
| `Sandbox(dim, K=4, rounds=2, steps=8, eta=None→0.5/steps, rho_min=0.05, readout="mean"\|"lowest_energy", gain_lo=0.8, gain_hi=1.25)` | init → debate×R → descent×S → consensus (mean or lowest-energy pick) |

`aux` keys: `anchor_energy`, `trajectory_variance`, `spread_ratio` (ρ), `per_trajectory_retention[B]`, `anchor_distance[B]`. Extras: `spread_floor(rho)`, `@no_grad post_round_state(hT) -> List[[K,B,T,H]]`.

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
| `MCQDataset(examples, tokenizer, max_len=2048)` | rows `{input_ids, answer_pos, answer_id, letter}`; drops over-length rows |
| `collate(batch, pad_id=0)` | `{input_ids, attention_mask, labels (−100 on pad), answer_pos, answer_target}` |
| `bind_letter_ids(tokenizer)` / `LETTER_IDS` | resolves the A–D token ids for letter-token eval |
| `answer_accuracy(model, batch, n_options=4)` | argmax over letter ids at `answer_pos` |

## `ekdant.train` — entrypoint

`python -m ekdant.train --variant sandbox|baseline --out runs/<variant>` (multi-GPU: `torchrun --nproc_per_node=N -m ekdant.train ...`). Loads `configs/{variant}_50m.json` (or `--config`), forces `use_sandbox=False` for baseline, trains/reuses a 32K BPE tokenizer on a MedMCQA sample, runs MedMCQA → MedQA stages (AdamW wd 0.1, per-stage cosine LR + warmup on optimizer steps, grad-clip 1.0 with logged pre-clip `gnorm`, bf16 autocast, optional `torch.compile`, DDP via `torchrun`), prints a progress line every `--log-every` optimizer steps (`step X/Y (%) loss lr gnorm tok/s eta`), and with `--eval-every 1` (default) evaluates the held-out splits after every epoch — appending `history[..].test`, rewriting `results.json` incrementally, and keeping `best.pt` at the peak MedMCQA-validation accuracy — finishing with a final evaluation of both splits (MedMCQA **validation** — the public test split is unlabelled (T18) — and MedQA test) into `results.json` with `history[stage,epoch,loss,lr,gnorm,test]`. Flags: `--batch-size 24`, `--grad-accum 2`, `--lr 3e-4`, `--warmup 200`, `--epochs 3`, `--limit`, `--no-compile`, `--log-every 100`, `--eval-every 1`, `--seed 17`, `--resume`, `--eval-only`. Shipped tests: `tests/test_model.py`, `test_sandbox.py`, `test_transformer.py`, `test_data.py`, `test_schedule.py` (75).

## DESIGN-ONLY APIs (not in the current package)

Thesis-design surface from the pre-restructure docs — none of these are importable from the shipped package:

| Design API | What it would be |
|---|---|
| `ssm_storage.SSMStorageStack` / `SelectiveSSMBlock` / `SSMStorageConfig` | L1: embed + RoPE + selective-SSM stack; `forward(ids, mask) -> (h[B,T,H], compressed[B,3·d_state])`; O(1) state — see [26a](../04-Model-DeepDive/26a_Internal_SSM_Build.md), [21](../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) |
| `EkdantForCausalLM`, `generate()`, HF `PretrainedConfig`/`PreTrainedModel` | HF Trainer-compatible wrapper + autoregressive generation |
| `src.models.factory` (`create_model`, `load_model`, `get_fsdp_layer_cls`) | one-arch factory |
| `src.config.schema`, `src.training.pipeline`, `scripts/production_validation.py` | YAML schema, staged pipeline, 8-phase validation harness |
| `src.data.*` registry pipeline (`build_registry`, `stream_dataset_with_fallbacks`, `document_quality_score`, `pack_sequences`, `WeightedMixedDataset`) | thesis data chapter — shipped `data.py` is the slim MCQ pipeline above |
