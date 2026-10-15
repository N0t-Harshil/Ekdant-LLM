# Configuration Guide (Ekdant, three sizes: XS ≈12M, S ≈21M, M ≈46M)

> [!note] **Two surfaces.** §1 is the **shipped** parameter surface: every knob in `09-Code/configs/sandbox_50m.json` and `baseline_50m.json` (read from the JSONs and `EkdantConfig`). §2–§4 are the **thesis-design** spec: the three YAML configs (`config_ekdant_xs/s/m.yaml`) are documented design, not files in this vault.

## 1. Shipped configs (`09-Code/configs/*.json`)

Two JSONs, identical architecture — **only `use_sandbox` differs** (sandbox 50.09M / baseline 43.53M at N_T=12; delta = L2's measured 6.56M). Every knob maps 1:1 to an `EkdantConfig` field (`09-Code/src/ekdant/model.py`); unknown keys would raise in `from_dict`.

| Knob | `sandbox_50m.json` | `baseline_50m.json` | Meaning |
|---|---|---|---|
| `vocab_size` | 32768 | 32768 | embedding rows; must match the trained BPE tokenizer |
| `dim` | 512 | 512 | model width H |
| `n_t` | 12 | 12 | T dense blocks (thesis S/M: 5/6 — deeper backbone for the 50M run) |
| `n_heads` | 8 | 8 | Q heads (head_dim = 64) |
| `n_kv_heads` | 2 | 2 | KV heads (GQA 8:2) |
| `window` | 512 | 512 | attention window W; KV bounded by W, never T |
| `ffn_hidden` | 1024 | 1024 | SwiGLU inner width (2×dim) |
| `max_seq_len` | 2048 | 2048 | dataset packing length (MCQ rows longer than this are dropped); matches the thesis 2048 spec |
| `pad_id` | 0 | 0 | pad token id — overwritten at runtime from the trained tokenizer |
| `use_sandbox` | **true** | **false** | **the only A/B difference** |
| `n_sandbox` | 4 | 4 | K trajectories (inactive in baseline) |
| `debate_rounds` | 2 | 2 | R cross-read debate rounds |
| `descent_steps` | 8 | 8 | S anchored-gated descent steps |
| `eta` | 0.0625 | 0.0625 | descent step size (= the `0.5/descent_steps` default rule) |
| `rho_min` | 0.05 | 0.05 | spread-floor threshold `ρ_min` |
| `lambda_sbx` | 1.0 | 1.0 | spread-floor loss weight (shipped; thesis design uses 0.1) |
| `readout` | `"mean"` | `"mean"` | consensus readout (`Sandbox` also accepts `"lowest_energy"`) |
| `dropout` | 0.1 | 0.1 | embedding dropout |

`validate()` fills `eta` when null and enforces divisibility (`dim % n_heads`, `n_heads % n_kv_heads`). Training-time knobs (batch size, LR, warmup, epochs, seed) are **CLI flags** on `python -m ekdant.train`, not JSON keys — defaults and flags in [API_Reference.md](API_Reference.md); run guide in `09-Code/README.md`.

## 2. Thesis design size table (~13 knobs)

Three configs from one schema: `config_ekdant_xs.yaml` (XS, smoke/debug), `config_ekdant_s.yaml` (S, ablations), `config_ekdant_m.yaml` (M, main run). Same recipe — 50K steps, lr 1e-4 cosine, warmup 3K, eff batch 16, AdamW fused, wd 0.1, clip 1.0, BF16, single GPU. No MoE anywhere: the sandbox debates instead.

| Parameter | XS (≈12M) | S (≈21M) | M (≈46M) | Notes |
|---|---|---|---|---|
| `model.architecture.hidden_size` | 256 | 320 | 512 | Width; heads 8Q/2KV (dim 32 / 40 / 64) |
| `model.architecture.vocab_size` | 32768 | 32768 | 32768 | Must match tokenizer |
| `model.architecture.max_position_embeddings` | 4096 | 4096 | 4096 | Pretrain window |
| `model.architecture.rope_theta` | 100K | 100K | 100K | Shared RoPE base (scaled for small head dims) |
| `model.architecture.tie_word_embeddings` | true | true | true | Tied head saves 8.4M / 10.5M / 16.8M |
| `model.architecture.d_state` | 96 | 128 | 256 | **L1 knob:** per-store dim (×3 stores: content/control/sandbox) |
| `model.architecture.n_ssm_layers` | 2 | 4 | 4 | L1 depth |
| `model.architecture.n_transformer_layers` | 3 | 5 | 6 | **T knob:** dense blocks, windowed GQA + SwiGLU |
| `model.architecture.transformer_window` | 512 | 512 | 512 | T local window (KV bounded by W) |
| `model.architecture.n_sandbox_paths` | 4 | 4 | 4 | **L2 knob:** K trajectories |
| `model.architecture.n_debate_rounds` | 2 | 2 | 2 | L2 debate depth |
| `model.architecture.sandbox_steps` | 8 | 8 | 8 | L2 anchored gated descent iterations (moves the readout; ablate 0/4/8) |
| `model.architecture.sandbox_weight` | 0.1 | 0.1 | 0.1 | Weight on the spread floor `max(0, ρ_min − ρ)²` |
| `model.architecture.sandbox_rho_min` | 0.05 | 0.05 | 0.05 | Collapse floor: min fraction of backbone activation energy the K hypotheses must keep spanning |
| `training.max_seq_length` | 2048 | 2048 | 2048 | Packing length |
| `training.pretrain.max_steps` | 50000 | 50000 | 50000 | Thesis run (~1.6B tokens @32K tok/step) |
| `distributed.strategy` | `none` | `none` | `none` | Single GPU all sizes |
| `data.use_registry` | true | true | true | Always true for thesis |
| `output.model_dir` | `./models/ekdant-xs` | `./models/ekdant-s` | `./models/ekdant-m` | Per-size outputs |

That is the complete parameter surface — every knob maps to an ablation in [11](../07-Management/11_Future_Work.md).

## 3. Param math (counted ≈, verify with `estimate_model_size`)

- Embed V·H tied: XS 8.4M · S 10.5M · M 16.8M.
- aSSM block ≈ shared trunk + 3 heads (≈1.5× single-store): XS ≈ 1.5M · S ≈ 4.8M · M ≈ 13M.
- T block ≈ 2.5H² GQA + 3·H·I SwiGLU (I=2H): XS ~0.6M ×3 ≈ 1.7M · S ~0.9M ×5 ≈ 4.4M · M ~2.2M ×6 ≈ 13.4M.
- L2 ≈ diverse init + R=2 debate nets + anchor `W_a [H,H]` + gate `W_g [2H,H]`: XS ~0.8M · S ~1.3M · M ~2.8M. (Rose from 0.6/1.0/2.2M when the peer-mean pull was replaced by anchored gated descent.)
- **Totals: XS ≈ 12M · S ≈ 21M · M ≈ 46M** (XS/S/M are size-class *labels*; the counts above are the actual budgets).

**Effective batch:** `per_device × gpus × accum` = 4×1×4=16 (32,768 tok/step @2048). **RoPE:** base 100K, window 4096. YaRN frequency rescaling is implemented but **off by default** at XS/S/M, because `T ≤ 2048` is well inside the 4096 window; enable it only for longer-context runs and record the factors used.

## 4. Tokenizer (32K BPE, trained in-house — design; shipped 09-Code uses HF datasets' text with its own prompt/letter-token pipeline)

- Train custom 32K BPE with `src/tokenizer_trainer.py` (thesis-design path, not in this vault) on a registry sample (code-heavy to match the 0.30 code weight) — same tokenizer for XS/S/M so sizes stay comparable.
- HF tokenizers only as a smoke-debug fallback, never the thesis runs (internal-only rule).
- Validate: decode 20 samples clean (no U+FFFD), vocab file pinned per run in `output.model_dir`.
