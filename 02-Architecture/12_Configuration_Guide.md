# Configuration Guide (Ekdant, 50M A/B)

> [!note] **Shipped surface.** §1 lists every knob in `09-Code/configs/sandbox_50m.json` and `baseline_50m.json` (read from the JSONs and `EkdantConfig`). §2 lists the training-time CLI flags. §3–§4 are as-built parameter math and the tokenizer. The earlier XS/S/M design sizes with L1 are summarised in §5 for the thesis only.

## 0. v2 config (`configs/v2_smollm2_135m.json`, `V2Config`) and flags

| Key | Value | Meaning |
|---|---|---|
| `backbone` | `HuggingFaceTB/SmolLM2-135M` | Hugging Face model id |
| `addon` | set by `--variant` (`baseline`→`none`, `sandbox`→`sandbox`) | whether L2 sits on the last hidden state |
| `n_sandbox` | 4 | K trajectories |
| `debate_rounds` | 4 | R (v1: 2) |
| `sandbox_mlp_ratio` | 4.0 | debate MLP width / H (v1: 2.0) |
| `descent_steps`, `eta` | 8, 0.0625 | S, η |
| `rho_min`, `lambda_sbx` | 0.05, 1.0 | spread floor |
| `readout` | `mean` | consensus |
| `max_seq_len`, `max_answer` | 1024, 512 | sequence cap (context trimmed), answer cap |

Flags (`python -m ekdant.train --task gen …`): `--variant {baseline,sandbox}`, `--batch-size 8`, `--grad-accum 4`, `--lr 5e-5` (backbone), `--addon-lr 1e-3`, `--warmup 200`, `--epochs 1` (PubMedQA), `--ft-epochs 3` (MedQuAD), `--eval-fraction 0.25`, `--pubmed-holdout 2000`, `--gen-subset 1000`, `--gen-subset-pubmed 500`, `--max-new 256`, `--max-new-pubmed 128`, `--gen-batch 32`, `--limit`, `--seed 17`, `--log-every 50`, `--gpu auto`, `--min-free-gb 20`, `--resume`, `--eval-only`.

Sections 1–5 below are the v1 (`--task mcq`) configs.

## 1. Model configs (`09-Code/configs/*.json`)

Two JSONs, identical architecture — **only `use_sandbox` differs** (sandbox 50.09M / baseline 43.53M at N_T=12; delta = L2's measured 6.56M). Every knob maps 1:1 to an `EkdantConfig` field (`09-Code/src/ekdant/model.py`); unknown keys raise in `from_dict`.

| Knob | `sandbox_50m.json` | `baseline_50m.json` | Meaning |
|---|---|---|---|
| `vocab_size` | 32768 | 32768 | embedding rows; must match the trained BPE tokenizer |
| `dim` | 512 | 512 | model width H |
| `n_t` | 12 | 12 | T dense blocks |
| `n_heads` | 8 | 8 | Q heads (head_dim = 64) |
| `n_kv_heads` | 2 | 2 | KV heads (GQA 8:2) |
| `window` | 512 | 512 | attention window W; KV bounded by W, never T |
| `ffn_hidden` | 1024 | 1024 | SwiGLU inner width (2×dim) |
| `max_seq_len` | 2048 | 2048 | longest example kept (nothing is dropped on MedMCQA/MedQA — p99 is 118/364 tokens) |
| `pad_id` | 0 | 0 | pad token id — overwritten at runtime from the trained tokenizer |
| `use_sandbox` | **true** | **false** | **the only A/B difference** |
| `n_sandbox` | 4 | 4 | K trajectories (inactive in baseline) |
| `debate_rounds` | 2 | 2 | R cross-read debate rounds |
| `descent_steps` | 8 | 8 | S anchored-gated descent steps |
| `eta` | 0.0625 | 0.0625 | descent step size (= the `0.5/descent_steps` default rule) |
| `rho_min` | 0.05 | 0.05 | spread-floor threshold `ρ_min` |
| `lambda_sbx` | 1.0 | 1.0 | spread-floor loss weight (original design value was 0.1) |
| `readout` | `"mean"` | `"mean"` | consensus readout: `"mean"` or `"lowest_energy"` (validated identically by `EkdantConfig` and `Sandbox`) |
| `dropout` | 0.1 | 0.1 | embedding dropout |

`validate()` fills `eta` when null and enforces divisibility (`dim % n_heads`, `n_heads % n_kv_heads`) and the readout name.

## 2. Training flags (`python -m ekdant.train`)

Not JSON keys. Defaults and the campaign recipe:

| Flag | Default | Campaign (`run.md` §3) |
|---|---|---|
| `--batch-size` / `--grad-accum` | 24 / 2 | 6 / 8 on a 16 GB T4 → effective 48 |
| `--lr`, `--warmup` | 3e-4, 200 | 3e-4, 200 (cosine per phase) |
| `--epochs` | 3 | 100 (phase 1, MedMCQA) |
| `--ft-lr`, `--ft-epochs` | = `--lr`, = `--epochs` | 5e-5, 10 (phase 2, MedQA 60 %) |
| `--answer-weight` | 0.0 | 10 |
| `--seed` | 17 | 17 (also locks the 60/40 MedQA partition) |
| `--eval-every` | 1 | 1 |

Full table: [API_Reference.md](API_Reference.md) and [`09-Code/README.md`](../09-Code/README.md). Fixed in code, not flags: AdamW (wd 0.1, betas 0.9/0.95), grad clip 1.0, bf16 autocast (fp16 fallback), `torch.compile` unless `--no-compile`.

## 3. Parameter math (as built — `estimate_model_size`)

- Embeddings, tied: V·H = 32768 × 512 = **16.78M**
- T block: 2.5H² (GQA projections) + 3·H·I (SwiGLU, I = 2H) = **2,229,248**; × 12 = **26.75M**
- L2: **6.56M** — `DiverseInit` K·H² + R debate rounds (full q/k/v H×H cross-projections, H→2H→H MLP, gate 2H→H, untied across rounds) + anchor W_a (H×H) and gate (2H→H)
- **Totals: baseline 43.53M · sandbox 50.09M**

L2 is 2.3× the thesis closed-form estimate (2.8M): that estimate budgeted ~0.97M for debate across R=2 rounds; the build spends ~4.7M. **Decision pending before any results table is published:** slim/tie L2, or restate the estimate (the A/B itself is unaffected — the baseline just omits L2).

**Effective batch:** `batch_size × grad_accum × n_gpu`. **RoPE:** base 100K.

## 4. Tokenizer (as built)

BPE configured for 32,768 entries — it learns **26,766** on the 20,000-prompt sample, so 6,002 embedding rows (3.07M parameters per arm, counted in the totals above) are never used (HF `tokenizers`; NFD + lowercase, whitespace pre-tokenizer, `min_frequency=2`, specials `<pad> <bos> <eos> <unk>`), trained once per run directory on 20,000 MedMCQA training prompts and saved to `<out>/tokenizer.json`. Copy it into the second arm's `--out` before starting that arm to guarantee an identical vocabulary. Details: [05_Data_Pipeline](../05-Data/05_Data_Pipeline.md) §4.

## 5. Design-time size table (thesis only — includes L1, not built)

The original three-size design (XS ≈12M / S ≈21M / M ≈46M, 1× A100, 50K steps) is kept in the thesis (§Parameters) as a closed-form estimate for the **full three-stage model**. It is superseded by §3 for every result.

| Knob | XS | S | M |
|---|---|---|---|
| width H | 256 | 320 | 512 |
| L1 `d_state` / blocks | 96 / 2 | 128 / 4 | 256 / 4 |
| T blocks | 3 | 5 | 6 |
| L2 params (est.) | 0.8M | 1.3M | 2.8M |
| Total (est.) | 12.4M | 21.0M | 46.0M |
