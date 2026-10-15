# 26b — Build Manual: dense Transformer for T (XS/S/M sizes)

> Prev: [26a](26a_Internal_SSM_Build.md) · Next: [26c](26c_Internal_Sandbox_Build.md) · Map: [README](../README.md) · Spec: [02](../02-Architecture/02_System_Architecture.md) §4

> [!note] T is ==plain dense Transformer blocks== — windowed GQA + SwiGLU, identical every block. No MoE (the sandbox debates instead), no sparse indexer, no dense/sparse split: at 10–40M those add machinery without capacity to justify it. Everything below is pure PyTorch core — hand-built masks, `scaled_dot_product_attention`, never `flash-attn`.

## 1. Block layout (file: `transformer_stack.py`, 169 lines)

N identical blocks (**thesis XS: 3, S: 5, M: 6; shipped A/B: N_T=12** at H=512). Each block is ==pre-norm ×2, residual ×2==:

1. RMSNorm → WindowedGQA (`window` = 512, RoPE shared base 100K — KV bounded by W) → residual.
2. RMSNorm → dense SwiGLU FFN (I = 2H default) → residual.

`TransformerStack.forward(x[B,T,H])` builds the RoPE cos/sin and the window mask **once per call** and threads them through every block; output `hT [B,T,H]` after a final `norm_out` RMSNorm. (The model adds one more RMSNorm `ln_f` after L2 in `model.py` — stack-internal norm here, model-external norm there.)

## 2. WindowedGQA (attention, one block)

- **Heads:** `head_dim = dim // n_heads` — H=512 → 64. Defaults n_heads=8 Q, n_kv_heads=2 KV (GQA ratio 4).
- **Projections:** all four bias-free `nn.Linear`: `q_proj` H→n_heads·hd (=H), `k_proj`/`v_proj` H→n_kv_heads·hd (=H/4 each), `o_proj` H→H.
- **RoPE:** applied to **q and k only**, shared base 1e5 (`rope_cache(head_dim, T, base)` → cos/sin; `apply_rope` = x·cos + rotate_half(x)·sin, rotate_half maps (a,b)→(−b,a)). No learned absolute positions.
- **GQA expansion:** `repeat_kv` broadcasts the 2 KV heads to 8 (each Q head reads its `head // n_kv_heads`-th KV head; never a shape mismatch).
- **Mask:** hand-built `window_mask(T, W)` — bool `[T,T]`, entry `(i,j)` true iff ==j ≤ i and i−j < W== (causal + sliding window in one arithmetic expression).
- **Attention:** `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=head_dim**-0.5, dropout_p=0.0)`.

## 3. SwiGLU (FFN, one block)

Three bias-free linears: `gate` H→I, `up` H→I, `down` I→H (I = `ffn_hidden`, default 2·H). Forward: `down(silu(gate(x)) * up(x))`. `down` is zero-initialized (§4).

## 4. TransformerStack (assembly + init)

- `nn.ModuleList` of N blocks + `norm_out` RMSNorm; stores `window`, `rope_base`, validates `dim % n_heads == 0`.
- **Zero-init:** `_reset_output_projections` zeros every `attn.o_proj.weight` and `ffn.down.weight` — at init each block is a ==residual identity== (attention/FFN contributions start at exactly 0), so early training cannot blow up the residual stream (tested).

## 5. Shapes table (H=512, 8Q/2KV, I=1024, head_dim=64)

| tensor | shape | notes |
|---|---|---|
| `x` in / `hT` out | `[B, T, 512]` | stack is shape-preserving |
| `q` | `[B, 8, T, 64]` | after view+transpose |
| `k`, `v` | `[B, 2, T, 64]` | n_kv_heads=2 |
| after `repeat_kv` | `[B, 8, T, 64]` | n_rep = 8//2 = 4 |
| mask | `[T, T]` bool | rebuilt per forward |
| FFN mid | `[B, T, 1024]` | gate·up after SiLU |
| RoPE cos/sin | `[T, 64]` each | one cache per forward, all blocks share |

Per-size dims (thesis): XS H=256/hd=32 · S H=320/hd=40 · M H=512/hd=64 — all 8Q+2KV, I=2H. `test_26b_dim_table_256_and_512` instantiates all three.

## 6. Params (shipped M, measured — matches `estimate_model_size`)

Per block = **2.5H² (GQA: q H² + k H²/4 + v H²/4 + o H²) + 3·H·I (SwiGLU) + 2H (norms)**:

| piece | formula | H=512, I=1024 |
|---|---|---|
| attn (q 262,144 + k/v 65,536×2 + o 262,144) | 2.5H² | 655,360 |
| FFN (gate+up 524,288×2 + down 524,288) | 3HI | 1,572,864 |
| 2× RMSNorm | 2H | 1,024 |
| **per block** | | **2,229,248 ≈ 2.229M** |
| **× N_T=12 (shipped)** | | **26,750,976 ≈ 26.75M** |

Thesis sizes: S per block ~0.9M ×5 ≈ 4.4M · M ~2.2M ×6 ≈ 13.4M; embeddings tied, separate (XS 8.4M · S 10.5M · M/shipped 16.777M at V=32K). Envelope check: shipped sandbox total = 16.777 + 26.75 + 6.56 (L2) + 0.0005 ≈ **50.09M**; baseline = **43.53M**.

## 7. Tests that prove it (`09-Code/tests/test_transformer.py`, 14 tests, CPU ~10s)

| test | proves |
|---|---|
| `test_shapes` | `[B,T,H]` in → `[B,T,H]` out |
| `test_default_init_is_residual_identity` | zero-init o_proj/down ⇒ finite identity output at start |
| `test_gradients_flow_through_every_block` | grads reach q/k/v/o + gate/up/down in every block |
| `test_mask_is_causal_and_windowed` | mask exact: `j ≤ i and i−j < W` for all (i,j) |
| `test_position_attends_only_inside_the_window` | row i sees exactly `i−W+1..i` |
| `test_future_tokens_do_not_change_the_past` | no leakage past a cut point |
| `test_window_ablation_changes_receptive_field` | W=8 vs W=32 reach differs as predicted |
| `test_kv_heads_are_repeated_not_mismatched` | `repeat_kv` mapping head→head//n_rep exact |
| `test_rope_is_reversible_by_rotation` | apply_rope(·, cos, −sin) inverts apply_rope(·, cos, sin) |
| `test_rotate_half_round_trip` | rotate_half² = −id |
| `test_rmsnorm_normalises_across_channels` | RMS = 1 across last dim |
| `test_swiglu_shape_and_grad` | SwiGLU shape-preserving, grads finite |
| `test_loss_decreases_over_training_steps` | 10 Adam steps reduce a regression loss |
| `test_26b_dim_table_256_and_512` | XS/S/M dim table (256/320/512) constructs and runs |

## Connections
- Spec: [02](../02-Architecture/02_System_Architecture.md) §4 · API: [API_Reference](../02-Architecture/API_Reference.md) · Prev build: [26a](26a_Internal_SSM_Build.md) · Next build: [26c](26c_Internal_Sandbox_Build.md) · Measured numbers: [14](../06-Training-Eval/14_Performance_Analysis.md) §1
