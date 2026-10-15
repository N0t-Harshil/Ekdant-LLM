# 25 — Thesis Ch. 2: Transformer Papers (10 papers, T)

> Use for Ch. 2 §D (why a dense Transformer stack is the right backbone) + T-layer defense.
> Order = bottom-up (normalization → attention → positions → FFN → efficient variants).
> PDFs: fetch into `papers/Transformer_Blocks/` per the table in §11 (not yet in library — SSM/Sandbox PDFs are).

## T1 — Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017.* arXiv:1706.03762.

- **Claim:** Multi-head self-attention + residual + norm blocks replace recurrence; the block our T copies in miniature.
- **Ekdant link:** T is literally this block (pre-norm variant) with a fixed window and GQA — every deviation from T1 is ablated, not assumed.
- **Thesis use:** Ch.2 opening: the standard being adapted; Ch.3: T block diagram annotated vs. T1.

## T2 — Su, J. et al. (2021). RoFormer: Rotary Position Embedding. arXiv:2104.09864.

- **Claim:** Rotate Q/K pairs by position — relative geometry without learned tables, extrapolates.
- **Ekdant link:** The single shared position scheme (L1 + T, base 100K scaled for small head dims).
- **Thesis use:** Justify one RoPE base everywhere instead of per-layer schemes.

## T3 — Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need (MQA). arXiv:1911.02150.

- **Claim:** One shared KV head decodes as well as many, far cheaper — the ancestor of all KV sharing.
- **Ekdant link:** Grandparent of our 8Q/2KV GQA; the memory argument in its purest form.
- **Thesis use:** One-line lineage before T4.

## T4 — Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models. arXiv:2305.13245.

- **Claim:** Grouped-query attention spans the quality/memory curve between MHA and MQA; uptraining conversion works.
- **Ekdant link:** Direct blueprint for T's 8Q/2KV heads (dims 32/40/64) — KV stays small at every size.
- **Thesis use:** Ch.2 §D centerpiece for attention choice; cite dims table from `12`.

## T5 — Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.

- **Claim:** Gated linear units (incl. SwiGLU/GEGLU) beat vanilla ReLU MLPs at fixed params.
- **Ekdant link:** Why T's FFN is dense SwiGLU (I=2H), not ReLU — free capacity, same budget.
- **Thesis use:** One paragraph; FFN dim choice cites this.

## T6 — Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS 2019.* arXiv:1910.07467.

- **Claim:** RMS without mean-centering matches LayerNorm, cheaper — the small-model default.
- **Ekdant link:** Every T/L1/L2 norm in the thesis; nothing fancier is needed at 10–40M.
- **Thesis use:** One line in implementation details.

## T7 — Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture (Pre-LN). *ICML 2020.* arXiv:2002.04745.

- **Claim:** Pre-LN placement stabilizes deep training (no warm-up surgery); post-LN needs care.
- **Ekdant link:** Why every T block is pre-norm → sublayer → residual, no exceptions.
- **Thesis use:** Training-stability paragraph alongside the NaN guide (`13` §6).

## T8 — Jiang, A.Q. et al. (2023). Mistral 7B (sliding-window attention). arXiv:2310.06825.

- **Claim:** Fixed sliding window + rolling KV cache matches full attention on most tasks at a fraction of memory.
- **Ekdant link:** Direct precedent for T's W=512 window — bounded KV with dense mechanics, no custom kernels.
- **Thesis use:** The citation that makes our window choice frontier-normal, not a hack.

## T9 — Peng, B. et al. (2023). YaRN: Efficient Context Window Extension. arXiv:2309.00071.

- **Claim:** NTK-aware RoPE rescaling extends context far beyond training length with brief finetunes.
- **Ekdant link:** The extension mechanism behind our YaRN note (`12`); 4096-window models can stretch at eval.
- **Thesis use:** Reproducibility appendix: exact YaRN factors used, if any.

## T10 — Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism. arXiv:2307.7212.

- **Claim:** IO-aware exact attention (tiling + parallelism) — the efficiency bar and the algorithm our SDPA call inherits.
- **Ekdant link:** Honesty anchor: we claim parity of *math*, not speed — PyTorch-core SDPA, no custom kernels per the internal rule.
- **Thesis use:** Ch.6 cost tables compare tok/s against this lineage openly.

## How to cite this section (template)

> T follows the standard decoder block (Vaswani et al. 2017) in pre-norm form (Xiong et al. 2020; Zhang & Sennrich 2019), with rotary positions (Su et al. 2021, extended per Peng et al. 2023), grouped queries (Shazeer 2019; Ainslie et al. 2023), gated feed-forward (Shazeer 2020), a Mistral-style fixed window (Jiang et al. 2023) computed with stock SDMA math in the FlashAttention lineage (Dao 2023) — adapted to XS/S/M dims, with MoE deliberately omitted since the sandbox debates instead.

## §11. Library (fetched — [`papers/Transformer_Blocks/`](../papers/Transformer_Blocks/) + [`index`](../papers/00_Papers_Index.md) Folder 3)

| # | Filename to save under `papers/Transformer_Blocks/` | arXiv |
|---|---|---|
| T1 | `T1_Vaswani_2017_Attention.pdf` | 1706.03762 |
| T2 | `T2_Su_2021_RoPE.pdf` | 2104.09864 |
| T3 | `T3_Shazeer_2019_MQA.pdf` | 1911.02150 |
| T4 | `T4_Ainslie_2023_GQA.pdf` | 2305.13245 |
| T5 | `T5_Shazeer_2020_GLU.pdf` | 2002.05202 |
| T6 | `T6_Zhang_2019_RMSNorm.pdf` | 1910.07467 |
| T7 | `T7_Xiong_2020_PreLN.pdf` | 2002.04745 |
| T8 | `T8_Jiang_2023_Mistral.pdf` | 2310.06825 |
| T9 | `T9_Peng_2023_YaRN.pdf` | 2309.00071 |
| T10 | `T10_Dao_2023_FlashAttention2.pdf` | 2307.7212 |
