# 19 — Thesis Ch. 2: SSM Storage Papers (20 papers, L1)

> Use for Ch. 2 §B (why selective SSM for memory) + L1 O(1) proof + backbone ablation (SSM vs. Transformer-tiny).
> Order = pedagogical (foundations → selective → hybrids → surveys). All arXiv IDs checked Sep 2026.
>
> **Local library:** `papers/SSM_Storage/` — 20 PDFs. Filename map:
> B1 [`B1_Gu_2021_S4.pdf`](../papers/SSM_Storage/B1_Gu_2021_S4.pdf), B2 [`B2_Gu_2022_S4D.pdf`](../papers/SSM_Storage/B2_Gu_2022_S4D.pdf), B3 [`B3_Gupta_2022_DSS.pdf`](../papers/SSM_Storage/B3_Gupta_2022_DSS.pdf),
> B4 [`B4_Smith_2022_S5.pdf`](../papers/SSM_Storage/B4_Smith_2022_S5.pdf), B5 [`B5_Gu_Dao_2023_Mamba.pdf`](../papers/SSM_Storage/B5_Gu_Dao_2023_Mamba.pdf), B6 [`B6_Dao_Gu_2024_Mamba2_SSD.pdf`](../papers/SSM_Storage/B6_Dao_Gu_2024_Mamba2_SSD.pdf),
> B7 [`B7_Poli_2023_Hyena.pdf`](../papers/SSM_Storage/B7_Poli_2023_Hyena.pdf), B8 [`B8_Fu_2022_H3.pdf`](../papers/SSM_Storage/B8_Fu_2022_H3.pdf), B9 [`B9_Sun_2023_RetNet.pdf`](../papers/SSM_Storage/B9_Sun_2023_RetNet.pdf),
> B10 [`B10_Peng_2023_RWKV.pdf`](../papers/SSM_Storage/B10_Peng_2023_RWKV.pdf), B11 [`B11_Lieber_2024_Jamba.pdf`](../papers/SSM_Storage/B11_Lieber_2024_Jamba.pdf),
> B12a [`B12a_Somvanshi_2025_S4_to_Mamba_Survey.pdf`](../papers/SSM_Storage/B12a_Somvanshi_2025_S4_to_Mamba_Survey.pdf) (B12b Patro closed-access, no PDF; B13–B20 inline below).
> Each entry below ends with its local PDF path. Deep-dive: `21_Thesis_Ch3_Layer1_SSM_Storage.md`.

## B1 — Gu, A., Goel, K. & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces (S4). *ICLR 2022.* arXiv:2111.00396.

- **Claim:** NPLR-parameterized SSM + HiPPO init models 16K-length sequences; dual recurrent/convolutional views.
- **Ekdant link:** Foundation of L1 storage: fixed-state recurrence with long-range memory.
- **Thesis use:** Derive L1 equations from S4; cite HiPPO for long-context motivation.

## B2 — Gu, A. et al. (2022). On the Parameterization and Initialization of Diagonal State Spaces (S4D). *NeurIPS 2022.* arXiv:2206.11893.

- **Claim:** Diagonal SSM matches S4 with simpler init/implementation.
- **Ekdant link:** Justifies thesis implementation choice (diagonal selective block, not full NPLR).
- **Thesis use:** Implementation footnote: why diagonal.

## B3 — Gupta, A., Gu, A. & Berant, J. (2022). Diagonal State Spaces (DSS). arXiv:2203.14343.

- **Claim:** DSS confirms diagonal restriction preserves Long Range Arena performance.
- **Ekdant link:** Second vote for diagonal simplicity in L1.
- **Thesis use:** One-line supporting cite for B2.

## B4 — Smith, J.T.H., Warrington, A. & Linderman, S. (2022). Simplified State Space Layers for Sequence Modeling (S5). *ICLR 2023.* arXiv:2208.04933.

- **Claim:** MIMO SSM, fewer hyperparameters, parallel scan; matches Transformer with fewer resources.
- **Ekdant link:** Thesis efficiency claim ancestor (fewer knobs, parallel training).
- **Thesis use:** Cite for `n_ssm_layers` scaling + parallel-scan training.

## B5 — Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

- **Claim:** Input-dependent Δ/B/C (selection) + hardware-aware scan; beats S4/H3/Hyena on copying/induction/scaling to 1.3B; extrapolates 4000× length.
- **Ekdant link:** **Primary L1 blueprint.** Copy selection mechanism + scan; report copying/induction sanity checks in thesis.
- **Thesis use:** Ch. 2 centerpiece; replicate Table 1 (selective copying) as L1 unit test; cite code github.com/state-spaces/mamba.

## B6 — Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality (Mamba-2). *ICML 2024.* arXiv:2405.21060.

- **Claim:** SSD framework unifies attention and SSM; Mamba-2 (2–8× faster) with multi-head selective SSM.
- **Ekdant link:** Justifies thesis framing (not anti-attention, but duality) + optional Mamba-2 block upgrade post-thesis.
- **Thesis use:** Discussion: SSD view; future work (multi-head SSM).

## B7 — Poli, M. et al. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. *ICML 2023.* arXiv:2302.10866.

- **Claim:** MLP-parameterized long convolutions match attention at scale with subquadratic cost.
- **Ekdant link:** Alternative subquadratic baseline; thesis should name Hyena as considered-and-deferred alternative.
- **Thesis use:** Related work one paragraph; why selective SSM chosen over Hyena (input-dependence).

## B8 — Fu, D.Y. et al. (2022). Hungry Hungry Hippos: Towards Language Modeling with State Space Models (H3). *ICLR 2023.* arXiv:2208.01814.

- **Claim:** H3 closes gap on induction/copying with shift-SSM + gating; FlashConv.
- **Ekdant link:** Ablation inspiration: gate vs. selection (thesis can cite H3 as gate-only baseline).
- **Thesis use:** Background: what selection adds over gating (Mamba Table 1).

## B9 — Sun, Y. et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models (RetNet). arXiv:2307.08621.

- **Claim:** Retention (recurrent + parallel + chunkwise) with O(1) inference; strong scaling.
- **Ekdant link:** Second O(1)-inference family; good thesis baseline citation (recurrent inference without KV growth).
- **Thesis use:** Related work: RetNet vs. SSM storage trade-offs (exact recall vs. compression).

## B10 — Peng, B. et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. *EMNLP 2023.* arXiv:2305.13048.

- **Claim:** Receptance-weighted key-value RNN trains like Transformer, infers like RNN.
- **Ekdant link:** Shows RNN revival is a family (thesis positions SSM within it).
- **Thesis use:** One-line family cite; no implementation needed.

## B11 — Lieber, O. et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model. arXiv:2403.19887.

- **Claim:** Interleaved Transformer+Mamba+MoE balances recall and efficiency at 52B scale.
- **Ekdant link:** Honest counterpoint: pure SSM weakness on exact recall/associative memory (thesis must admit; see surveys).
- **Thesis use:** Limitations: when hybrids beat pure SSM; why thesis stays pure (falsifiability) + future hybrid.

## B12 — Surveys (read 1, skim 1)

- **Somvanshi, S. et al. (2025). Advancing Intelligent Sequence Modeling: S4 to Mamba. arXiv:2503.18970.** Lineage S4→S4D/DSS→S5→Mamba/Mamba-2 + hybrids; efficiency vs. retrieval trade-off summary. **Thesis Table source** for SSM family tree. PDF: [`B12a_Somvanshi_2025_S4_to_Mamba_Survey.pdf`](../papers/SSM_Storage/B12a_Somvanshi_2025_S4_to_Mamba_Survey.pdf).
- **Patro, B.N. et al. (2025). Mamba-360: Survey of SSMs as Transformer Alternatives. *Engineering Applications of AI.*** Gating/structural/recurrent taxonomy + 100+ applications. No open PDF — cite via DOI, see `papers/SSM_Storage/SSM_Storage_Index.md` placeholder note.

## B13 — Gu, A. et al. (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. *NeurIPS 2020.* arXiv:2008.07669.

- **Claim:** Optimal online function approximation (Legendre/Laguerre bases) for long-range memory; the init S4 inherits.
- **Ekdant link:** Mathematical root of L1: why SSM state compresses without learned forgetting first.
- **Thesis use:** Ch. 2 foundations, one derivation paragraph. PDF: [`B13_Gu_2020_HiPPO.pdf`](../papers/SSM_Storage/B13_Gu_2020_HiPPO.pdf).

## B14 — Tay, Y. et al. (2020). Long Range Arena. *ICLR 2021.* arXiv:2011.04006.

- **Claim:** 6-task long-context benchmark (1K–16K) that made SSM progress measurable.
- **Ekdant link:** Every B1–B5 paper reports LRA; thesis L1 unit tests mirror ListOps/Text/Retrieval splits.
- **Thesis use:** Eval protocol justification. PDF: [`B14_Tay_2020_Long_Range_Arena.pdf`](../papers/SSM_Storage/B14_Tay_2020_Long_Range_Arena.pdf).

## B15 — Nguyen, E. et al. (2022). S4ND: Modeling Images and Videos as Multidimensional Signals with State Spaces. *NeurIPS 2022.* arXiv:2205.13493.

- **Claim:** Multi-dimensional SSM (PDE view) extends S4 beyond 1-D sequences.
- **Ekdant link:** Shows SSM storage generalizes beyond text (thesis future-work hook, no implementation).
- **Thesis use:** One-line related work. PDF: [`B15_Nguyen_2022_S4ND.pdf`](../papers/SSM_Storage/B15_Nguyen_2022_S4ND.pdf).

## B16 — Hasani, R. et al. (2022). Liquid Structural State-Space Models (Liquid-S4). *AAAI 2023.* arXiv:2209.12951.

- **Claim:** Input-dependent state transition (liquid time-constant) inside S4 improves generalization.
- **Ekdant link:** Conceptual bridge S4 → selection (Mamba): input-dependence matters before full selectivity.
- **Thesis use:** Lineage paragraph between B1 and B5. PDF: [`B16_Hasani_2022_Liquid_S4.pdf`](../papers/SSM_Storage/B16_Hasani_2022_Liquid_S4.pdf).

## B17 — De, S. et al. (2024). Griffin: Mixing Gated Linear Recurrences with Local Attention. *ICML 2024.* arXiv:2402.19427.

- **Claim:** Gated linear RNN + local attention matches strong baselines with O(1) cache; Hawk variant pure-RNN.
- **Ekdant link:** Strongest non-Mamba recurrent baseline; thesis names Griffin as considered alternative to selective SSM.
- **Thesis use:** Related-work comparison table. PDF: [`B17_De_2024_Griffin.pdf`](../papers/SSM_Storage/B17_De_2024_Griffin.pdf).

## B18 — Wang, J. et al. (2024). MambaByte: Token-free Selective State Space Model. arXiv:2401.13660.

- **Claim:** Selective SSM directly on bytes matches subword Transformers with smaller footprint.
- **Ekdant link:** Proves L1 storage is tokenizer-agnostic (thesis can keep BPE without apology, byte-level as future).
- **Thesis use:** Footnote + future work. PDF: [`B18_Wang_2024_MambaByte.pdf`](../papers/SSM_Storage/B18_Wang_2024_MambaByte.pdf).

## B19 — Beck, M. et al. (2024). xLSTM: Extended Long Short-Term Memory. *NeurIPS 2024.* arXiv:2405.04517.

- **Claim:** Modernized LSTM (exponential gating, matrix memory) scales to LLM sizes.
- **Ekdant link:** RNN-revival family member; thesis positions SSM among xLSTM/Griffin/RWKV alternatives and states selection criterion (associative scan + HiPPO init).
- **Thesis use:** Family-context paragraph. PDF: [`B19_Beck_2024_xLSTM.pdf`](../papers/SSM_Storage/B19_Beck_2024_xLSTM.pdf).

## B20 — Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention. *NeurIPS 2022.* arXiv:2205.14135.

- **Claim:** IO-aware exact attention (tiling, recomputation) — the efficiency bar SSMs must beat.
- **Ekdant link:** Fair-comparison anchor: thesis reports tok/s and memory vs. FlashAttention-2 baseline, not naive attention.
- **Thesis use:** Ch. 6 baselines + cost tables. PDF: [`B20_Dao_2022_FlashAttention.pdf`](../papers/SSM_Storage/B20_Dao_2022_FlashAttention.pdf).

## How to cite this section (template)

> L1 storage follows the selective-SSM line (S4: Gu et al. 2021; diagonal: Gu et al. 2022; S5: Smith et al. 2022; selection + scan: Gu & Dao 2023; duality: Dao & Gu 2024), with Hyena (Poli et al. 2023), H3 (Fu et al. 2022), RetNet (Sun et al. 2023) and RWKV (Peng et al. 2023) as family context and Jamba (Lieber et al. 2024) plus recent surveys (Somvanshi et al. 2025; Patro et al. 2025) marking the known retrieval limitation the thesis acknowledges.
