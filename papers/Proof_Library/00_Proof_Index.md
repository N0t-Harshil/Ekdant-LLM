# Proof Index — every load-bearing claim, its paper, its vault home

> Each row: vault claim → downloadable proof PDF (in this vault) → where the claim lives. Method papers (B/C/T) prove design choices; R-papers prove training/data/eval choices. Engineering rules (dedup thresholds, smoke-step counts) are marked as decisions, not theorems.

## L1 — selective SSM (design proofs in `../SSM_Storage/`)

| Claim (vault) | Proof | Lives in |
|---|---|---|
| Input-dependent Δ/B/C + scan beats static SSM | [`B5_Gu_Dao_2023_Mamba.pdf`](../SSM_Storage/B5_Gu_Dao_2023_Mamba.pdf) | [21](../../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) §2–3, [26a](../../04-Model-DeepDive/26a_Internal_SSM_Build.md) |
| Diagonal SSM keeps S4 power simply | [`B2_Gu_2022_S4D.pdf`](../SSM_Storage/B2_Gu_2022_S4D.pdf) | [21](../../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) §6 init |
| HiPPO init for long memory | [`B13_Gu_2020_HiPPO.pdf`](../SSM_Storage/B13_Gu_2020_HiPPO.pdf) | [26a](../../04-Model-DeepDive/26a_Internal_SSM_Build.md) §2 |
| Multi-channel states precedent (our 3 stores) | [`B4_Smith_2022_S5.pdf`](../SSM_Storage/B4_Smith_2022_S5.pdf), [`B6_Dao_Gu_2024_Mamba2_SSD.pdf`](../SSM_Storage/B6_Dao_Gu_2024_Mamba2_SSD.pdf) | [21](../../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) §3 |
| Pure SSM trails attention on recall (honesty) | [`B11_Lieber_2024_Jamba.pdf`](../SSM_Storage/B11_Lieber_2024_Jamba.pdf) | [02](../../02-Architecture/02_System_Architecture.md) §3–4, [10](../../07-Management/10_Known_Issues.md) T10 |
| Worked numbers (no code) | — | [28](../../04-Model-DeepDive/28_Worked_Examples.md) §1 |

## T — dense Transformer (design proofs in `../Transformer_Blocks/`)

| Claim (vault) | Proof | Lives in |
|---|---|---|
| Decoder block structure | [`T1_Vaswani_2017_Attention.pdf`](../Transformer_Blocks/T1_Vaswani_2017_Attention.pdf) | [02](../../02-Architecture/02_System_Architecture.md) §4 |
| Rotary positions | [`T2_Su_2021_RoPE.pdf`](../Transformer_Blocks/T2_Su_2021_RoPE.pdf) | [12](../../02-Architecture/12_Configuration_Guide.md), [26b](../../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) |
| KV sharing (our 8Q/2KV) | [`T3_Shazeer_2019_MQA.pdf`](../Transformer_Blocks/T3_Shazeer_2019_MQA.pdf), [`T4_Ainslie_2023_GQA.pdf`](../Transformer_Blocks/T4_Ainslie_2023_GQA.pdf) | [26b](../../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) §1 |
| SwiGLU FFN | [`T5_Shazeer_2020_GLU.pdf`](../Transformer_Blocks/T5_Shazeer_2020_GLU.pdf) | [26b](../../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) §1 |
| RMSNorm / pre-norm | [`T6_Zhang_2019_RMSNorm.pdf`](../Transformer_Blocks/T6_Zhang_2019_RMSNorm.pdf), [`T7_Xiong_2020_PreLN.pdf`](../Transformer_Blocks/T7_Xiong_2020_PreLN.pdf) | [26b](../../04-Model-DeepDive/26b_Dense_Transformer_for_T.md) §1 |
| Fixed window attention | [`T8_Jiang_2023_Mistral.pdf`](../Transformer_Blocks/T8_Jiang_2023_Mistral.pdf) | [02](../../02-Architecture/02_System_Architecture.md) §4 |
| RoPE extension (YaRN) | [`T9_Peng_2023_YaRN.pdf`](../Transformer_Blocks/T9_Peng_2023_YaRN.pdf) | [12](../../02-Architecture/12_Configuration_Guide.md) |
| Exact-attention efficiency bar | [`T10_Dao_2023_FlashAttention2.pdf`](../Transformer_Blocks/T10_Dao_2023_FlashAttention2.pdf) | [14](../../06-Training-Eval/14_Performance_Analysis.md) |

## L2 — sandbox (design proofs in `../Sandbox_Reasoning/`)

| Claim (vault) | Proof | Lives in |
|---|---|---|
| K=1 CoT control / voting baseline / debate blueprint / energy blueprint / cost honesty | C1 / C2 / C3 / C11 / C8 in `../Sandbox_Reasoning/` | [22](../../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) §2+§6 |
| Full 20-paper backing | C1–C20 | [20](../../03-Background/20_Thesis_Ch2_Sandbox_Papers.md) |
| Worked numbers (no code) | — | [28](../../04-Model-DeepDive/28_Worked_Examples.md) §2–3 |

## Training / data / eval (proofs in this folder)

| Claim (vault) | Proof | Lives in |
|---|---|---|
| AdamW fused, wd 0.1 | [`R1_Loshchilov_2017_AdamW.pdf`](R1_Loshchilov_2017_AdamW.pdf) | [07](../../06-Training-Eval/07_Training_Pipeline.md), [12](../../02-Architecture/12_Configuration_Guide.md) |
| Cosine schedule, warmup 3K | [`R2_Loshchilov_2016_SGDR.pdf`](R2_Loshchilov_2016_SGDR.pdf) | [07](../../06-Training-Eval/07_Training_Pipeline.md) |
| Token budgets (1.6B tokens vs sizes) | [`R3_Hoffmann_2022_Chinchilla.pdf`](R3_Hoffmann_2022_Chinchilla.pdf) | [12](../../02-Architecture/12_Configuration_Guide.md) |
| BPE tokenizer | [`R4_Sennrich_2015_BPE.pdf`](R4_Sennrich_2015_BPE.pdf) | [12](../../02-Architecture/12_Configuration_Guide.md) tokenizer § |
| Calibration as signal (agreement vs accuracy) | [`R5_Guo_2017_Calibration.pdf`](R5_Guo_2017_Calibration.pdf) | [22](../../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) §5–6 (RQ3) |
| DPO alignment stage | [`R6_Rafailov_2023_DPO.pdf`](R6_Rafailov_2023_DPO.pdf) | [07](../../06-Training-Eval/07_Training_Pipeline.md) stages |

## Decisions (engineering rules, no paper — stated as such)

Packing <5% padding · quality thresholds per category · 50-step smoke · checkpoint retention · effective batch 16 · sandbox_weight 0.1 (tune 0.05–0.3) · sandbox_rho_min 0.05 · GSM8K/HumanEval scaling-held-out · window 512. Each is a falsifiable default with an ablation, listed in [11](../../07-Management/11_Future_Work.md).
