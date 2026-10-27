# 20 — Thesis Ch. 2: Sandbox Papers (30 papers, L2)

> Use for Ch. 2 §C (why parallel-trajectory sandbox) + L2 ablations (K, rounds, energy vs. debate).
> Thesis stance: sandbox = in-graph multi-path reasoning with consensus; honest about cost (Smit et al. 2024).
>
> **Local library:** `papers/Sandbox_Reasoning/` — 30 PDFs (C1–C20, C21–C29). C1–C20 are the thesis argument; **C21–C28 are the novelty audit** ([29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md)) and must be cited beside the contribution, not buried in related work. Filename map:
> C1 [`C1_Wei_2022_Chain_of_Thought.pdf`](../papers/Sandbox_Reasoning/C1_Wei_2022_Chain_of_Thought.pdf), C2 [`C2_Wang_2022_Self_Consistency.pdf`](../papers/Sandbox_Reasoning/C2_Wang_2022_Self_Consistency.pdf),
> C3 [`C3_Du_2023_Multiagent_Debate.pdf`](../papers/Sandbox_Reasoning/C3_Du_2023_Multiagent_Debate.pdf), C4 [`C4_Yao_2023_Tree_of_Thoughts.pdf`](../papers/Sandbox_Reasoning/C4_Yao_2023_Tree_of_Thoughts.pdf),
> C5 [`C5_Yao_2022_ReAct.pdf`](../papers/Sandbox_Reasoning/C5_Yao_2022_ReAct.pdf), C6 [`C6_Shinn_2023_Reflexion.pdf`](../papers/Sandbox_Reasoning/C6_Shinn_2023_Reflexion.pdf), C7 [`C7_Madaan_2023_Self_Refine.pdf`](../papers/Sandbox_Reasoning/C7_Madaan_2023_Self_Refine.pdf),
> C8 [`C8_Smit_2024_Should_We_Be_Going_MAD.pdf`](../papers/Sandbox_Reasoning/C8_Smit_2024_Should_We_Be_Going_MAD.pdf), C9 [`C9_Wu_2025_Can_Agents_Really_Debate.pdf`](../papers/Sandbox_Reasoning/C9_Wu_2025_Can_Agents_Really_Debate.pdf),
> C10 [`C10_Yi_2026_Latent_Agents_ACL.pdf`](../papers/Sandbox_Reasoning/C10_Yi_2026_Latent_Agents_ACL.pdf), C11 [`C11_Energy_Calibration_Implicit_CoT_2025.pdf`](../papers/Sandbox_Reasoning/C11_Energy_Calibration_Implicit_CoT_2025.pdf),
> C12 [`C12_Lightman_2023_Lets_Verify_Step_by_Step.pdf`](../papers/Sandbox_Reasoning/C12_Lightman_2023_Lets_Verify_Step_by_Step.pdf) (C13–C20 have inline PDF lines below).
> Deep-dive: `22_Thesis_Ch3_Layer2_Sandbox.md`. Index: `papers/00_Papers_Index.md`.

## C1 — Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022.* arXiv:2201.11903.

- **Claim:** Intermediate steps unlock reasoning; single-path baseline every later method must beat.
- **Ekdant link:** L2 K=1 control = CoT-equivalent single path.
- **Thesis use:** Baseline definition; all sandbox gains measured vs. C1-style single path.

## C2 — Wang, X. et al. (2022). Self-Consistency Improves Chain of Thought Reasoning. *ICLR 2023.* arXiv:2203.11171.

- **Claim:** Sample K paths, majority vote beats greedy; hallucinations drop.
- **Ekdant link:** Direct ancestor of sandbox consensus (we move voting inside the graph + add refinement).
- **Thesis use:** Primary motivation for K>1; replicate majority-vote baseline before claiming debate/energy adds more.

## C3 — Du, Y. et al. (2023). Improving Factuality and Reasoning through Multiagent Debate. arXiv:2305.14325.

- **Claim:** Copies of one LLM propose → debate → revise over rounds; beats single-model CoT/reflection on reasoning + factuality; rounds + agents both matter.
- **Ekdant link:** **Primary debate blueprint** for `sandbox.py` debate rounds (cross-trajectory read + revise + consensus).
- **Thesis use:** Ch. 2 centerpiece for debate half of L2; implement rounds=0 vs. 2 vs. 3 comparison.

## C4 — Yao, S. et al. (2023). Tree of Thoughts: Deliberate Problem Solving with LLMs. *NeurIPS 2023.* arXiv:2305.10601.

- **Claim:** Tree search over thoughts (explore/evaluate/backtrack) beats linear CoT on planning puzzles.
- **Ekdant link:** Justifies keeping multiple live trajectories (not just resampling at the end).
- **Thesis use:** Cite for search-vs.-sampling framing; note thesis uses shallow debate, not full tree (scope honesty).

## C5 — Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting. *ICLR 2023.* arXiv:2210.03629.

- **Claim:** Interleaved reason+act traces improve grounded tasks.
- **Ekdant link:** Outside thesis scope by design (the model uses no external tools); cite to draw that boundary.
- **Thesis use:** Scope paragraph: reasoning without acting is the tested claim.

## C6 — Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023.* arXiv:2303.11366.

- **Claim:** Verbal self-reflection + episodic memory improves trial-to-trial decisions.
- **Ekdant link:** Reflection runs as sandbox refine steps in latent space.
- **Thesis use:** Reflection lives inside L2's refine steps — no separate module needed.

## C7 — Madaan, A. et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023.* arXiv:2303.17651.

- **Claim:** Generate → self-critique → refine loop improves quality without extra training.
- **Ekdant link:** Same loop as C6, different wording; supports `sandbox_steps` as iterative refine.
- **Thesis use:** Second vote for refine-loop; cite once, move on.

## C8 — Smit, A.P. et al. (2024). Should we be going MAD? A Look at Multi-Agent Debate Strategies. *ICML 2024.* PMLR 235.

- **Claim:** Current MAD does not reliably beat self-consistency/ensembling; sensitive to hyperparameters; tuned Multi-Persona can win.
- **Ekdant link:** **Honesty anchor.** Thesis must report cost (tokens × K) and tune comparison (SC vs. debate at equal budget).
- **Thesis use:** Limitations + eval protocol: always compare sandbox vs. self-consistency at matched FLOPs/tokens.

## C9 — Wu, H., Li, Z. & Li, L. (2025). Can LLM Agents Really Debate? arXiv:2511.07784.

- **Claim:** Controlled Knight-Knave-Spy study: intrinsic strength + diversity drive debate; order/confidence visibility matter little; majority pressure can suppress correction.
- **Ekdant link:** Design guidance: diversify trajectory init (not just order/visibility tweaks); watch for false consensus.
- **Thesis use:** Ch. 6 analysis: report overturn rate (wrong→right vs. right→wrong) per debate round.

## C10 — Yi, J.S.K., Mueller, A. & Lee, D. (2026). Latent Agents: Internalized Multi-Agent Debate. *ACL 2026.* 

- **Claim:** Two-stage fine-tune distills explicit debate into single-model latent subspaces; matches debate at 93% fewer tokens; agent directions steerable.
- **Ekdant link:** Future-work blueprint: distill K-path sandbox into K=1 student (thesis discussion chapter).
- **Thesis use:** Future work + efficiency defense: explicit sandbox now, internalization later.

## C11 — Energy-based calibration for implicit CoT (2025). arXiv:2511.07124 (Think Consistently, Reason Efficiently).

- **Claim:** EBM + Langevin dynamics calibrates latent thought tokens toward low-energy consistent regions; single calibrated CoT ≈ 10-path ensemble.
- **Ekdant link:** **Primary energy blueprint** for `sandbox_steps` (energy descent in latent space, not token space).
- **Thesis use:** Ch. 2 energy half of L2; energy calibration for implicit CoT — the descent-step prior. NOTE: a distance-to-*peer-mean* target is provably inert (see [29](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) §3.1); Ekdant anchors to `a_t = W_a·hT_t` instead.

## C12 — Lightman, H. et al. (2023). Let's Verify Step by Step. arXiv:2305.20050.

- **Claim:** Process supervision (per-step rewards) beats outcome supervision.
- **Ekdant link:** Supervision for sandbox: thesis uses cheap consensus-MSE now, process rewards later.
- **Thesis use:** Future work: replace MSE with learned step verifier. PDF: [`C12_Lightman_2023_Lets_Verify_Step_by_Step.pdf`](../papers/Sandbox_Reasoning/C12_Lightman_2023_Lets_Verify_Step_by_Step.pdf).

## C13 — Besta, M. et al. (2024). Graph of Thoughts. *AAAI 2024.* arXiv:2308.09687.

- **Claim:** Arbitrary graph (not chain/tree) over thought units with aggregation/refinement; beats ToT on sorting/document tasks at lower cost.
- **Ekdant link:** Justifies sandbox consensus as graph aggregation (K nodes → 1), not just voting.
- **Thesis use:** Design note: consensus = mean/lowest-energy aggregation. PDF: [`C13_Besta_2024_Graph_of_Thoughts.pdf`](../papers/Sandbox_Reasoning/C13_Besta_2024_Graph_of_Thoughts.pdf).

## C14 — Huang, J. et al. (2022). Large Language Models Can Self-Improve. *EMNLP 2023.* arXiv:2210.11610.

- **Claim:** Self-generated rationales + self-consistency filtering improve reasoning without labels.
- **Ekdant link:** Training-time counterpart to sandbox: high-consistency trajectories as SFT targets.
- **Thesis use:** Future training loop (filtered self-improve on sandbox outputs). PDF: [`C14_Huang_2022_Self_Improve.pdf`](../papers/Sandbox_Reasoning/C14_Huang_2022_Self_Improve.pdf).

## C15 — Zelikman, E. et al. (2022). STaR: Bootstrapping Reasoning with Reasoning. *NeurIPS 2022.* arXiv:2203.14465.

- **Claim:** Rationalize-then-finetune loop (answer-hinted rationales) grows reasoning ability iteratively.
- **Ekdant link:** Same loop as C14 with hinting; thesis SFT stage can adopt STaR-style filtering on K paths.
- **Thesis use:** SFT design citation. PDF: [`C15_Zelikman_2022_STaR.pdf`](../papers/Sandbox_Reasoning/C15_Zelikman_2022_STaR.pdf).

## C16 — Uesato, J. et al. (2022). Solving Math Word Problems with Process- and Outcome-Based Feedback. arXiv:2211.14275.

- **Claim:** Process-based evaluator (per-step) helps reranking more than outcome-only on GSM8K.
- **Ekdant link:** Evidence for per-step sandbox energy over final-answer vote.
- **Thesis use:** Eval: rerank K trajectories by step-energy, not just final consensus. PDF: [`C16_Uesato_2022_Process_Feedback.pdf`](../papers/Sandbox_Reasoning/C16_Uesato_2022_Process_Feedback.pdf).

## C17 — Zhou, D. et al. (2022). Least-to-Most Prompting. *ICLR 2023.* arXiv:2205.10625.

- **Claim:** Decompose-then-solve (easy→hard subproblems) beats CoT on compositional tasks (SCAN, GSM8K).
- **Ekdant link:** Decomposition without a planner module: each sandbox trajectory can take a different decomposition.
- **Thesis use:** Prompting protocol for eval; cite as planner-free decomposition. PDF: [`C17_Zhou_2022_Least_to_Most.pdf`](../papers/Sandbox_Reasoning/C17_Zhou_2022_Least_to_Most.pdf).

## C18 — Chan, C.M. et al. (2023). ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate. arXiv:2308.07201.

- **Claim:** Multi-role debate among evaluators improves judgment quality over single judge.
- **Ekdant link:** Debate for evaluation, not generation — thesis adopts for human-eval protocol (debate-judge on sandbox outputs).
- **Thesis use:** Ch. 6 eval design. PDF: [`C18_Chan_2023_ChatEval.pdf`](../papers/Sandbox_Reasoning/C18_Chan_2023_ChatEval.pdf).

## C19 — Liang, T. et al. (2023). Encouraging Divergent Thinking in LLMs through Multi-Agent Debate. arXiv:2305.19118.

- **Claim:** Explicitly divergent personas + debate improve creative/factual trade-off vs. uniform agents.
- **Ekdant link:** Direct answer to C9's diversity finding: initialize K trajectories with diverse projections, not copies.
- **Thesis use:** Implementation: diverse init in `sandbox.py`. PDF: [`C19_Liang_2023_Divergent_Debate.pdf`](../papers/Sandbox_Reasoning/C19_Liang_2023_Divergent_Debate.pdf).

## C20 — Wan, G. et al. (2025). Reasoning-Aware Self-Consistency (RASC). *NAACL 2025.* ACL Anthology 2025.naacl-long.184.

- **Claim:** Score rationales (not just answers) to stop sampling early with equal accuracy at lower cost.
- **Ekdant link:** Cost control for sandbox: rationale-aware early-stop (stop refining agreed trajectories).
- **Thesis use:** Efficiency protocol + cost tables. PDF: [`C20_Wan_2025_RASC.pdf`](../papers/Sandbox_Reasoning/C20_Wan_2025_RASC.pdf).

---

## C21–C28 — Prior-art audit (added by the L2 novelty check)

> Every ingredient of L2 has prior art. C21–C28 are the closest ancestors and the
> evidence that forced the spread floor. Full comparison table:
> [29_Prior_Art_and_Novelty_Check.md](../04-Model-DeepDive/29_Prior_Art_and_Novelty_Check.md) · §1.

| Tag | Paper | What it already does | PDF |
|---|---|---|---|
| C21 | Burtsev & Rumshisky 2021, arXiv:2107.10342 | Parallel hypothesis **streams inside one model**, merged at the end — closest ancestor | [`C21_Burtsev_2021_Multi_Stream_Transformers.pdf`](../papers/Sandbox_Reasoning/C21_Burtsev_2021_Multi_Stream_Transformers.pdf) |
| C22 | Shazeer et al. 2020, arXiv:2003.02436 | Talking-heads: projections across the head dimension so heads read **each other** | [`C22_Shazeer_2020_Talking_Heads_Attention.pdf`](../papers/Sandbox_Reasoning/C22_Shazeer_2020_Talking_Heads_Attention.pdf) |
| C23 | Kerce & Fox 2026, arXiv:2603.07461 | Residual stream split into function-separated components, measured at **29M params** | [`C23_Kerce_2026_Dual_Stream_Transformer.pdf`](../papers/Sandbox_Reasoning/C23_Kerce_2026_Dual_Stream_Transformer.pdf) |
| C24 | Rauba, Fanconi & van der Schaar 2026, Tiny Autoregressive Recursive Models, arXiv:2603.08082 (OpenReview `ikNCYypbT9`) | Compute-matched ladder: single-stream / two-stream / nested placement, weighted readout `Σ_k w_k H_k`. **Result: untied depth and a flat two-stream baseline generalize best; the full Autoregressive TRM shows no reliable gain** | [`C24_Rauba_2026_Tiny_Autoregressive_Recursive_Models.pdf`](../papers/Sandbox_Reasoning/C24_Rauba_2026_Tiny_Autoregressive_Recursive_Models.pdf) |
| C25 | Energy Transformer, NeurIPS 2023 | Iterate a block to descend a global energy to a fixed point | [`C25_Energy_Transformer_NeurIPS2023.pdf`](../papers/Sandbox_Reasoning/C25_Energy_Transformer_NeurIPS2023.pdf) |
| C26 | NRGPT, arXiv:2512.16762 | Fixed-count gradient steps on a **parameterized** energy as the forward pass | [`C26_NRGPT_Energy_Alternative_for_GPT.pdf`](../papers/Sandbox_Reasoning/C26_NRGPT_Energy_Alternative_for_GPT.pdf) |
| C27 | Wang et al. 2025, Mixture-of-Agents, ICLR 2025 | Layered proposal/aggregation over full agent responses | [`C27_Wang_2025_Mixture_of_Agents_ICLR.pdf`](../papers/Sandbox_Reasoning/C27_Wang_2025_Mixture_of_Agents_ICLR.pdf) |
| C28 | Li et al. 2026, arXiv:2510.20817 + ACL 2026 Findings 1266 | KL-regularized RL **is designed to mode-collapse**; entropy collapse in RLVR — why a bare agreement loss cannot be used | [`C28a_Li_2026_KL_Regularized_RL_Mode_Collapse.pdf`](../papers/Sandbox_Reasoning/C28a_Li_2026_KL_Regularized_RL_Mode_Collapse.pdf), [`C28b_ACL2026_Entropy_Collapse_RLVR.pdf`](../papers/Sandbox_Reasoning/C28b_ACL2026_Entropy_Collapse_RLVR.pdf) |

| C29 | Jolicoeur-Martineau 2025, Less is More: Recursive Reasoning with Tiny Networks, arXiv:2510.04871 | The original TRM: a 7M-parameter network recursing on itself, updating answer `y` and latent `z` — the object C24 extends | [`C29_JolicoeurMartineau_2025_Less_is_More_TRM.pdf`](../papers/Sandbox_Reasoning/C29_JolicoeurMartineau_2025_Less_is_More_TRM.pdf) |

**Consequence for the thesis claim.** Do **not** claim that multi-hypothesis in-model reasoning is new.
Claim the *combination*: multi-trajectory tensor-level debate placed **after** a dense backbone, with a
consensus stage that is independent of the SSM memory (T38; an earlier draft wrote it back into an SSM store), at 12–50M parameters trained
from scratch. Name C21 and C25/C26 as the closest ancestors and state the differences explicitly.

## How to cite this section (template)

> Single-path CoT (Wei et al. 2022) is the K=1 control; self-consistency (Wang et al. 2022) is the matched-budget baseline. Sandbox debate follows Du et al. (2023) with ToT-style parallel search (Yao et al. 2023), folded reflection (Shinn et al. 2023; Madaan et al. 2023), energy descent in latent space (2025 EBM calibration), and process-supervision outlook (Lightman et al. 2023), evaluated under the cost-aware critique (Smit et al. 2024) with debate-dynamics checks (Wu et al. 2025) and internalization as future work (Yi et al. 2026). Acting with tools is outside the thesis scope (ReAct: Yao et al. 2022).
