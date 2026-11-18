# Ekdant v0.1.0 — sandbox v0.1.0 and the text-generation protocol: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phases 1–2 of the spec for **Ekdant v0.1.0**: the sandbox layer v0.1.0 (the original state + four role generators R1–R4, one communication round, a learned soft selection gate) at any configurable depth, plus the new text-generation protocol (stage 1 on medical reasoning text, benchmark MedLFQA with factuality scores) — all tested on CPU, ready for the Phase 3/4 GPU runs.

**Versions (author, 2026-11-18):** v0.0.1 = v1 (from scratch), v0.0.2–v0.0.5 = v2–v5 (v0.0.5 = `configs/v5_final.json`); **v0.1.0 = this work**; v1.0.0 = when it is shown to work. The layer keeps the name **sandbox**.

**Architecture:** `sandbox.py` gets `SandboxV01` (the sandbox, v0.1.0) next to the unchanged `Sandbox` (v0.0.x); a factory `build_sandbox` picks one from `V2Config.sandbox_version`. `GatedAddOn` keeps hooks, norms, the `enabled` switch and the KV cache; for v0.1.0 it adds the selected change Σᵢ pᵢΔᵢ (Δ₀ = 0 for h) sized to the stream. Data and benchmark: new loaders in `gen_data.py`, an NLI judge and factuality scores in `gen_eval.py`, wired into `train_gen.py` behind flags whose defaults keep the old protocol.

**Tech Stack:** PyTorch ≥ 2.2, `transformers` < 5, `datasets`; SmolLM2-135M (`tiny_llama()` in tests); NLI judge `pritamdeka/PubMedBERT-MNLI-MedNLI` (approved by the author 2026-11-18, scoring only, never trained).

**Spec:** [30_L2_v6_Roles_Spec.md](30_L2_v6_Roles_Spec.md) (with the decisions table).

## Global Constraints

- RULES.md in full: every arm equals the backbone at step 0; causal, per-row forward; KV-cache decoding equals a full forward; no new files under `src/ekdant/`; only add (never rename/remove) checkpoint and `results.json` keys; tests CPU-only, tiny, seeded; no commit unless the author asks.
- Defaults reproduce v0.0.5 and the old protocol exactly: `sandbox_version = "0.0.5"`, `--stage1 pubmedqa`, no MedLFQA test unless `--medlfqa`.
- Layer convention unchanged: `addon_layer = k` = after decoder layer k (1-based); −1 = after the final norm.
- v0.1.0 scope: soft selection only, one insertion point (`addon_layers` rejected), no auxiliary losses.
- The multiple-choice code stays (RULES §0.9) but is no longer a benchmark of v0.1.0.
- Every number in the docs comes from code or a run (`addon_params`, loader counts); nothing estimated.
- Architecture docs explain v0.1.0 twice: in plain words and technically.

## Review Focus

1. Right-padded rows: diagnostics count real tokens only — Task 3 `test_sandbox01_diagnostics_ignore_padding`.
2. Massive mid-stream activations: change sized to the stream's rms — Task 2 `test_sandbox01_mid_change_is_sized_to_the_stream`.
3. Cached generation with the context-reading roles — Task 2 `test_sandbox01_kv_cache_step_matches_full_forward`.
4. Reasoning rows longer than `max_answer`: dropped and counted, never cut (a cut row loses its final answer) — Task 4 `test_reasoning_rows_too_long_are_dropped_not_cut`.
5. A benchmark question that also appears in training data: removed from every training set — Task 4 `test_medlfqa_questions_never_reach_training`.

---

### Task 1: `SandboxV01` — generators, communication round, selection gate

**Files:** modify `09-Code/src/ekdant/sandbox.py`; test `09-Code/tests/test_sandbox_v01.py` (new).

**Produces:**
- `CANDIDATES = ("h", "R1_explore", "R2_consistency", "R3_uncertainty", "R4_supported")`
- `class SandboxV01(nn.Module)`: `__init__(dim, heads=0, hidden_ratio=4.0, rounds=1, gate_hidden=256)`; `forward(hT, mask=None, attn_mask=None, ctx=None, passes=None) -> SandboxOutput` with `consensus` = Σ pᵢΔᵢ [B,T,D], `trajectories` = the five candidates [5,B,T,D], `features` = p [B,T,5] float32; attributes `force: Optional[int]`, `debate_passes = 1`, `steps = 0`, `draw_passes() -> None`; `role_parameters() -> Dict[str, List[Parameter]]` keyed by the four roles, `"gate"`, `"comm"`.

**Design:** memory as `Sandbox.memory`. Each generator `Linear → SiLU → Linear(H→D)`, H = D·hidden_ratio, last layer zero-init. R1: input x, output projected orthogonal to x. R2: `[x, read2]`, own causal query over the memory. R4: `[x, read4, Δ1, Δ2]`, own query. R3: `[x, spread]`, spread = per-channel std of {Δ1, Δ2, Δ4}. Round (×rounds, own weights): candidates → RMSNorm → `CrossRead` (never itself) → `Δi += σ(Wγ,i[norm Ci; cross_i]) ⊙ Wo,i cross_i` (Wo zero-init, Wγ bias −2); h never updated. Gate: `s_i = MLP_g([norm Ci; x; history])` (3D → gate_hidden → 1, last layer zero-init), softmax in float32; `force = i` → one-hot.

- [ ] Tests (D 16, heads 2, B 2, T 5): `test_sandbox01_is_a_no_op_at_init` · `test_sandbox01_shapes` · `test_sandbox01_gate_weights_are_float32_and_normalised` (0.2 at init) · `test_original_state_is_candidate_zero` · `test_r1_change_is_orthogonal_to_the_state` (rounds 0) · `test_four_roles_produce_different_candidates` (pairwise cos < 0.99) · `test_gradients_reach_every_role_and_the_gate` · `test_sandbox01_is_causal` · `test_forced_candidate_gives_one_hot_weights`.
- [ ] Run → fail; implement; run → pass.

### Task 2: config, factory, integration

**Files:** `sandbox.py` (`build_sandbox`), `model.py` (`V2Config`, `GatedAddOn`, `MultiAddOn`, `EkdantLM`), `train_gen.py` (re-validate after the variant sets `addon`), new `configs/ekdant_v0.1.0.json`; tests in `test_sandbox_v01.py`.

**Produces:** `build_sandbox(dim, version="0.0.5", **kw)`; `V2Config.sandbox_version = "0.0.5"`, `sandbox_rounds_v01 = 1`, `sandbox_hidden_ratio_v01 = 4.0`, `sandbox_gate_hidden_v01 = 256`, `sandbox_select = "soft"`; `sandbox_kwargs()` adds the v0.1.0 keys only for version 0.1.0; `validate()` rejects unknown versions, `sandbox_select != "soft"`, v0.1.0 with `addon_layers`, `sandbox_confidence`, `sandbox_aligned`, `sandbox_train_passes > 1`, `gate_force`, or rounds < 0. `GatedAddOn` v0.1.0 branch: `h + consensus · rms(h).detach()` (x = in_norm(h) mid, h at the end); `EkdantLM.force_candidate(i)`; `L2Cache` also for v0.1.0. `configs/ekdant_v0.1.0.json` = v5's keys **without `addon`** (so both arms use the one file, `--variant` decides) + `sandbox_version "0.1.0"`, `sandbox_context true`.

- [ ] Tests: `test_sandbox01_arm_equals_backbone_at_init` (depths −1, 1, 2) · `test_sandbox01_l2_off_equals_backbone` · `test_sandbox01_changes_the_output_once_trained` · `test_sandbox01_mid_change_is_sized_to_the_stream` · `test_sandbox01_kv_cache_step_matches_full_forward` · `test_sandbox01_is_causal_through_the_model` · `test_sandbox01_checkpoint_round_trip` · `test_sandbox01_invalid_configs_are_rejected` · `test_old_sandbox_is_unchanged_by_default` · `test_v010_config_serves_both_arms` · `test_force_candidate_restores_soft_selection`.
- [ ] Run → fail; implement; run with `test_backbone.py test_midlayer.py test_multisite.py` → pass.

### Task 3: diagnostics, per-role gradients, `--addon-layer`

**Files:** `sandbox.py`, `model.py`, `train_gen.py`; tests.

**Produces:** aux keys (detached, real tokens only): `sel_p` [5], `sel_top` [5], `sel_entropy`, `cand_cos` [5,5], `delta_rms` [4] (no `spread_ratio`, so no spread floor). `EkdantLM.role_grad_norms() -> Dict[str, float]`. `train_gen.apply_addon_layer(cfg, layer) -> V2Config` and `--addon-layer` (fresh runs; error on resume with another depth). Log lines add `sel=`, `H=`, `gr=`; epoch history adds `"sandbox_v01": {...}`.

- [ ] Tests: `test_sandbox01_diagnostics_shapes_and_ranges` · `test_sandbox01_diagnostics_ignore_padding` · `test_role_grad_norms_name_every_role` · `test_addon_layer_flag_overrides_the_config` · `test_addon_layer_flag_is_rejected_on_resume_with_a_different_depth`.

### Task 4: the new data — reasoning text (stage 1) and MedLFQA (benchmark)

**Files:** `gen_data.py`, `train_gen.py`; tests in `tests/test_gen_data.py`.

**Produces:**
- `GenExample` gains `must_have: tuple = ()`, `nice_to_have: tuple = ()`; `_rows_to_examples` keeps every field (`dataclasses.replace`).
- `REASONING = ("FreedomIntelligence/medical-o1-reasoning-SFT", "en")`; `load_reasoning(limit, seed) -> List[GenExample]`: question → answer `"{Complex_CoT}\nFinal answer: {Response}"`.
- `MEDLFQA = "dmis-lab/MedLFQA"`; `load_medlfqa(limit, seed)` (split `test`): answer = `Free_form_answer`, `must_have`, `nice_to_have`.
- `exclude_questions(examples, keys) -> Tuple[List[GenExample], int]`.
- `GenDataset(..., cut_long: bool = True)`: with `False`, rows whose answer is longer than `max_answer` are dropped and counted (`dropped_long`).
- `train_gen`: `--stage1 {pubmedqa,reasoning}` (default `pubmedqa`; `--evidence`/`--pair-loss` need `pubmedqa`), `--medlfqa` (loads the benchmark and removes its questions from stage 1 and MedQuAD before splitting). The internal stage keys (`pubmedqa`, `pubmedqa_held`) and `results.json` keys stay; `results["protocol"]` records `stage1`, the dataset names and the removed/dropped counts.

- [ ] Tests (fake rows, no download — monkeypatch `load_dataset`): `test_reasoning_rows_become_reasoning_then_final_answer` · `test_reasoning_rows_too_long_are_dropped_not_cut` · `test_medlfqa_rows_keep_their_statements` · `test_medlfqa_questions_never_reach_training` · `test_rows_to_examples_keeps_every_field` · `test_stage1_flag_defaults_to_pubmedqa` · `test_evidence_needs_pubmedqa_stage1`.

### Task 5: the MedLFQA benchmark — NLI judge, factuality, L2-off, comparison, candidate forcing

**Files:** `gen_eval.py`, `train_gen.py`; tests in `tests/test_gen_eval.py`.

**Produces:**
- `NLI_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"`; `class NLIJudge` (`__call__(premises, hypotheses) -> List[str]` in {"entailment", "neutral", "contradiction"}; labels read from the model's `id2label`, error if one is missing; premise truncated first, max 512 tokens; batched, `no_grad`).
- `factuality_scores(preds, examples, judge) -> Dict`: per row, comprehensiveness = share of `must_have` entailed by the answer; hallucination = share of `must_have + nice_to_have` contradicted (Manes et al. 2024 / OLAPH); means ×100 plus `row_comp`, `row_hall`.
- `train_gen` with `--medlfqa`: `results["test"]["medlfqa"]` = reference-answer ppl (+ rows, groups), ROUGE-L / F1 and factuality on `--gen-subset-medlfqa` (default 1000, seeded) rows at `--max-new-medlfqa` (default 256), and the L2-off NLL with its CI; the generated texts are written to `medlfqa_generations.jsonl`.
- `compare`: when both runs have `medlfqa`, a paired bootstrap of the per-token NLL and of comprehensiveness / hallucination (grouped by question).
- `python -m ekdant.gen_eval candidates RUN [--limit N]`: for a v0.1.0 run, the MedLFQA NLL with each candidate forced (h, R1…R4) → `candidates.json` (never touches `results.json`); error for older runs.

- [ ] Tests (a fake judge for factuality; the real `NLIJudge` only with a tiny random BERT built from `BertConfig`): `test_factuality_counts_entailed_must_haves` · `test_hallucination_counts_contradicted_statements` · `test_nli_judge_maps_labels_from_the_model_config` · `test_nli_judge_rejects_a_model_without_entailment` · `test_compare_reports_medlfqa_when_present` · `test_candidates_needs_a_v010_run`.

### Task 6: size, smoke runs, full suite, vault

- [ ] Measure `addon_params()` for `ekdant_v0.1.0.json` on the real backbone; report to the author if it is off v0.0.5's 19,594,944 by more than ±25 %.
- [ ] CPU smoke (RULES §5): `--config configs/ekdant_v0.1.0.json --variant sandbox --stage1 reasoning --medlfqa --l2-warmup 4 --limit 64 ...` at `--addon-layer 20` and at the end, and `--variant baseline` with the same flags. Expected: finishes, finite losses, `sel=`/`H=`/`gr=` logged, `results.json` has `test.medlfqa` with comprehensiveness and hallucination.
- [ ] `python -m pytest -q` → all pass; record the count.
- [ ] Vault sync (RULES §6): `API_Reference.md`, `02_System_Architecture.md`, `26c` (pointer to v0.1.0), `12_Configuration_Guide.md`, `05_Data_Pipeline.md` §v2, `07_Training_Pipeline.md`, `09_Validation_System.md`, `run.md` (A3t: Phase 3 sanity and Phase 4 placement commands, after layers 5/10/15/20/25 and the end), `10_Known_Issues.md` (T64), `STATUS.md` (version row, version names), `README.md` (links to 30 / 30a); link check.

## Later phases (not in this plan)

Hard / Gumbel selection; auxiliary losses (R3 correctness, R2 flipped-pair consistency, entropy, balance, diversity); more rounds; several insertion points; the GPU runs.

## Connections
- Specification: [30_L2_v6_Roles_Spec.md](30_L2_v6_Roles_Spec.md)
- Current sandbox (v0.0.x): [26c_Internal_Sandbox_Build.md](26c_Internal_Sandbox_Build.md)
- Runbook: [../run.md](../run.md)
