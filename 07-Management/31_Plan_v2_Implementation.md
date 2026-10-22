# v2 Pretrained Backbone + Generative QA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the L2 sandbox (and a same-size MLP control) on a pretrained SmolLM2-135M, train it on PubMedQA then half of MedQuAD, and benchmark the other half with perplexity, ROUGE-L, token F1 and a paired bootstrap.

**Architecture:** A HF backbone produces the last hidden state; an arm-specific add-on joins as a zero-init gated residual `h + g ⊙ (f(h) − h)`; the backbone's tied `lm_head` produces logits. A new generative data/eval/training path sits beside the untouched v1 MCQ path and is selected with `--task gen`.

**Tech Stack:** Python ≥ 3.10, PyTorch ≥ 2.2, `transformers` (new), `datasets` (existing), pytest.

**Spec:** [30_Design_v2_Pretrained_Backbone.md](30_Design_v2_Pretrained_Backbone.md). All paths below are relative to `09-Code/` unless they start with a vault folder.

## Global Constraints

- Backbone id `HuggingFaceTB/SmolLM2-135M`; H = 576; vocab 49,152; tied head.
- Sandbox add-on: K=4, `debate_rounds=4`, `sandbox_mlp_ratio=4.0`, S=8, η=0.0625, readout `mean`, ρ_min=0.05, λ_sbx=1.0 → **19.59M** params (measured).
- MLP control add-on params within **±1 %** of the sandbox add-on's.
- Gate `g ∈ R^576`, initialised to **0**; at init all arms give identical logits.
- Seed 17 everywhere (split, holdout, generation subsets, shuffles).
- PubMedQA held-out = **2,000** rows; MedQuAD split by unique `question.strip().lower()`: 50 % fine-tune half (of which **10 %** of its questions = validation slice), 50 % benchmark.
- Prompts exactly: `Question: {q}\nContext: {abstract}\nAnswer:` (PubMedQA) and `Question: {q}\nAnswer:` (MedQuAD); target ` {answer}` + EOS. Loss on answer tokens only.
- Answer cap **512** tokens; sequence cap **1,024** (context trimmed from its end).
- LRs: backbone **5e-5**, add-on (incl. gate) **1e-3**; cosine per phase, warmup 200. Phase 1 = 1 epoch, phase 2 = 3 epochs. `--batch-size 8 --grad-accum 4`.
- Generation subsets: **1,000** MedQuAD benchmark rows (max 256 new tokens), **500** PubMedQA held-out rows (max 128). Bootstrap 1,000 resamples, 95 % CI.
- GPU pre-flight: `--min-free-gb 20` default; `--gpu auto` = most free memory.
- RULES: no commit/push unless the user asks; never commit `runs/`, `*.log`, `*.pt`, datasets; every fix/feature has a test; the vault is updated in the same change (Task 7). The v1 MCQ path and its 94 tests must stay green.
- New files allowed by the approved spec: `src/ekdant/backbone.py`, `gen_data.py`, `gen_eval.py`. **One more is needed: `src/ekdant/train_gen.py`** (keeps the 699-line `train.py` from doubling) — confirm with the user in the plan review.

## Review Focus

1. **Empty or whitespace MedQuAD answers / questions** — such rows must be dropped at load, never produce a zero-token answer (division by zero in perplexity). Test in Task 3.
2. **An answer longer than 1,024 − prompt even after context trimming** (MedQuAD has no context; a 5,646-token answer) — must be cut to 512 and still fit; test in Task 3.
3. **Batch with left padding during generation** — positions must count real tokens only, otherwise batched output differs from one-at-a-time output; test in Task 4.
4. **Resume after a mid-epoch stop in phase 1** (one epoch ≈ 85M tokens) — must skip already-seen batches and restore the optimiser, not restart the epoch; test in Task 5.
5. **GPU with too little free memory / no CUDA** — pre-flight must refuse with a clear message on CUDA and fall back to CPU (for tests/smoke) without CUDA; test in Task 5.

---

### Task 1: Configurable debate width + add-on blocks

**Files:**
- Modify: `src/ekdant/sandbox.py` (`Sandbox.__init__`: new `mlp_ratio: float = 2.0`, passed to every `DebateRound(dim, mlp_ratio=mlp_ratio)`)
- Modify: `src/ekdant/model.py` (new classes below; `EkdantModel` untouched)
- Test: `tests/test_addon.py` (new)

**Interfaces:**
- Produces:
  - `mlp_hidden_for(target_params: int, dim: int) -> int` — SwiGLU hidden width whose `3·dim·hidden + dim` is closest to `target_params`.
  - `class GatedAddOn(nn.Module)`: `__init__(self, dim: int, kind: str, sandbox_kwargs: dict, mlp_hidden: int)`, `kind ∈ {"sandbox","mlp"}`; attribute `gate: nn.Parameter` shape `[dim]` (zeros); `forward(h: Tensor[B,T,H], mask: Optional[Tensor[B,T]]) -> Tuple[Tensor[B,T,H], Dict[str, Tensor]]` returning `h + gate * (f(h) - h)` and aux (sandbox aux for `sandbox`, `{}` for `mlp`). `f` for `mlp` = `SwiGLU(dim, mlp_hidden)(RMSNorm(dim)(h))`; for `sandbox` = `Sandbox(...).consensus`.
  - `GatedAddOn.gate_magnitude() -> float` = `gate.abs().mean()`.

- [ ] **Step 1: Write failing tests** in `tests/test_addon.py`:
  - `test_sandbox_mlp_ratio_sets_debate_width`: parameter count of `Sandbox(576, rounds=4, mlp_ratio=4.0)` satisfies `abs(n - 19.59e6) < 0.01e6`, and `Sandbox(576)` count `abs(n - 8.30e6) < 0.01e6`.
  - `test_mlp_control_matches_sandbox_params`: `h = mlp_hidden_for(n_sbx, 576)`; `abs(n_mlp - n_sbx) / n_sbx < 0.01` where both counts exclude the gate.
  - `test_gated_addon_is_identity_at_init`: for both kinds, `out, _ = addon(h, None)`; `torch.equal(out, h)`.
  - `test_gate_gets_gradient_at_init`: loss = `out.pow(2).sum()`; after backward `addon.gate.grad.abs().sum() > 0`.
- [ ] **Step 2:** `pytest tests/test_addon.py -v` → FAIL (`mlp_hidden_for` / `GatedAddOn` undefined, `mlp_ratio` unexpected kwarg).
- [ ] **Step 3:** Implement the `Sandbox` kwarg, `mlp_hidden_for`, `GatedAddOn`; export both from `ekdant/__init__.py`.
- [ ] **Step 4:** `pytest -q` → all pass (94 old + 4 new).

### Task 2: Pretrained backbone model

**Files:**
- Create: `src/ekdant/backbone.py`
- Modify: `src/ekdant/model.py` (new `V2Config`, `EkdantLM`)
- Modify: `pyproject.toml` (add `"transformers>=4.45"`)
- Create: `configs/v2_smollm2_135m.json`
- Test: `tests/test_backbone.py` (new)

**Interfaces:**
- Consumes: `GatedAddOn`, `mlp_hidden_for` (Task 1).
- Produces:
  - `backbone.load_backbone(model_id: str, dtype=torch.float32) -> transformers.PreTrainedModel` (AutoModelForCausalLM; offline-first like `data.load_examples`: try local, on failure `_hub_offline(False)` once).
  - `backbone.tiny_llama(vocab: int = 64, dim: int = 32, layers: int = 2) -> PreTrainedModel` — random `LlamaConfig` model with tied embeddings, for tests only.
  - `@dataclass V2Config`: `backbone: str = "HuggingFaceTB/SmolLM2-135M"`, `addon: str = "none"` (`none|sandbox|mlp`), `n_sandbox=4`, `debate_rounds=4`, `sandbox_mlp_ratio=4.0`, `descent_steps=8`, `eta=0.0625`, `rho_min=0.05`, `lambda_sbx=1.0`, `readout="mean"`, `max_seq_len=1024`, `max_answer=512`; `to_dict` / `from_dict`.
  - `class EkdantLM(nn.Module)`: `__init__(self, cfg: V2Config, lm: PreTrainedModel)`; attributes `lm`, `addon: Optional[GatedAddOn]`; `forward(input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, position_ids=None) -> ModelOutput` with extra field use: `ModelOutput(logits, loss, aux)` plus `aux["past_key_values"]` when `use_cache`. Loss = token CE over `labels != -100` (shift by one) + `lambda_sbx * relu(rho_min − spread_ratio)²` in the sandbox arm. The `mlp` add-on is sized with `mlp_hidden_for(<sandbox add-on params at the same cfg>, H)`.
  - `EkdantLM.addon_params() -> int`, `EkdantLM.param_groups(backbone_lr: float, addon_lr: float, wd: float = 0.1) -> list[dict]` (wd only on ≥2-D backbone weights and ≥2-D add-on weights; gate and 1-D params wd 0).

- [ ] **Step 1: Write failing tests** in `tests/test_backbone.py` (all on `tiny_llama()`, CPU):
  - `test_all_arms_equal_backbone_at_init`: same seed-copied `lm`, three `EkdantLM`s (`none|sandbox|mlp`); `torch.equal` on logits for a padded batch.
  - `test_loss_counts_answer_tokens_only`: labels `-100` except 3 positions; loss equals manual CE over those 3 shifted positions (baseline arm, `atol=1e-6`).
  - `test_kv_cache_step_matches_full_forward_with_sandbox`: sandbox arm with gate set to 0.5; logits of the last position from (prefix forward with cache → one-token forward) equal full-forward last-position logits (`atol=1e-5`).
  - `test_param_groups_split_lrs`: every parameter appears in exactly one group; add-on params (incl. `gate`) have lr 1e-3, backbone 5e-5.
  - `test_v2_config_round_trip`: `V2Config.from_dict(cfg.to_dict()) == cfg`; shipped `configs/v2_smollm2_135m.json` loads.
- [ ] **Step 2:** `pytest tests/test_backbone.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `backbone.py`, `V2Config`, `EkdantLM`, config JSON, dependency line. The hidden state is `lm.model(...).last_hidden_state` (already final-normed); logits = `lm.lm_head(addon(h))`.
- [ ] **Step 4:** `pip install -e .[dev]` then `pytest -q` → all pass.

### Task 3: Generative datasets and the question-grouped split

**Files:**
- Create: `src/ekdant/gen_data.py`
- Test: `tests/test_gen_data.py` (new)

**Interfaces:**
- Produces:
  - `@dataclass GenExample`: `question: str`, `answer: str`, `context: str = ""`, `qtype: str = ""`.
  - `load_pubmedqa(limit: Optional[int] = None, seed: int = 17) -> List[GenExample]` (`qiaojin/PubMedQA`, `pqa_artificial`, train; context = `" ".join(row["context"]["contexts"])`, answer = `long_answer`).
  - `load_medquad(limit=None, seed=17) -> List[GenExample]` (`keivalya/MedQuad-MedicalQnADataset`; `qtype` kept). Both drop rows whose question or answer is empty after `.strip()`; both use the offline-first pattern from `data.load_examples`.
  - `format_gen_prompt(ex: GenExample) -> str` (exact strings in Global Constraints).
  - `class GenDataset(Dataset)`: `__init__(examples, tokenizer, max_len=1024, max_answer=512)`; item = `{"input_ids": LongTensor, "labels": LongTensor, "idx": int}`; answer ids = `tokenizer(" " + answer)` cut to `max_answer`, then EOS appended only if not cut; context trimmed by tokens from its end until the row fits `max_len`; rows that still do not fit are dropped and counted in `self.dropped`.
  - `gen_collate(batch, pad_id: int) -> Dict[str, Tensor]`: right-pads `input_ids`, `attention_mask`, `labels` (pad → −100), carries `idx`.
  - `question_key(q: str) -> str` = `q.strip().lower()`.
  - `split_by_question(examples, seed=17, bench_frac=0.5, val_frac=0.1) -> Tuple[list, list, list]` = `(ft, val, bench)`; one `torch.randperm` over sorted unique keys.
  - `holdout(examples, n=2000, seed=17) -> Tuple[list, list]` = `(train, held)`.

- [ ] **Step 1: Write failing tests** (synthetic examples + a stub tokenizer that maps whitespace words to ids; no downloads):
  - `test_prompt_strings_exact`: both templates, character-exact.
  - `test_labels_mask_prompt`: every label before the answer start is −100; answer labels equal answer ids; last label is EOS.
  - `test_long_answer_capped_and_fits`: 5,000-word answer, no context → `len(labels != -100) == 512`, no EOS, total ≤ 1,024.
  - `test_context_trimmed_not_answer`: 2,000-word context + 40-word answer → total ≤ 1,024, all 40 answer tokens + EOS present.
  - `test_empty_rows_dropped` (via a `_rows_to_examples` helper used by both loaders): rows with `""`/`"   "` question or answer are absent.
  - `test_split_by_question_no_leak_and_stable`: duplicate questions differing only in case/whitespace go to the same set; sets disjoint by key; ≈50 % keys in bench; val ≈10 % of ft keys; identical on a second call.
  - `test_holdout_sizes`: 2,000 held, rest train, disjoint, stable.
- [ ] **Step 2:** `pytest tests/test_gen_data.py -v` → FAIL.
- [ ] **Step 3:** Implement `gen_data.py`.
- [ ] **Step 4:** `pytest -q` → all pass.

### Task 4: Generative evaluation

**Files:**
- Create: `src/ekdant/gen_eval.py`
- Test: `tests/test_gen_eval.py` (new)

**Interfaces:**
- Consumes: `EkdantLM` (Task 2), `GenDataset`, `gen_collate`, `format_gen_prompt` (Task 3).
- Produces:
  - `answer_nll(model, loader, device, autocast) -> Dict` → `{"ppl": float, "nll": float, "tokens": int, "n": int, "row_nll": List[float], "row_tokens": List[int], "row_idx": List[int], "addon": {...}}` (`addon` = mean sandbox diagnostics over answer tokens, sandbox arm only).
  - `by_type(result: Dict, examples: List[GenExample]) -> Dict[str, float]` (perplexity per `qtype`).
  - `normalise_text(s: str) -> List[str]`: lowercase, strip ASCII punctuation, split on whitespace.
  - `rouge_l(pred: str, ref: str) -> float` (LCS F-measure, β = 1) and `token_f1(pred: str, ref: str) -> float` (multiset overlap), both on `normalise_text`; empty-vs-empty = 1.0, empty-vs-nonempty = 0.0.
  - `generate_greedy(model, tokenizer, prompts: List[str], max_new: int, device, batch_size: int = 16) -> List[str]`: left-pads each batch; `position_ids = attention_mask.cumsum(-1) - 1` (clamped at 0) on the prefill and incremented per step; KV cache via `use_cache=True`; stops a row at EOS.
  - `paired_bootstrap(a: List[float], b: List[float], groups: List[str], n: int = 1000, seed: int = 17) -> Dict` → `{"diff": mean(b − a), "lo": float, "hi": float}`, resampling **groups** (question keys) with replacement.
  - CLI: `python -m ekdant.gen_eval compare <run_dir> <run_dir> [...]` prints a table (benchmark ppl, ROUGE-L, F1, `|g|`) and the pairwise bootstrap of per-row benchmark NLL against the first run.

- [ ] **Step 1: Write failing tests:**
  - `test_rouge_l_and_f1_known_values`: `rouge_l("the cat sat on the mat", "the cat is on the mat") == pytest.approx(0.8333, abs=1e-4)`; `token_f1` same pair `== pytest.approx(0.8333, abs=1e-4)`; punctuation/case ignored; empty cases as specified.
  - `test_answer_nll_matches_model_loss`: on `tiny_llama` baseline arm, `result["nll"]` equals the model's own loss on the same single batch (`atol=1e-5`), `tokens` equals the count of non −100 labels.
  - `test_generate_batched_equals_single`: three prompts of different lengths; `generate_greedy(..., batch_size=3)` == per-prompt `batch_size=1` outputs (sandbox arm, gate 0.5).
  - `test_bootstrap_ci_contains_true_shift`: `a` random, `b = a + 0.3`, 200 groups → `lo < 0.3 < hi` and `lo > 0`.
  - `test_by_type_groups_rows`.
- [ ] **Step 2:** `pytest tests/test_gen_eval.py -v` → FAIL.
- [ ] **Step 3:** Implement `gen_eval.py`.
- [ ] **Step 4:** `pytest -q` → all pass.

### Task 5: Generative training entry point

**Files:**
- Create: `src/ekdant/train_gen.py` (pending the user's OK on the 4th new file)
- Modify: `src/ekdant/train.py` (`--task {mcq,gen}`, default `mcq`; `gen` hands the remaining argv to `train_gen.main`; `--variant` choices unchanged for `mcq`)
- Test: `tests/test_train_gen.py` (new)

**Interfaces:**
- Consumes: Tasks 2–4; `lr_at`, `fmt_duration`, `log` from `train.py`.
- Produces:
  - `pick_device(gpu: str, min_free_gb: float) -> torch.device` — `gpu` = `"auto"` or an index; no CUDA → `cpu`; CUDA with `mem_get_info` free < `min_free_gb` → `SystemExit` with a message naming free/total GB and suggesting `CUDA_VISIBLE_DEVICES`.
  - `main(argv: Optional[List[str]] = None) -> None`. Flags: `--variant {baseline,sandbox,mlp_control}` (→ `addon none|sandbox|mlp`), `--config` (default `configs/v2_smollm2_135m.json`), `--out`, `--batch-size 8`, `--grad-accum 4`, `--lr 5e-5`, `--addon-lr 1e-3`, `--warmup 200`, `--epochs 1`, `--ft-epochs 3`, `--eval-fraction 0.25`, `--gen-subset 1000`, `--gen-subset-pubmed 500`, `--limit`, `--seed 17`, `--log-every 50`, `--gpu auto`, `--min-free-gb 20`, `--resume`, `--eval-only`, `--model-id` (override, tests use `"tiny"` → `tiny_llama()` + stub tokenizer).
  - Protocol, all into `<out>/results.json`:
    1. pre-flight + log `params total / addon`, free GPU memory; fresh-run guard (refuse if `last.pt` exists without `--resume`).
    2. `zero_shot`: `answer_nll` on PubMedQA held-out, MedQuAD val, MedQuAD bench.
    3. phase 1 (PubMedQA train): validation (`answer_nll` on held-out) every `eval_fraction` of the epoch → `history`; mid-epoch `last.pt` at each validation.
    4. `phase1`: held-out + MedQuAD val.
    5. phase 2 (MedQuAD ft): each epoch `answer_nll` on val → `best.pt` on lowest `nll`.
    6. `selected`: better of `best.pt` / `last.pt` on val (ties keep `best.pt`).
    7. `test`: `medquad_bench` (`answer_nll` + `by_type` + ROUGE-L/F1 means on the 1,000-row seeded subset, with `row_*` lists kept for bootstrap), `pubmedqa_held` (`answer_nll` + ROUGE-L/F1 on 500 rows), `gate_magnitude`, cost (`params`, `tokens_seen`, `wall_clock_s`, `peak_mem_gb`, `tokens_per_s`).
  - `last.pt` = `{model, optimizer, cfg, stage, epoch, batches_done, opt_step, history, best_val_nll}`; batch order per epoch from `torch.Generator().manual_seed(seed * 1000 + stage * 100 + epoch)`, so resume skips exactly `batches_done` batches.

- [ ] **Step 1: Write failing tests** (CPU, `--model-id tiny`, monkeypatched loaders returning ~60 synthetic rows, `--limit` small):
  - `test_pick_device_refuses_low_memory` (monkeypatch `torch.cuda.is_available → True`, `device_count → 2`, `mem_get_info → (5e9, 80e9)`): `SystemExit`; with `(50e9, 80e9)` on index 1 and `"auto"` → `cuda:1`; no CUDA → `cpu`.
  - `test_gen_run_end_to_end_writes_results`: `main([...])` for `sandbox`; `results.json` has keys `zero_shot`, `phase1`, `selected`, `test.medquad_bench.ppl`, `test.medquad_bench.rouge_l`, `test.gate_magnitude`; all perplexities finite.
  - `test_resume_mid_epoch_matches_uninterrupted`: run A full; run B stopped after the first mid-epoch checkpoint (monkeypatch a stop hook) then `--resume`; final `last.pt` model tensors `torch.allclose(atol=1e-6)` to run A's.
  - `test_fresh_run_guard`: second fresh `main` into the same `--out` → `SystemExit`.
  - `test_mcq_path_untouched`: `python -m ekdant.train --task mcq` arg parsing still accepts the v1 flags (parse only).
- [ ] **Step 2:** `pytest tests/test_train_gen.py -v` → FAIL.
- [ ] **Step 3:** Implement `train_gen.py` and the `train.py` dispatch.
- [ ] **Step 4:** `pytest -q` → all pass.
- [ ] **Step 5: Local real-model smoke (CPU, not a result):** `python -m ekdant.train --task gen --variant sandbox --out <scratchpad>/smoke --limit 64 --epochs 1 --ft-epochs 1 --gen-subset 8 --gen-subset-pubmed 8 --batch-size 2 --grad-accum 1` → finishes, writes `results.json`; record params total / add-on from its log for Task 7.

### Task 6: Run commands on the shared 4×A100 pod

**Files:**
- Modify: `run.md` (new top section "v2 runs"; v1 sections kept under a "v1 (MCQ, from scratch)" heading)

- [ ] **Step 1:** Write the v2 quick start: `nvidia-smi` check; three parallel commands, one per free GPU:
  ```bash
  FLAGS="--task gen --batch-size 8 --grad-accum 4 --epochs 1 --ft-epochs 3"
  CUDA_VISIBLE_DEVICES=1 nohup python -m ekdant.train $FLAGS --variant baseline    --out runs/v2_baseline > v2_baseline.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 nohup python -m ekdant.train $FLAGS --variant sandbox     --out runs/v2_sandbox  > v2_sandbox.log  2>&1 &
  CUDA_VISIBLE_DEVICES=3 nohup python -m ekdant.train $FLAGS --variant mlp_control --out runs/v2_mlp      > v2_mlp.log      2>&1 &
  python -m ekdant.gen_eval compare runs/v2_baseline runs/v2_sandbox runs/v2_mlp
  ```
  plus: resume, what each log line means, the `results.json` keys, how to read the bootstrap (a difference counts only if the CI excludes 0), OOM/pre-flight troubleshooting, and "times not yet measured".
- [ ] **Step 2:** Verify every flag in `run.md` exists: `python -m ekdant.train --task gen --help` lists each one.

### Task 7: Vault sync (same change as the code)

**Files (spec §11):** `README.md`, `01-Introduction/01_Project_Overview.md`, `01-Introduction/STATUS.md`, `RULES.md`, `project_explanation.md`, `02-Architecture/02_System_Architecture.md`, `03_Codebase_Map.md`, `12_Configuration_Guide.md`, `API_Reference.md`, `04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md`, `26c_Internal_Sandbox_Build.md`, `05-Data/05_Data_Pipeline.md`, `06-Training-Eval/07_Training_Pipeline.md`, `09_Validation_System.md`, `13_Debugging_Guide.md`, `14_Performance_Analysis.md`, `07-Management/10_Known_Issues.md`, `11_Future_Work.md`, `30_Design_v2_Pretrained_Backbone.md`, `08-Thesis/thesis.tex`, `thesis.md`, `GLOSSARY.md`, `09-Code/README.md`.

- [ ] **Step 1:** Known Issues: add `T28` — v1 could not test L2 (from-scratch 43.53M sat at the majority-letter rate, T27) → v2 decision, with the new tests named.
- [ ] **Step 2:** RULES: project facts (backbone, three arms, 19.59M add-on, datasets/split, perplexity primary, majority-rate rule kept for `--task mcq`, 4 new modules, `transformers` dependency, one arm per free GPU); sync map rows for the new files.
- [ ] **Step 3:** Architecture/config/API/codebase-map/22/26c: the v2 forward, gated residual, R=4 / 4× MLP, MLP control, every new signature from Tasks 1–5.
- [ ] **Step 4:** Data (05) and training/eval (07, 09, 13, 14): datasets, measured lengths, split, prompts, caps, protocol, metrics, bootstrap, pre-flight; test table counts.
- [ ] **Step 5:** Thesis (`thesis.tex`, `thesis.md`, GLOSSARY): method chapters rewritten for v2; v1 kept as the "from-scratch pilot" that motivated v2 (its numbers unchanged); results table rows `[PENDING]` for the three arms. No invented numbers.
- [ ] **Step 6:** README, overview, STATUS, `project_explanation.md`, `09-Code/README.md`, Future Work; mark spec 30 "built"; update test counts everywhere (`grep -rn "94 " --include=*.md --include=*.tex`).
- [ ] **Step 7: Verify:** `pytest -q` all pass; the scratchpad `texcheck.py` prints `begin/end` equal, brace balance 0, no undefined refs, no broken links; `grep -rn "chance = 25\|Chance is \$25" .` returns nothing outside v1 history.
- [ ] **Step 8:** Report to the user; commit/push only if they ask.

Connections: spec → [30_Design_v2_Pretrained_Backbone.md](30_Design_v2_Pretrained_Backbone.md) · issues → [10_Known_Issues.md](10_Known_Issues.md)
