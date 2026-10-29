# RULES.md — how to work on the Ekdant project

You are an AI coding agent working inside **Ekdant**: an MSc thesis (Hochschule Heilbronn, author Harshil Gorasiya) that is **both** an Obsidian research vault **and** a PyTorch codebase. Every rule below exists because breaking it has already cost time on this project. Read this whole file before your first action in a session. When a rule and a user instruction conflict, the user wins — but say which rule you are setting aside.

---

## 0. The ten rules you must never break

1. **Never invent a number.** Every number you write (accuracy, parameter count, row count, step count, test count) must come from the code, a config, a log, `results.json`, or an existing vault file you just read. If you cannot point to the source, write `[PENDING]` or ask.
2. **L1 is not built, and L1 and L2 are independent.** The selective-SSM storage layer (L1) is *specified* in the vault and thesis but **does not exist in code**. L1 and L2 never feed each other — the former M3 write-back was removed (T38); do not reintroduce it. Never describe them as implemented, trained, measured or tested.
3. **Never touch the benchmark protocols** without the user explicitly asking. **v2 (current, `--task gen`):** PubMedQA `pqa_artificial` (phase 1, 2,000 rows held out) → MedQuAD fine-tune half (phase 2; 10 % of its questions pick the checkpoint) → MedQuAD benchmark half scored once. MedQuAD is split by unique question — never let one question reach two sets. **v1 (`--task mcq`):** MedMCQA (phase 1) → seed-locked 60 % of MedQA (phase 2) → untouched 40 % benchmark scored once. In both: never train on the benchmark, never select a checkpoint on it; never score MedMCQA's public **test** split (no labels, `cop = -1`).
4. **Keep the comparison fair.** v2: the two arms (`baseline`, `sandbox`) share one config file and differ **only** in `--variant` (L2 or not); every flag is identical. v1: `sandbox_50m.json` and `baseline_50m.json` differ **only** in `use_sandbox`. Anything that changes one arm's training must change every arm's identically.
5. **No external kernels or frameworks.** Never add `mamba-ssm`, `causal-conv1d`, `flash-attn`, `triton`, AutoGen, CrewAI, LangGraph. The only pretrained weights are the v2 backbone (SmolLM2-135M through `transformers`, approved by the user 2026-10-22). Do not add any other dependency or model without asking.
6. **Every code fix gets a test** in `09-Code/tests/`, and the full suite must pass before you say you are done.
7. **Code and vault move together.** A change is not finished until every vault file that describes it is updated (see §6, the sync map).
8. **Report honestly.** If a test fails, show the failure. If you skipped a step, say so. A tie or a negative result is reported as a tie or a negative result — never softened.
9. **Do not delete, move or rename files or folders** unless the user asked. Never run destructive git commands (`reset --hard`, `push --force`, `clean -f`, `checkout -- .`).
10. **Never commit** unless the user asks. Never commit `runs/`, `*.log`, `.remember/`, `.obsidian/`, checkpoints (`*.pt`), datasets, or any secret: a Hugging Face token lives only in the git-ignored `09-Code/configs/hf_token.txt` or the `HF_TOKEN` variable — **never in a config JSON** (configs are saved into `last.pt` and `results.json`).

---

## 1. What this project is (facts — verify before relying on them)

| Fact | Value | Source of truth |
|---|---|---|
| Degree | MSc thesis, Hochschule Heilbronn | `08-Thesis/thesis.tex` title page |
| Built model, v2 (current) | `SmolLM2-135M → last hidden state h → h + g ⊙ (RMSNorm(f(h)) − h) → SmolLM2 tied lm_head`; `g ∈ R^576` starts at 0, so every arm equals the backbone at step 0 | `09-Code/src/ekdant/model.py` (`EkdantLM`, `GatedAddOn`), `backbone.py` |
| v2 arms | `baseline` (no add-on) **134,515,008** params; `sandbox` (+L2 **19,594,944**, total 154,109,952, 12.7 %) | measured with `count_parameters` on the real backbone |
| L2 importance (T44) | every L2 run ends with the L2-off test (benchmark NLL with all L2 switched off, paired CI) — report it with every L2 result: it, not the gate value, shows whether the model depends on L2. `--l2-warmup N` freezes SmolLM2 for N phase-1 steps (`run.md` A3f/A3g) | `model.py` `l2_off`, `train_gen.py` |
| rollback (T43) | `--gate-rollback`: a burst restores the last in-memory snapshot (every 100 steps), then the floor re-opens 2× slower and 0.05 lower (≥ 0.8 × floor). Used by the forced runs with a fast ramp (100 + 500 steps, `run.md` A3e) | `train_gen.py` `GateRamp.rollback`, `take_snapshot` |
| forced L2 (T42) | `--gate-floor 0.5 --gate-force`: L2 must be used — gate never below 0.5, output scale frozen, mid output sized to the stream. Runs `v26_end_forced` / `v26_mix_forced` vs the v25 runs (`run.md` A3e). Optional runs keep the free gate | `model.py` `GatedAddOn(force=…)`, `train_gen.py` `--gate-force` |
| more epochs (T41, current for new runs) | 3 PubMedQA + 5 MedQuAD epochs for every arm (`run.md` A3d); the whole model is trained — no LoRA (tried and removed at the author's request). Never compare these runs with the 1 + 3-epoch runs | `run.md` A3d |
| the mix (T40) | one shared L2 at 9 sites — after layers 10, 15, 20, 22, 24, 26, 28, 30 and at the end (`configs/v2_sandbox_mix.json`, 19,608,768 params, L2 checkpointing on) — compared with `v22_baseline` / `v22_end` (`run.md` A3c) | `model.py` `MultiAddOn.final_site` |
| gate floor (T39, optional) | `--gate-floor 0.7`: after 200 steps the gate is ramped open to 0.7 over 2,000 steps and held; brake events move the ramp back. Off by default — only experiment run 4 (`run.md` A3b) uses it | `train_gen.py` `GateRamp` |
| v2.3 gate control (current, T37) | gate LR boost 10× fading over 1,000 steps; effective gate `tanh`-capped at 1.0 (`gate_cap`); brake on a 3× L2 gradient spike or `l2` > 50 (gates ×0.9, L2/gate LRs ×0.5, recover ×1.01). Used by every sandbox run of the four-run experiment | `train_gen.py` `GateBrake`, spec 30 §15 |
| v2.2 experiment | four runs at 1 PubMedQA epoch with identical flags: baseline, L2 after layer 20 (`v2_sandbox.json`), **one shared L2 after layers 10/20/30** (`v2_sandbox_multi.json`, 19,598,976 params, own gate + norms per site), L2 at the end (`v2_sandbox_final.json`) — `run.md` A3, T36. v2.1 (2 epochs) tied and the 2nd epoch hurt both arms | spec 30 §14 |
| v2.1 placement | L2 after decoder layer **20** of 30: `h + g ⊙ RMSNorm(L2(RMSNorm(h)))`; add-on **19,595,520**, total 154,110,528; gate LR **1e-3** (`--gate-lr`), L2 matrices 1e-4; 2 PubMedQA epochs. v2.0 (after the final norm) is `configs/v2_sandbox_final.json` | `model.py`, spec 30 §13, T35 |
| v2 L2 | K=4, **R=4 debate rounds, debate MLP 4×**, S=8, η=0.0625, ρ_min=0.05, λ_sbx=1.0, readout `mean` (scaled to v1's 13.1 % share of the model) | `configs/v2_baseline.json`, `configs/v2_sandbox.json` (identical except `addon`) |
| v2 data | PubMedQA `pqa_artificial` 211,269 (2,000 held out); MedQuAD 16,407 rows / 14,345 unique questions, split 50/50 by question, 10 % of the fine-tune half's questions = validation | `gen_data.py`, `train_gen.py` |
| v2 metrics | reference-answer perplexity (primary) + per-question-type; ROUGE-L / token F1 on 1,000 generated answers; paired bootstrap 95 % CI between arms | `gen_eval.py` |
| Built model, v1 (`--task mcq`, from-scratch pilot) | `embed → T (windowed dense Transformer) → L2 (sandbox) → tied head` | `09-Code/src/ekdant/model.py` (`EkdantModel`) |
| Not built | L1 selective SSM (independent of L2; no write-back, T38) | `04-Model-DeepDive/21_*`, `26a_*` (design only) |
| A/B sizes | sandbox **50.09M** / baseline **43.53M** (L2 = 6.56M, T = 26.75M, embeddings 16.78M) | `estimate_model_size`, `tests/test_model.py` |
| Backbone | H=512, N_T=12, 8 Q heads / 2 KV heads, window 512, SwiGLU 1024, RoPE base 1e5 | `09-Code/configs/*.json` |
| Sandbox | K=4 trajectories, R=2 debate rounds, S=8 anchored descent steps, η=0.0625, ρ_min=0.05, λ_sbx=1.0, readout `mean` | same |
| Data | MedMCQA 182,496 labelled train / 4,183 val (4,170 usable after row filters); MedQA 10,178 train → 6,106 fine-tune / 4,072 benchmark; MedQA test 1,273 | `09-Code/src/ekdant/data.py`, `train.py` |
| Seed | 17 (also fixes the 60/40 MedQA split) | `train.py --seed` |
| Tokenizer | BPE, 32,768 configured (26,766 actually learned), lower-cased | `data.py::Tokenizer` |
| Dependencies | `torch>=2.2`, `datasets>=2.19,<5`, `pyarrow>=12,<21`, `tokenizers>=0.19`, `transformers>=4.45,<5`, `pytest>=8`; tested on PyTorch 2.2.2 + transformers 4.57.6 (pod mirror) and PyTorch 2.9.0 | `09-Code/requirements.txt` = `pyproject.toml` (`tests/test_requirements.py`) |
| Tests | run `pytest -q` and count — do not trust a number written in a doc | `09-Code/tests/` |
| Completed results | **v2.2 + gate control (seed 17, current):** L2 at the end 3.7761 vs baseline 3.7826 (per-token NLL −0.0017 [−0.0019, −0.0016]), ROUGE-L 0.3202 vs 0.3144, F1 0.3852 vs 0.3788; layer 20 −0.0003, layers 10/20/30 −0.0009 (`run.md` A7d). Earlier v2 paired runs (seed 17): benchmark ppl 3.7817 baseline vs 3.7795 sandbox, per-token NLL −0.0006 [−0.0007, −0.0005], ROUGE-L 0.3127/0.3137, F1 0.3770/0.3789, gate 0.0054 — a practical tie (`run.md` A7a). v1: 3-epoch A/B tie (MedMCQA val 33.31 % vs 32.01 %); legacy 100-epoch baseline (33.33 % / `medqa_bench` 26.89 %); 2-epoch baseline 33.91 % / 25.44 % / 29.62 % — all at the majority-letter rate (T27) | `run.md` §6, `07-Management/10_Known_Issues.md` |

If any of these disagree with the code, **the code is right** and the doc is stale — fix the doc (and tell the user).

### Source-of-truth order (highest wins)
1. Code in `09-Code/src/ekdant/` and the two config JSONs
2. Logs and `results.json` from real runs
3. `02-Architecture/02_System_Architecture.md` (canonical spec)
4. `08-Thesis/thesis.tex` (authoritative over `thesis.md`)
5. Every other vault note

---

## 2. Map of the repository

```
README.md                  single map of the vault — start here
RULES.md / AGENTS.md       this file (AGENTS.md just points here)
project_explanation.md     presentation story for the supervisor
run.md                     operations runbook: train / resume / test / benchmark
01-Introduction/           overview + STATUS.md (readiness score, version history)
02-Architecture/           02 canonical spec · 03 codebase map · 12 config guide · API_Reference
03-Background/             paper notes (B = SSM, C = sandbox, T = Transformer) — thesis Ch. 2
04-Model-DeepDive/         how L1/L2 work, build manuals 26a/b/c, worked examples, novelty audit
05-Data/                   MedMCQA → MedQA pipeline
06-Training-Eval/          07 training · 09 validation · 13 debugging · 14 performance
07-Management/             10_Known_Issues (T-numbered log) · 11_Future_Work
08-Thesis/                 thesis.tex (source) · thesis.md (outline) · thesis.pdf (built) · GLOSSARY · figures/
09-Code/                   the Python package, tests, configs
papers/                    66 PDFs + indexes — never edit or delete PDFs
```

Code modules (`09-Code/src/ekdant/`): `model.py` (v1 `EkdantModel`; v2 `V2Config`, `GatedAddOn`, `EkdantLM`), `transformer_stack.py` (v1 T), `sandbox.py` (L2), `output_head.py` (v1 tied head), `data.py` (v1 datasets, tokenizer, collate), `train.py` (v1 protocol + `--task` dispatch), `backbone.py` (v2 SmolLM2 loading), `gen_data.py` (v2 datasets, split), `gen_eval.py` (v2 metrics, generation, bootstrap, `compare`), `train_gen.py` (v2 protocol), `__init__.py` (re-exports).

---

## 3. How to think before you act

Work in this order every time. Do not skip steps because a task "looks small".

1. **Restate the task** in one sentence. If it has more than one reasonable reading and the readings lead to different changes, ask one short question instead of guessing.
2. **Read before you write.** Open every file you will change, and the code it calls. Search (`grep`) for every other place the same name, number or claim appears — in code *and* in the vault.
3. **Separate fact from design.** For each statement you are about to write, ask: is this measured/implemented (fact), or specified/planned (design)? Label design as design.
4. **Find the root cause.** For a bug, reproduce it (a failing test is best), explain *why* it happens, then fix the cause, not the symptom. Do not add `try/except` to hide an error.
5. **Pick the smallest correct change.** Do not refactor, rename, reformat or "improve" code you were not asked to touch. Do not add features, flags or files the task does not need.
6. **Check the blast radius.** Will this change the baseline arm? Old checkpoints (`last.pt` keys)? `results.json` keys other scripts read? A running campaign? If yes, tell the user before doing it.
7. **Verify.** Run the tests. For training-loop changes also run the smoke run (§5). Read the output — do not assume it passed.
8. **Sync the vault** (§6), then re-run the link check.
9. **Report** (§9).

### When to stop and ask the user
- The change would alter the protocol, the A/B fairness, a default hyper-parameter, or a thesis claim.
- You would need a new dependency, a new top-level folder, or to delete/move files.
- Numbers in two sources disagree and you cannot tell which is right.
- A test fails and the fix is not obvious after one honest attempt.
- The task is research judgment (e.g. "should L2 have a residual path?"). Give a recommendation with reasons; do not silently implement it.

---

## 4. Rules for changing code

- **Match the surrounding code**: same naming, type hints, docstring density, comment style, import order. Comments explain *why*, never restate *what*.
- **Keep the forward pass causal.** Position *t* may depend only on positions ≤ *t* of the same example. Nothing may mix information across batch rows. Every sandbox op is per position; keep it that way — v2 generation relies on it to reuse the backbone's KV cache (`test_kv_cache_step_matches_full_forward_with_sandbox`).
- **v2 add-ons start as a no-op.** The gate is initialised to 0 so all arms equal the backbone at step 0 (`test_all_arms_equal_backbone_at_init`). Never initialise it otherwise without asking.
- **v2 loss is on answer tokens only** (prompt labels −100); v2 diagnostics are over answer tokens.
- **Padding is on the right** and has `labels = -100`. Statistics (means, variances, ρ) must count real tokens only — pass `attention_mask`.
- **v1 evaluation reads `logits[answer_pos - 1]`** (the position that *predicts* the letter) and compares against the letter **index** 0–3, never a vocab id.
- **Config changes**: a new knob goes in `EkdantConfig` with a default that reproduces old behaviour, in both JSONs (identical value unless it is `use_sandbox`), in `validate()` if it has legal values, and in `12_Configuration_Guide.md`.
- **CLI flags** default to the old behaviour so old commands and `--resume` keep working.
- **Checkpoints**: never rename or remove keys in `last.pt` / `best.pt` (`model, cfg, step, completed, opt_step, history, best_medmcqa`); only add.
- **`results.json`**: only add keys; never rename `phase1`, `test.medqa_bench`, `test.medqa_test`, `test.medmcqa_val`, `selected`, `history`.
- **No new files under `src/ekdant/`** without asking (11 modules today: the v1 seven + `backbone.py`, `gen_data.py`, `gen_eval.py`, `train_gen.py`, approved 2026-10-22).
- **Tests**: one test per behaviour, named as a sentence (`test_lowest_energy_readout_is_per_position_and_causal`), CPU-only, tiny shapes, seeded, < 1 s each. Put it in the file for that module.

---

## 5. Commands (run from `09-Code/` unless stated)

```bash
pip install -e .[dev]                 # the trailing dot is required
python -m pytest -q                   # full suite, CPU, ~10 s — must be all green
python -m pytest tests/test_sandbox.py -v -k readout    # one area

# v2 smoke (CPU, ~10 min, downloads SmolLM2 once) — required after any v2 change
python -m ekdant.train --task gen --variant sandbox --out <scratch>/v2 --limit 64 --epochs 1 --ft-epochs 1 --gen-subset 4 --gen-subset-pubmed 4 --max-new 24 --max-new-pubmed 24 --batch-size 2 --grad-accum 2 --warmup 2 --gpu cpu

# v1 smoke run per arm (~2 min CPU each) — required after any v1 train.py / model.py change
python -m ekdant.train --variant sandbox  --out <scratch>/sbx --limit 64 --epochs 1 --ft-epochs 1 --batch-size 4 --no-compile
python -m ekdant.train --variant baseline --out <scratch>/bas --limit 64 --epochs 1 --ft-epochs 1 --batch-size 4 --no-compile
```

Never write smoke outputs inside the repo (use a temp/scratch folder). `--limit` runs are pipeline checks, never results.

**Link check** (run from the repo root after editing any `.md`): every relative Markdown link must resolve to a real file. A quick check:

```bash
python - <<'EOF'
import re, os, glob
bad = [(p, l) for p in glob.glob('**/*.md', recursive=True)
       for l in re.findall(r'\]\(([^)#\s]+\.(?:md|pdf|png|tex|json|py))', open(p, encoding='utf8').read())
       if not l.startswith('http') and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(p), l)))]
print(bad or 'all links ok')
EOF
```

**LaTeX**: after editing `thesis.tex`, build with `pdflatex thesis.tex` three times from `08-Thesis/` if a TeX install is available; if not, say the PDF is stale. Never commit a `thesis.pdf` you did not just build.

### Windows environment notes
- The machine is Windows. Shell may be PowerShell or Git Bash — use the syntax of the shell you are in.
- Files use **CRLF** line endings. Preserve them; do not convert whole files.
- In Git Bash, heredocs can mangle backslashes. For any script containing `\` (LaTeX, regex), write it to a temporary `.py` file and run that instead of piping a heredoc.

---

## 6. Sync map — if you change X, update Y

| You changed | Also update |
|---|---|
| Any behaviour in `model.py` / `sandbox.py` / `transformer_stack.py` | `02-Architecture/API_Reference.md`, `02_System_Architecture.md`, the matching `04-Model-DeepDive/26*` build manual, `thesis.tex` if the method text describes it |
| `train_gen.py` / `gen_data.py` / `gen_eval.py` / `backbone.py` (v2) | `run.md` Part A, `05-Data/05_Data_Pipeline.md` §v2, `06-Training-Eval/07_Training_Pipeline.md` §v2, `09_Validation_System.md`, `API_Reference.md`, `thesis.tex` §Data / §Evaluation |
| `train.py` flags or protocol | `09-Code/README.md` flag table, `run.md` (§3 flags, §6, §8), `06-Training-Eval/07_Training_Pipeline.md`, `12_Configuration_Guide.md` §2 |
| A dependency or version pin | `09-Code/pyproject.toml` **and** `09-Code/requirements.txt` (the test fails otherwise), `run.md` A1 requirements table, `09-Code/README.md` Install, RULES §1 |
| A config JSON | `12_Configuration_Guide.md` §1, `thesis.tex` parameter table if sizes change, `estimate_model_size` tests |
| `data.py` | `05-Data/05_Data_Pipeline.md`, `thesis.tex` §Data, `thesis.md` Ch. 4 |
| Number of tests | every doc that states the count — find them with `git grep -n "[0-9]\+ tests\|[0-9]\+-test\|[0-9]\+ passing"` and set them to what `pytest -q` printed; also the per-file table in `09_Validation_System.md` |
| A new result from a real run | `run.md` §6 reference numbers, `01-Introduction/STATUS.md`, `README.md` status table, `thesis.tex` §Results (and `thesis.md` Ch. 6), `project_explanation.md` §8 |
| A bug found or fixed | a new row `T<next number>` in `07-Management/10_Known_Issues.md` (Issue · Location · Cause · Fix/Status with test name · Priority); mention it in `07_Training_Pipeline.md` §4 if it is a training-loop issue |
| Scope, readiness or a milestone | `01-Introduction/STATUS.md` (and append a row to its version history — never rewrite old rows), `README.md` |
| Future plans | `07-Management/11_Future_Work.md` |

After the sync, `grep` again for the old wording/number to make sure no copy is left behind.

---

## 7. Rules for writing in the vault and the thesis

**Voice and honesty**
- Plain, precise, academic English. Short sentences. One idea per paragraph.
- State what was done, what was measured, and what is still open. Never write "proves", "significant", "state of the art" or "outperforms" unless a result with an error bar supports it. At one seed, a gap smaller than ~2σ (σ ≈ 0.7 pt on MedMCQA val) is a tie.
- Keep the project's habit of citing evidence against itself (e.g. C24, the compute-matched negative result). Never remove a limitation, defect, or negative result to make the work look better.
- Label design-only material: "specified, not implemented", "design-time estimate", "projection, not measured".

**Numbers**
- Use the exact figure from the source (`50.09M`, `33.31 %`, `182,496`), with the unit and the split it was measured on. Thousands separators in prose (`182,496`); `{,}` in LaTeX (`182{,}496`).
- Every new number needs a traceable source; mention it when it is not obvious ("from `baseline_100ep.log`").

**Markdown / Obsidian**
- Relative links only (from a subfolder, link to the runbook as `../run.md`), and keep them working. Keep the existing heading structure, the `==highlight==` style, callouts (`> [!note]`), and the "Connections" footers.
- Do not create new hub/index notes — `README.md` is the single map. A new note needs a link from `README.md` or from the folder's main note.
- Match the spelling already used in the file you are editing.

**LaTeX (`thesis.tex`)**
- Escape `%`, `_`, `&`, `#` in text; code names go in `\texttt{...}` with `\_`.
- Use the existing macros (`\rhodim`, `\lam`, `\Kdim`, …), colours and table style (`booktabs`, `\rowcolor{ekHeadBG}` headers). Every `\ref` needs a `\label`, every `\cite` a `\bibitem`.
- Keep `thesis.md` (the outline) consistent with `thesis.tex`; the `.tex` is authoritative.
- Results tables stay `[PENDING]` until a real run fills them.

---

## 8. Git rules

- Work only on what was asked; keep each commit to one logical change.
- Before committing: `pytest -q` green, link check clean, `git status` reviewed — nothing from `runs/`, logs, checkpoints, `.remember/` or `.obsidian/` staged.
- Message: a one-line summary in the imperative (≤ 72 chars), a blank line, then bullets saying *what* changed and *why*, with Known-Issues numbers (`T24`) where relevant.
- Never amend or rewrite pushed history; never `--force`; never skip hooks.
- Do not push unless the user asks.

---

## 9. How to report back

End every task with a short report, in this order:
1. **What changed** — files and the reason, one line each.
2. **Verification** — the exact test result (`138 passed`), smoke runs, link check. If you could not verify something, say exactly what and why.
3. **What you did not do** and **decisions for the user** — open questions, risks (e.g. "this changes the sandbox arm's training; the baseline is unaffected"), recommendations you chose not to implement.

No filler, no praise of the work, no claims beyond what you verified.

---

## 10. Known traps (learned the hard way)

- MedMCQA's public test split is unlabelled → scoring it measured the "A"-prediction rate (T18).
- The answer letter is 1 token in ~34–173, so plain LM loss barely trains the decision → `--answer-weight` (T22).
- Long fine-tunes on 6,106 MedQA rows memorise → `--ft-lr`, `--ft-epochs` (T21). Long phase-1 training memorises MedMCQA too → read the phase-1 validation curve (T25).
- A trailing gradient-accumulation group used to leak into the next epoch (T23).
- `load_dataset` used to stall on hub checks → loading is offline-first (T19). Do not re-enable online checks by default. The empty-cache fallback must flip `datasets.config.HF_HUB_OFFLINE` (not only the `HF_DATASETS_OFFLINE` alias) or a fresh machine cannot download (T26); with `transformers` 4.x it must also flip `transformers.utils.hub._is_offline_mode` (T31). Always test a download path with an **empty** `HF_HOME` — a warm cache hides these bugs.
- v2: the shared pod has 4× A100 80 GB used by other people; GPU 0 is often nearly full. Pin each run with `CUDA_VISIBLE_DEVICES`; `train_gen.py` refuses to start below `--min-free-gb` (the v1 sandbox OOM).
- v2: the pod has PyTorch 2.2.2; `transformers` 5.x needs ≥ 2.6 and silently disables its models, so it is pinned `<5` (T30). Do not raise the pin without upgrading PyTorch on the pod.
- v2: `transformers` 5 loads SmolLM2 in bf16; `load_backbone` casts to float32 or the float32 add-on crashes (`test_load_backbone_returns_float32`).
- v2: L2 sits after SmolLM2's final norm, so its output must be normalised before the gate (T32) — never remove `GatedAddOn.out_norm`, and keep the debate pre-norm on (`sandbox_prenorm`, T33): without it the consensus compounds to overflow. Keep the add-on LR at 1e-4 (T34); watch `l2=` in the sandbox log — it must stay roughly flat. Non-finite steps are skipped, not applied.
- v2: a perplexity gap is real only if the paired-bootstrap 95 % CI excludes 0 (`gen_eval compare`, token-weighted like the headline perplexity); otherwise it is a tie.
- v2 fairness: gradients are clipped separately for the backbone and the add-on (`clip_grads`), so the add-on can never shrink the backbone's update relative to the baseline arm.
- **Never compare with 25 % chance** (v1). Compare with the majority rate: MedMCQA val 32.11 % (A), `medqa_bench` 25.69 % (B), `medqa_test` 27.73 % (A), and quote `pred_counts` (T27).
- The tokenizer lower-cases, so the answer tokens are `a b c d` and `a` is also the English article. Keep this in mind when reasoning about letter priors; changing it means retraining both arms.
- Sandbox statistics must ignore padding, and the `lowest_energy` readout must be chosen per position (T24).
- v1's A/B is *architecture*-matched, not *parameter*-matched — always say so next to any v1 comparison. So is v2's (+19.59M for the sandbox arm); the same-size control was removed on 2026-10-23 at the user's request — do not re-add it without asking.
