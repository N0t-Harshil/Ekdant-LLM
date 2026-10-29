> Data for the evaluated build: **MedMCQA then MedQA**, via `09-Code/src/ekdant/data.py`. Thesis Ch. 4 source.
> An earlier design for a general 54-dataset pre-training corpus (code/web/docs mixture, 18 doc scrapers, 8-stage pipeline) was dropped when the project narrowed to medical multiple-choice QA; it lived outside this repo and is not part of any result.

# Data Pipeline (Ekdant) — Thesis Ch. 4 source

## v2 data (current, `gen_data.py`)

```
PubMedQA pqa_artificial (211,269) ─ seeded hold-out ─┬─ 209,269 ── phase 1: question + abstract → conclusion
                                                      └─ 2,000 ── validation every 25 % of the epoch · retention
MedQuAD (16,407 rows, 14,345 unique questions) ─ split by question (seed 17) ─┬─ 50 %: fine-tune half ─┬─ 90 % of its questions ── phase 2
                                                                              │                        └─ 10 % ── checkpoint selection
                                                                              └─ 50 %: benchmark half ── scored once
```

| Source | Fields used | Notes |
|---|---|---|
| `qiaojin/PubMedQA`, `pqa_artificial`, train | `question`, `context.contexts` (joined), `long_answer` | abstract p50 324 / p99 627 tokens; conclusion p50 47 / p99 130 (SmolLM2 tokenizer); ~85M tokens per epoch. `pqa_labeled` shares 0 PubMed ids with it (unused, available as a later test) |
| `keivalya/MedQuad-MedicalQnADataset`, train | `Question`, `Answer`, `qtype` | 3,474 rows share a question; answer p50 182 / p90 529 / p99 1,636 / max 5,646 tokens; 1,761 answers exceed 512 |

- **Prompts:** `Question: {q}\nContext: {abstract}\nAnswer:` (PubMedQA) and `Question: {q}\nAnswer:` (MedQuAD); target ` {answer}` + EOS.
- **Labels:** −100 on every prompt token — the loss and perplexity cover answer tokens only.
- **Caps:** answers cut at 512 tokens (no EOS when cut; 89 % of MedQuAD answers fit whole); the abstract is trimmed from its end so a row fits 1,024 tokens. Rows that still do not fit are dropped and counted.
- **Split by question:** keys are `question.strip().lower()`; one seeded permutation over the sorted keys, so rows sharing a question stay together and every arm gets identical sets. The same disease can still appear in both halves under different question types.
- **Empty rows** (blank question or answer) are dropped at load.
- **Tokenizer:** SmolLM2's own (49,152 entries, case-preserving); pad = EOS (id 0).
- Loading is offline-first, like v1 (T19/T26).

The rest of this note describes the v1 MCQ data (`--task mcq`).

## Overview

Two public datasets, used **strictly in sequence, never jointly**. The same code path serves both A/B arms, and the tokenizer is trained once and reused so both see the identical vocabulary.

```
MedMCQA train  (182,822 raw → 182,496 labelled)  ── phase 1 ──▶ train
MedMCQA val    (4,183 → 4,170 usable)            ── never trained on ──▶ phase-1 score · checkpoint selection · retention score
MedQA train    (10,201) ─ seed-locked permutation ─┬─ 60 % = 6,106  ── phase 2 ──▶ fine-tune
                                                   └─ 40 % = 4,072  ── never trained on ──▶ benchmark (scored once)
MedQA test     (1,273) ── never trained on ──▶ second untouched number
```

## §1 Datasets

| Dataset | Source | Rows used | Role |
|---|---|---|---|
| MedMCQA train | `openlifescienceai/medmcqa` | 182,496 (326 unlabelled rows dropped from 182,822) | phase-1 training |
| MedMCQA validation | same | 4,170 usable of 4,183 (13 dropped: question under 12 characters) | phase-1 result, per-epoch checkpoint selection, post-fine-tune retention |
| MedQA train 60 % | `GBaker/MedQA-USMLE-4-options` | 6,106 | phase-2 fine-tune |
| MedQA train 40 % | same | 4,072 | `medqa_bench` — the decision gate, scored once |
| MedQA test | same | 1,273 | `medqa_test` — official, untouched |

**Never score MedMCQA's public test split.** It ships `cop = -1` for all 6,150 rows (no released labels). An earlier evaluator did score it, and "accuracy" collapsed to each model's rate of predicting "A" (Known Issues T18).

The 60/40 MedQA partition comes from one `torch.randperm` seeded with `--seed` (default 17), so both arms train on identical rows and are benchmarked on identical rows (`train.py::split_frac`, covered by `tests/test_eval_split.py`).

## §2 Prompt and target

`format_prompt(question, options)` renders the question, four lettered options, then `Answer:`; the training target is the answer letter that follows.

```
<question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Answer:
```

Evaluation reads the logits at the one position that predicts the letter token (`answer_pos - 1`) and takes the argmax over the four letter-token ids — no free-form generation and no option re-scoring.

**Chance is not 25 %.** The answer letters are not balanced, so a model that always outputs the most common letter scores the *majority rate*: MedMCQA validation **32.11 %** (always A), `medqa_bench` **25.69 %** (always B), `medqa_test` **27.73 %** (always A). Compare every accuracy with these, and check `pred_counts` — a model piled on one letter has learned only the label prior (T27).

Gold-letter balance: MedMCQA train A 29.29 % (most common); MedQA train close to balanced (60 % fine-tune part: A 25.25, B 26.33, C 25.20, D 23.21 %).

## §3 Loading and normalisation (`data.py`)

- **`load_examples(name, split, limit, seed=17)`** loads either dataset and normalises each row to `MCQExample(question, options, answer)`. Four non-empty options are required; `normalise_options` handles dict and list forms; MedMCQA's flat `opa..opd` + `cop` (0-indexed int) schema is handled explicitly (the old MedQA-shaped reader produced **0 examples** — T16). Answers resolve through `letter_from_answer` (letter, int index, digit string or option-text match); questions under 12 characters are dropped; an optional seeded subsample supports `--limit` smoke runs.
- **Offline reads.** Datasets are downloaded once and then read from the local cache (`HF_HUB_OFFLINE=1` by default); `load_examples` goes online for exactly one attempt when the cache is empty. This removed silent training hangs caused by hub revalidation (T19).
- **`MCQDataset`** tokenizes each example once (`prefix + " {answer}" + EOS`), caches `answer_pos` / `answer_id` / `letter` per row, and drops examples longer than `max_len` (the config's `max_seq_len`, 2048) or with an empty answer encoding.
- **`collate`** right-pads to the batch maximum, builds `attention_mask` and `labels` (pad → −100), and carries `answer_pos` and `answer_target`.
- **`LETTER_IDS` / `bind_letter_ids(tokenizer)`** resolve A/B/C/D to vocabulary ids (bare letter, `" A"`, `"\nA"`, else first sub-token; never the pad id).

## §4 Tokenizer

`Tokenizer.train` wraps HF `tokenizers`: BPE (`unk_token="<unk>"`), NFD + lowercase normaliser, whitespace pre-tokenizer, `BpeTrainer(vocab_size=32768, min_frequency=2)` — on the 20,000-prompt sample it learns **26,766** entries, so 6,002 of the model's 32,768 embedding rows (3.07M parameters per arm) are never used, specials `<pad> <bos> <eos> <unk>`. It is trained once per run directory on 20,000 MedMCQA training prompts (the same seeded sample for both arms), saved to `<out>/tokenizer.json`, and reused for phase 2 and on resume. To guarantee a byte-identical vocabulary across arms, copy the first arm's `tokenizer.json` into the second arm's `--out` directory before starting it; a present file is loaded instead of retrained.

## §5 Length audit

Measured with the real tokenizer on all four splits:

| Split | p50 tokens | p99 tokens | max |
|---|---|---|---|
| MedMCQA | 34 | 118 | — |
| MedQA | 173 | 364 | 857 |

**0 of ~198k rows exceed the 2048 cap**, so nothing is truncated or dropped. Because the answer letter is 1 token in ~34 (MedMCQA) or ~173 (MedQA), plain LM cross-entropy gives the scored decision only ~1–3 % of its gradient — the reason for `--answer-weight` (T22).

## §6 Commands

```bash
cd 09-Code
pip install -e .[dev] && pytest -q                         # 201 tests, includes the data and split tests
python -m ekdant.train --variant sandbox --out /tmp/sbx --limit 400 --epochs 1 --batch-size 4   # smoke
python -m ekdant.train --variant sandbox|baseline --out runs/<variant> ...                        # real run (see run.md §3)
```

Connections: prev [Ch.3 (04-Model-DeepDive)](../04-Model-DeepDive/22_Thesis_Ch3_Layer2_Sandbox.md) · next [07_Training_Pipeline (06-Training-Eval)](../06-Training-Eval/07_Training_Pipeline.md)
