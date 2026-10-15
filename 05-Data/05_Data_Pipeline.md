> Consolidated from 7 files (pipeline overview, stages 1–8, cache, builder, registry weights/catalog). Thesis Ch. 4 source.

# Data Pipeline (Ekdant) — Thesis Ch. 4 source

## Overview

The data pipeline is the most mature subsystem in this codebase. ==54 primary dataset entries== (+4 fallback-only, 58 registered) across ==8 categories== flow through ==8 sequential stages== from the HuggingFace Hub to tokenized, packed training samples (packing target ==<5% padding==), supporting pretraining, SFT, and preference optimization with quality gates at every stage. Never bypass the registry (`data.use_registry: true`).

```
Raw Datasets (54 primary + 4 fallback-only, 8 categories)
  ▼ Stage 1  Registry          build_registry() — 54 primary entries (src/data/registry.py)
  ▼ Stage 2  Streaming         stream_dataset_with_fallbacks(), 4 driver families (streaming.py + drivers.py)
  ▼ Stage 3  Quality Filtering document_quality_score(), 8-component analysis (quality.py + pipeline.py)
  ▼ Stage 4  Code Processing   AST filter + function sampling (ast_filter.py + function_sampler.py)
  ▼ Stage 5  Boilerplate       license-header stripping, file-pattern skip (pipeline.py)
  ▼ Stage 6  Weighted Sampling WeightedMixedDataset, adaptive resampling
  ▼ Stage 7  Packing           EOS-separated, random window, metadata (pack_sequences)
  ▼ Stage 8  Tokenization      HF tokenizer, vocab_size=32K (S/M; pipeline supports 64K/128K)
  ▼ Disk Cache                 SHA256-keyed, stage-specific, save_to_disk
  ▼ DataLoader → Training
```

Module paths (`src/data/...`) are **thesis design** — the shipped training code uses the simpler `09-Code/src/ekdant/data.py` MCQ pipeline (§10). Model details live in [02_System_Architecture.md](../02-Architecture/02_System_Architecture.md).

**Registry-mode flow** (`use_registry: True`): `build_registry()` creates the 54 primary entries → per entry `detect_driver()` picks the driver family, then `stream_dataset_with_fallbacks()` loads samples (warm runs skip HF resolution; script-family resumes iterator) → filter pipeline (boilerplate removal → length check → quality scoring → AST filter → exact dedup → simhash dedup → function sampling) → cleaned texts tokenized via `tokenizer(text)["input_ids"]` → packed via `pack_sequences()` → cached to disk (SHA256-keyed) → combined into `WeightedMixedDataset` with adaptive sampling → sanity checks + health report (per-dataset and global stats).

## §1 Dataset Registry (`src/data/registry.py`)

**Entry point**: `build_registry()` → `DatasetRegistry` with ==54 primary entries== (58 incl. fallback-only), built by 8 registration helpers (one per category). Each entry is a `DatasetInfo` dataclass: HuggingFace path/name/split, category, weight (fraction of total training tokens), quality score (0.84–0.99), fallback chains, text fields, language, domain, function-sampling flag, priority, licensing.

### Category weight targets (weights normalized to ==1.0==)

| Category | Weight | # Entries | Primary sources |
|---|---|---|---|
| `code` | 0.30 | 22 | Programming code across 19 languages — OPC FineWeb Code, Stack v2, CodeSearchNet, CodeParrot, CodeContests |
| `web_text` | 0.20 | 1 | General web text — FineWeb, fallbacks: Dolma, RedPajama, Falcon RefinedWeb |
| `docs` | 0.15 | 19 | Documentation — FineWeb-Edu + the 18 scraped doc sources (§9) |
| `wiki` | 0.10 | 1 | Wikipedia (English, 20231101) |
| `math` | 0.10 | 6 | Mathematical reasoning and proofs — OpenWebMath, NuminaMath, MathPile, FineWeb-Edu proxy, LeanDojo |
| `science` | 0.05 | 1 | Scientific literature proxy (FineWeb-Edu) |
| `books` | 0.05 | 2 | Books proxies (FineWeb-Edu, FineWeb) |
| `structured_knowledge` | 0.05 | 2 | Structured knowledge proxies (FineWeb-Edu, FineWeb) |

**Weight normalization**: `DatasetRegistry.normalize_weights()` (`src/data/registry.py:126`) matches `CATEGORY_WEIGHTS` — group entries by category → sum current weights → scale each by `target / current` → adjust last entry per category for floating-point rounding → report final total (should be ==1.0==; drift fixed on the last entry).

**Fallback resolution** (`stream_dataset_with_fallbacks()`, `src/data/streaming.py:471`): chain `[primary] + [lookup(fallbacks)]`; try in order, skip already-tried paths; first non-zero result wins, logged via `registry.log_fallback()`; all fail → logged error, yields nothing. Current chains:

- `opc-fineweb-code` → code-search-net → codeparrot-clean
- `fineweb` → dolma → RedPajama → falcon-refinedweb (3-tier chain)
- `NuminaMath-CoT` ↔ `NuminaMath-1.5` (bidirectional)
- `MathPile` → open-web-math · `code_contests` → codeparrot-clean
- All 18 doc sources → FineWeb-Edu (`python-docs` json path → fineweb-edu)

### Per-dataset catalog (54 entries by category)

**Code (30%)**

| Dataset | Weight | Quality | Fields / Lang / Domain | Fallbacks / Notes |
|---|---|---|---|---|
| `OpenCoder-LLM/opc-fineweb-code-corpus` | 0.045 (15% code) | 0.90 | text; algorithms; MIT; pri 10 | code-search-net, codeparrot-clean; replaces removed FineWeb-Code, permissive high-quality |
| `bigcode/the-stack-v2-dedup` (18 lang entries) | ~0.09 total (30% code) | 0.88 | content,text; backend; pri 5; function sampling | code-search-net, codeparrot-clean (sql: b-mc2/sql-create-context); Python 0.25, C++ 0.15, JS 0.10, TS 0.08, Java 0.08, Rust 0.07, Go 0.06, SQL 0.05, Shell 0.04, C# 0.04, PHP 0.03, C 0.02, others 0.01; targets `CODE_LANG_TARGETS` (`registry.py:166`) |
| `code-search-net/code_search_net` | 0.06 (20% code) | 0.92 | code, func_code_string, whole_func_string, text; backend; pri 10; function sampling | namespaced id of legacy `code_search_net` (same repo, `hf://` compatible) |
| `codeparrot/codeparrot-clean` | 0.06 (20% code) | 0.84 | content,text; python/backend; MIT; pri 10; function sampling | fallback target for other code datasets; cleaned GitHub Python |
| `deepmind/code_contests` | 0.045 (15% code) | 0.90 | description, solutions, text; algorithms; Apache-2.0; pri 20 | codeparrot-clean; competitive programming + tests |

**Web Text (20%)** — `HuggingFaceFW/fineweb`, 0.20, quality 0.93: text; web; ODC-BY; pri 1; fallbacks allenai/dolma, RedPajama-1T, falcon-refinedweb; primary web corpus.

**Documentation (15%)** — 18 doc-builder sources (`json` path, pri 5, fallback fineweb-edu, quality 0.90–0.99) plus a FineWeb-Edu anchor:

| Source | Language | Weight fraction | Quality | Domain |
|---|---|---|---|---|
| python-docs | python | 0.020 | 0.99 | backend |
| pytorch-docs | python | 0.020 | 0.99 | ml |
| numpy-docs | python | 0.010 | 0.98 | ml |
| rust-book | rust | 0.010 | 0.98 | backend |
| go-docs | go | 0.010 | 0.97 | backend |
| mdn-docs | javascript | 0.010 | 0.95 | web |
| fastapi-docs | python | 0.005 | 0.96 | backend |
| cuda-docs | cpp | 0.005 | 0.95 | ml |
| linux-kernel-docs | text | 0.005 | 0.94 | os |
| opencv-docs | cpp | 0.005 | 0.93 | cv |
| kubernetes-docs | shell | 0.005 | 0.92 | cloud |
| docker-docs | shell | 0.005 | 0.91 | cloud |
| postgresql-docs | sql | 0.003 | 0.92 | backend |
| sqlite-docs | sql | 0.003 | 0.91 | backend |
| cudnn-docs | cpp | 0.003 | 0.92 | ml |
| onnx-docs | python | 0.003 | 0.91 | ml |
| rfcs | text | 0.003 | 0.90 | protocols |
| lang-specs | text | 0.003 | 0.90 | languages |

Plus `HuggingFaceFW/fineweb-edu` (docs fallback, sample-10BT): ~0.02 (3% docs), quality 0.92, text; general; ODC-BY; pri 10 — fallback for all 18 doc sources when the JSONL is not built.

**Wiki (10%)** — `wikimedia/wikipedia` 20231101.en, 0.10, quality 0.95: text; web; CC-BY-SA; pri 1; Nov-2023 English dump; largest non-weight-limited entry.

**Math (10%)**

| Dataset | Weight | Quality | Notes |
|---|---|---|---|
| `open-web-math/open-web-math` | 0.042 (42% math) | 0.92 | text, problem, solution; science; pri 10; largest math, general web math |
| `AI-MO/NuminaMath-CoT` | 0.02 (20%) | 0.90 | text, problem, solution; fallback NuminaMath-1.5; CoT reasoning |
| `AI-MO/NuminaMath-1.5` | 0.015 (15%) | 0.91 | text, problem, solution; fallback NuminaMath-CoT (bidirectional) |
| `GAIR/MathPile` | 0.010 (10%) | 0.93 | text, problem, solution; fallback open-web-math; pri 10; ⚠️ GATED — needs HF access approval |
| `HuggingFaceFW/fineweb-edu` sample-10BT (math proxy) | 0.01 (10%) | 0.85 | text, problem, solution; fallback open-web-math; edu web as math proxy |
| `akjadhav/leandojo-lean4-formal-informal-strings-split` | 0.003 (3%) | 0.93 | text, problem, solution; fallback open-web-math; Lean 4 formal/informal, theorem proving |

**Science (5%)** — `fineweb-edu` sample-10BT (science proxy), 0.05, quality 0.85: text, title, abstract; science; S2ORC/PubMed/OpenAlex/ACL removed (datasets 2.20+ incompat).

**Books (5%)** — `fineweb-edu` sample-10BT (books proxy) 0.025 (50%), quality 0.88, long context true; `fineweb` (books proxy) 0.015 (30%, adj. from 50%), quality 0.85, pri 10 — PG19/Gutenberg removed (compat).

**Structured Knowledge (5%)** — `fineweb-edu` sample-10BT (knowledge proxy) 0.025 (50%), quality 0.85, pri 15; `fineweb` (knowledge proxy) 0.025 (50%), quality 0.82, pri 15 — WIT/Wikidata/DBpedia/ConceptNet/WordNet removed (compat).

## §2 Stage 2 — Streaming (`streaming.py` + `drivers.py`)

**Entry point**: `stream_dataset_with_fallbacks(info, registry, limit)` → `Generator[dict]`.

**Driver families** (`detect_driver()`; zero network on warm run, next 2 drivers warmed in background, logs `[DIAG] <family> driver: (hits)`):

| Family | Detected when | Behavior |
|---|---|---|
| `LocalDatasetDriver` | Existing local dir with recognized files | Never touches Hub; scans/loads locally |
| `FileDatasetDriver` | Metadata record verifies | Direct Arrow reuse, shard-parallel, no HF resolution |
| `ScriptDatasetDriver` | Builder verifies / no shard sources | `load_dataset_builder()` + iterator resume at exact raw row |
| `StreamingDatasetDriver` | Builder inspection fails (gated/not-found/offline) | Cold path; logs guidance once, short-circuits |

- **Dataset types**: `Dataset` / `IterableDataset` iterate directly; `DatasetDict` / `IterableDatasetDict` extract `ds[split]`.
- **Text fields** (`detect_text_fields`): known_fields → TEXT_FIELD_PRIORITY (text, content, body, code, document…) → any string >50 chars → `["text"]`.
- **Gated detection** (`_is_gated_error`): keywords gated/access/permission/401/403 → check `HF_TOKEN`, print access instructions.
- **Output**: `{**original_sample, "text": extracted_text}` — preserves file paths, language, scores.

## §3 Stage 3 — Quality Filtering & Dedup (`quality.py` + `pipeline.py`)

**Entry point**: `document_quality_score(text, category, language)` → `{"final": float, ...components}` (8 components):

| Component | Logic | Range |
|---|---|---|
| `length` | `min(1.0, len/50000)` | 0.0–1.0 |
| `perplexity` | word-len avg/var/vocab → est. ppl, `1.0-ppl/200` | 0.0–1.0 |
| `language_conf` | regex signatures, 13 langs, hit ratio | 0.0–1.0 |
| `formatting` | line-len/var/blank/indent/unique/punct/caps | −0.5–1.0 |
| `code_quality` | py compile, hints, tests, ident-diversity, comments, func-count | −0.5–1.0 |
| `doc_completeness` | headers, code blocks, URLs, links, words | 0.0–1.0 |
| `toxicity` | `1.0 - hits*0.25` | 0.0–1.0 |
| `exact_duplicate` | MD5 first 500 chars | 0.0/1.0 |

**Per-category component weights**: `code`: code_quality 0.35, lang 0.15, format 0.15, ppl 0.15 · `docs`/`wiki`: doc_complete 0.35, format 0.20, ppl 0.15 · `math`/`science`: ppl 0.35–0.40, format 0.15, doc 0.15–0.20 · `web_text`: ppl 0.30, format 0.25, lang 0.20 · `books`: ppl 0.40, format 0.20, lang 0.15.
**Thresholds** (`QUALITY_THRESHOLDS`): code 0.35, math 0.30, science 0.30, web_text 0.35, books 0.40, docs 0.35, wiki 0.35, structured 0.30.

**Dedup (3 strategies + 1 heavy option)**: Exact — MD5 first 1000 chars, normalized, global first-pass; MinHash — 128-perm, 5-gram, Jaccard ≥0.85; SimHash — 64-bit, cosine ≥0.85; Semantic — MiniLM-L6-v2, cosine ≥0.92 (heavy).

**Contamination filter** (`ContaminationFilter`) — benchmark signature patterns:

| Benchmark | Patterns |
|---|---|
| HumanEval | `def check*`, `def test_`, `assert.*==` |
| MBPP | `""" >>>`, `def check*` |
| MMLU | `Answer: [A-D]`, `Question N:`, `(A)` |
| GSM8K | `#### number`, `Let's think step by step`, `<<...>>` |
| ARC | `grid = [[`, `A. [`, `output_grid` |

## §4 Stage 4 — Code-Specific Processing (`ast_filter.py` + `function_sampler.py`)

**Entry point**: `filter_code(text, language, ast_filter_cfg)` → `(ok, reason)` — code category only.

- **AST filter**: `identifier_ratio` (valid ident/token); `executable_ratio` (`ast.parse()` sliding windows). Rejects autogenerated code ("auto-generated", "do not edit", "generated by"), one-liners (>95% single-token lines), low-ident (<5%), low-exec (<10%).
- **Function sampling** (Stack v2, CodeSearchNet, CodeParrot only): `extract_functions` regex — Python `def`+indent, JS/TS `function`+braces, Rust `fn`+braces, else indent heuristic. `sample_functions_or_fallback`: ≤`per_file_limit` funcs, `min/max_body_lines` filter, fallback to original if none survive/fail. `is_code_empty_or_trivial` drops empty/whitespace/single-line-trivial.

## §5 Stage 5 — Boilerplate Removal (`pipeline.py`)

**Entry point**: `remove_boilerplate(text, keywords)` → `str`. Scans first 50 lines for 20 license/codegen keywords: copyright, license, spdx, apache, mit license, licensed under, all rights reserved, gnu general public, bsd license, mozilla public, permission is hereby granted, this file is part of, generated by, auto-generated, do not edit, this code was generated.
**Skip patterns** (`BOILERPLATE_FILE_PATTERNS`, 20 patterns): `_pb2.py`, `_pb2_grpc.py`, generated/, build/, dist/, vendor/, node_modules/, .git/, __pycache__/, .min.js/.css, .bundle.js, package-lock.json, yarn.lock, go.sum, poetry.lock, .tfstate, .parquet, .arrow, .o/.obj/.exe/.dll/.so.
**Cleaning**: collapse 4+ newlines → 2; strip trailing whitespace/line; final `.strip()`.

## §6 Stage 6 — Weighted Sampling (`WeightedMixedDataset`)

**Entry point**: `WeightedMixedDataset(datasets_with_weights, total_samples, ...)`.
**Adaptive resampling**: `effective_weight = base_weight × quality_score × remaining_token_ratio`, where `remaining = max(0, size-consumed)/max(size, 1)` — renormalized every sample, shifting weight toward under-consumed datasets.
**Stratification**: language (10 langs, e.g. python 0.35, javascript 0.12, typescript 0.08…); domain (22 domains: backend, frontend, ml, cv, nlp…); or none (pure weighted random).
**Reproducibility**: `numpy.random.default_rng(seed)` from `project.seed` (default 42). **Tracking**: `stats()` reports per-dataset weights, consumed counts, remaining ratios.

## §7 Stage 7 — Sequence Packing (`pack_sequences`)

**Entry point**: `pack_sequences(tokenized_samples, max_seq_length, eos_token_id)` → `(packed, packing_eff)`.
**Algorithm**: sample > max_len → `random_window_sample()`; buffer has room (incl. EOS) → append with EOS; else pad to max_len, emit, start new buffer; end → emit final (padded).
**EOS separation**: EOS inserted between segments; EOS label = `-100` (ignored in loss). **Random window**: `start = randint(0, len-max_len); tokens[start:start+max_len]` — no start-truncation bias.
**Sequence metadata**: `input_ids` (concat + EOS), `labels` (same; pad/EOS → −100), `attention_mask` (1 real / 0 pad), `_segments` (doc count), `_avg_quality` (mean score). Hub rule: packing also carries `_dataset`/`_category` — ==never strip these metadata fields==.
**Efficiency**: `packing_eff = 1.0 - padded/total_slots`; target ==<5% padding==.

## §8 Stage 8 — Tokenization & Disk Cache

**Entry point**: HF `PreTrainedTokenizerBase` (transformers).

| Config | vocab_size | max_seq_length | Notes |
|---|---|---|---|
| `config_foundation.yaml` | 64000 | 2048 | Foundation pretraining |
| `config.yaml` | 128000 | 4096 | Full training (4× A100) |
| `config_small.yaml` | 128000 | 2048 | Single-GPU debug |

**Model**: `Xenova/claude-tokenizer`, BPE; specials `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>` plus instruction markers; fields `input_ids`, `labels`, `attention_mask`, `_segments`, `_avg_quality`. **Modes**: pretraining `input_ids`=`labels` (causal); SFT prompt masked to −100; preference `chosen_*`/`rejected_*` pairs.

**Cache layer**: `_cached_dataset_path(ds_info, stage_name)` → `Path`. Key: `sha256(f"{path}_{max_samples}_{max_seq_length}_{vocab_size}_v3_foundation")[:16]`. Store: `{cache_dir}/tokenized/{stage_name}/{prefix}/` via `save_to_disk()`/`load_from_disk()` per dataset; invalidated on config change; warm runs load from cache; corrupt cache → re-stream.

**Config knobs (condensed)**: `data.quality.*` (min_length 30, max_length 500000, dedup minhash/0.85, contamination 7 benchmarks); `data.preprocessing.*` (remove_boilerplate true, min_text_length 100, 20 license keywords, 20 file patterns); `data.ast_filter.*` (code_filtering, reject_no_parse/autogen, one-liner 0.95, ident 0.05, exec 0.10); `data.function_sampling.*` (disabled by default, body 3–500 lines, 20/file).
**Throughput (4× A100, 200GB cache)**: registry ~0.04s; streaming ~10K/s/dataset (net/HF limits); quality ~5K/s; AST ~50K/s; simhash ~20K/s; tokenize ~15K/s; packing ~100K seqs/s; cache ~50MB/s.
**Resilience**: empty dataset → logged/skipped with fallback; gated → user instructions; all-failed → `RuntimeError` + checklist; partial failures don't crash; network → 3 retries; schema drift → auto-detection.

## §9 Documentation Builder — 18 Web Scrapers (`src/data/doc_builder.py`)

Multi-threaded web-scraping framework for the docs corpus: 18 sources across languages, frameworks, tools, and specs. Each source is a scraper class inheriting a common abstract base, sharing HTTP, extraction, URL-normalization, and failure-tracking infrastructure. All 18 register in the `SCRAPERS` dictionary (`doc_builder.py:839`).

| # | Key | Class | Language | Domain | Pages |
|---|---|---|---|---|---|
| 1 | `python` | `PythonDocScraper` | python | backend | ~1000+ |
| 2 | `pytorch` | `PyTorchDocScraper` | python | ml | ~500+ |
| 3 | `numpy` | `NumPyDocScraper` | python | ml | ~500+ |
| 4 | `fastapi` | `FastAPIDocScraper` | python | backend | ~200+ |
| 5 | `opencv` | `OpenCVDocScraper` | cpp | cv | ~200+ |
| 6 | `onnx` | `ONNXDocScraper` | python | ml | ~100+ |
| 7 | `docker` | `DockerDocScraper` | shell | cloud | ~12* |
| 8 | `kubernetes` | `KubernetesDocScraper` | shell | cloud | ~7* |
| 9 | `postgresql` | `PostgreSQLDocScraper` | sql | backend | ~12* |
| 10 | `sqlite` | `SQLiteDocScraper` | sql | backend | ~200+ |
| 11 | `mdn` | `MDNDocScraper` | javascript | web | ~500+ |
| 12 | `rust-book` | `RustBookScraper` | rust | backend | ~200+ |
| 13 | `go-docs` | `GoDocScraper` | go | backend | ~200+ |
| 14 | `cuda` | `CUDADocScraper` | cpp | ml | ~300+ |
| 15 | `cudnn` | `CuDNNDocScraper` | cpp | ml | ~66 |
| 16 | `rfcs` | `RFCDocScraper` | text | protocols | ~9000+ |
| 17 | `linux-kernel` | `LinuxKernelDocScraper` | text | os | ~58 |
| 18 | `lang-specs` | `LangSpecScraper` | text | languages | ~18 |

*JS-rendered navigation limits static crawling (see limitations). Page counts reflect the last crawl; regenerate with `python src/data/doc_builder.py`.

**Class hierarchy**: `DocScraper` (ABC, `doc_builder.py:147`) → `SphinxScraper` (`:267`; Python, PyTorch, NumPy, FastAPI, OpenCV, ONNX, Docker, Kubernetes, PostgreSQL, Go) plus direct subclasses `SQLiteDocScraper`, `MDNDocScraper`, `RustBookScraper`, `RFCDocScraper`, `LinuxKernelDocScraper`, `LangSpecScraper`, `CUDADocScraper`, `CuDNNDocScraper`.

**DocScraper base**: `_fetch(url, session)` — retry logic (4 attempts, no retry on 4xx except 429), 30s timeout, client-side redirect following (meta refresh + `location.replace`), returns `(html, final_url)` or None; `_clean_text(html)`; `_extract_section_text(html, section_tag, fallback)`; `_normalize_url(url)` (strip fragments, normalize trailing slashes); `_has_template_syntax(text)`; `_is_invalid_href(href)`; `scrape()` generator — yields `DocPage`, tracks visited/failed URLs, enforces `MAX_PAGES`/`MIN_TEXT_LENGTH`/`MAX_TEXT_LENGTH`, re-applies template/href filters as defense-in-depth; `scrape_to_jsonl(output_path)` — JSONL fields `source, language, category, title, url, text, text_length`; `get_config()`/`SourceConfig` dataclass (base URL, language, category, selectors, length limits, timeout, retry count).

**Template URL filtering (security hardening)**: all `discover_urls()` reject template placeholders (`{{ }}`, `{% %}`, `${ }`, `<% %>`) and `javascript:`/`void(0)`/`#` hrefs **before they enter the crawl queue**; each rejection is logged (`Filtered template URL`, INFO) and counted in `_filtered_count` (`filtered=N` in the per-source summary). Guarantees URLs such as `https://docs.docker.com/{{ meta.url | default(url) | safeUrl }}` never reach `_fetch()` or appear as "404 fetching …" failures.

**SphinxScraper**: `URL_PATTERNS` seed paths (e.g. `"tutorial/"`, `"library/"`), `EXCLUDE_PATTERNS` (`.pdf`, `_sources`, `genindex`), `CONTENT_SELECTOR`/`CONTENT_FALLBACK` (default `main`/`body`), `discover_urls()` seeds from BASE_URL + patterns then follows same-domain links.

**URL discovery strategies** (one per scraper where non-generic):

- **SphinxScraper** (Docker, Kubernetes, OpenCV, ONNX, NumPy, FastAPI, Go, Python): seeds + same-domain link following; template/invalid hrefs filtered pre-yield.
- **PyTorchDocScraper** (`:335`): seed URLs (tensors.html, torch.html, nn.html, optim.html…) + `genindex.html` alphabetical index for broader coverage.
- **MDNDocScraper** (`:517`): hardcoded `CORE_TOPICS` (HTML, CSS, JavaScript, HTTP, Web/API, Web/Guide, SVG, MathML, Web_Components, Events, Performance, Security, Accessibility); one-level discovery from `/en-US/docs/Web`; filters deprecated prefixes (old SVG path APIs, GestureEvent, Microsoft-specific APIs).
- **RFCDocScraper** (`:622`): parses the plain-text IETF index (`rfc-index.txt`) and constructs `https://www.rfc-editor.org/rfc/rfc{NUMBER}.txt` — avoids the 16MB Nuxt SSR HTML index.
- **PostgreSQLDocScraper** (`:442`): matches `https://www.postgresql.org/docs/` with any version path (`/docs/16/`, `/docs/current/`) — version-agnostic; excludes non-HTML resources.
- **LinuxKernelDocScraper** (`:651`): full recursive same-domain discovery; excludes PDFs and fragment-only links.
- **SQLiteDocScraper** (`:483`): fetches `https://sqlite.org/docs.html`, follows same-domain links, excludes `.pdf`, `.txt`, `.zip`, `.tar`.
- **LangSpecScraper** (`:689`): fully hardcoded URLs — 9 Python language-reference + 9 Rust reference pages; curated set, no discovery.
- **CUDADocScraper** (`:752`): 18 hardcoded seeds (programming guide, runtime/driver/math APIs, best practices, cuBLAS, cuRAND, cuSOLVER, cuSPARSE, cuFFT, NVRTC, NVML, Thrust, installation, NVCC, release notes, PTX, C++ guide); one level of `.html` link following. **Fix applied**: the href loop was previously outside the `for` loop (indentation bug) — only the last href per page was processed; all links now processed, so CUDA page counts increase on regeneration.
- **CuDNNDocScraper** (`:790`): 7 seeds under `https://docs.nvidia.com/deeplearning/cudnn/latest/` (api overview, graph/ops library, developer guide, release notes, installation, index); `.html` sub-link following; dual selector `article` → `main`.

**Text extraction** — three stages: (1) section selection via `_extract_section_text` regex on the primary tag with `<body>` fallback; (2) `_clean_text` regex removal of `<script>`, `<style>`, `<nav>`, `<footer>` blocks, strip all tags, collapse whitespace; (3) length filtering — discard < `MIN_TEXT_LENGTH` (default 200; 5000 RFCs, 500 Linux kernel, 3000 lang-specs), truncate > `MAX_TEXT_LENGTH` (default 100000; 500000 RFCs, 200000 Linux kernel). Linux kernel threshold was lowered 3000 → 500; `div.document` excludes the nav sidebar (`div.sphinxsidebar`), so nav/index pages still get filtered — verified on the server re-run.

| Scraper | Primary selector | Fallback |
|---|---|---|
| SphinxScraper (most) | `main` | `body` |
| MDNDocScraper | `article` | `main` |
| SQLiteDocScraper | `body` | none |
| LinuxKernelDocScraper | `div.document` | `body` |
| RFCDocScraper | plain text (no HTML cleaning) | — |

**Failure handling**: `_failed` set — once a URL fails (404/timeout/conn error) it is never re-fetched in the session; `_filtered_count` counts template/invalid URLs separately (never treated as failed requests); `_summary()` reports discovered/duplicate/filtered/failed counts; `scrape_all()` (`:861`) runs each scraper in its own thread with configurable timeout (default 600s) — over-timeout threads abandoned and the source reported timed out; `MAX_PAGES` checked at every yield point; `_find_client_redirect()` follows up to 4 hops of `<meta http-equiv="refresh">` / `location.replace()`/`location.href` redirects.

**Known limitations**: (1) PyTorch docs behind Cloudflare anti-bot — may return 0 pages despite genindex; fixes: readthedocs.io mirror or `cloudscraper`. (2) JS-rendered content (MDN SPA routing, RFC Nuxt SSR) may be missing without a headless browser. (3) Aggressive scraping can rate-limit Docker/Kubernetes/NVIDIA docs; retry-with-backoff mitigates partially. (4) The 16MB Nuxt SSR RFC index timed out — fixed via the IETF text index. (5) PostgreSQL docs moved to Bootstrap JS navigation — URL-validation fix accepts all version paths; page counts expected to rise next server run. (6) cuDNN seeds updated to `/latest/`; remove permanently-404 seeds rather than retrying. (7) Docker/Kubernetes Docusaurus/Hugo render deep nav client-side — coverage limited to static links (Docker ~12, Kubernetes ~7); no headless-browser integration. (8) OpenCV Doxygen cross-reference pages can't be followed statically; residual failures expected and non-blocking.

**Registry integration**: after scraping, `build_doc_registry()` (`doc_builder.py:925`) creates `DatasetInfo` entries — `path="json"` with `data_dir` pointing at the JSONL output, proportional weights from the `weights` dict (default: python 0.18, pytorch 0.13, mdn 0.15, rust-book 0.10, etc.), `quality_score=0.95`, `category="docs"` — registered into the main registry via `build_registry()`. `scrape_all()` is also called programmatically from `production_validation.py` Phase 2.

## §10 Shipped Code Note — `09-Code/src/ekdant/data.py`

The runnable training code does **not** ship the thesis-design 54-dataset registry. `09-Code/src/ekdant/data.py` is a simpler medical MCQ pipeline: MedMCQA (`openlifescienceai/medmcqa`, 182,822 train raw — 182,496 kept after dropping 326 unlabelled `cop=-1` rows — / 4,183 validation; the public test split ships `cop=-1`, unlabelled, never evaluate on it) then MedQA (`GBaker/MedQA-USMLE-4-options`, 10,178 train / 1,273 test), both reformatted into one prompt template and trained as a causal LM whose next token after `Answer:` is the correct letter. The answer-token index is tracked exactly, so evaluation reads the letter logits at one position instead of scoring four continuations. `load_examples` is **offline-first**: it defaults `HF_HUB_OFFLINE` so cached reads never revalidate against the hub (etag checks stalled training on bad networks — T19), going online only for the first-run download.

- **`load_examples(name, split, limit, seed=17)`**: loads either dataset, `_normalise_row` each row into an `MCQExample(question, options, answer)` — 4 non-empty options required (`normalise_options` handles dict/list), answer resolved by `letter_from_answer` (letter, int index, digit string, or option-text match, fallback "A"), question <12 chars dropped; optional seeded subsample.
- **`Tokenizer`**: thin wrapper over HF `tokenizers` — BPE (`BPE(unk_token="<unk>")`), NFD + lowercase normalizer, Whitespace pre-tokenizer, `BpeTrainer(vocab_size, min_frequency=2)`, specials `<pad> <bos> <eos> <unk>`; `train()`/`load()`/`encode(text, add_eos=True)`.
- **`MCQDataset`**: tokenizes each example once (`prefix` + `" {answer}"` + EOS), caches `answer_pos`/`answer_id`/`letter` per row, drops examples exceeding `max_len` (default 1024) or with empty answer encoding.
- **`collate(batch, pad_id=0)`**: right-pads to batch max, builds `attention_mask`, `labels` (clone of input_ids, pad → −100), plus `answer_pos` and `answer_target` tensors.
- **`LETTER_IDS` + `bind_letter_ids(tokenizer)`**: resolves each of A/B/C/D to a vocabulary id — tries bare letter, `" A"`, `"\nA"` surface forms, falls back to the first sub-token of an encode (never the pad id). `answer_accuracy(model, batch)` gathers logits at `answer_pos` and argmaxes over `LETTER_IDS[:n_options]`.

## §11 Commands

**Thesis design only — these scripts are documented spec and are NOT present in this vault** (`src/data/`, `scripts/` paths describe the intended thesis tooling):

```bash
python scripts/verify_datasets.py            # registry/dataset verification
python scripts/corpus_audit.py               # corpus audit
python scripts/production_validation.py      # 8-phase harness (Phase 2 = doc scraping, Phase 4/5/6 = data validation)
python src/data/doc_builder.py               # scrape all 18 doc sources
python src/data/doc_builder.py --sources python numpy fastapi
python src/data/doc_builder.py --max-per-source 500 --output-dir data/my_docs
```

Validation chain (design): `verify_datasets.py` → corpus audit → Phases 4/5/6 in [validation](../06-Training-Eval/09_Validation_System.md).

**Runnable in this vault** (`09-Code/`, uses `data.py`, §10):

```bash
pip install -e .[dev] && pytest -q
python -m ekdant.train --variant sandbox --limit 50     # data-load smoke test
python -m ekdant.train --variant sandbox|baseline       # full A/B run
```

Connections: prev [Ch.3 (04-Model-DeepDive)](../04-Model-DeepDive/21_Thesis_Ch3_Layer1_SSM_Storage.md) · next [07_Training_Pipeline (06-Training-Eval)](../06-Training-Eval/07_Training_Pipeline.md)
