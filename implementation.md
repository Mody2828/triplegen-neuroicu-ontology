# Implementation Summary: LLM-Driven Ontology Engineering for Neuro-ICU

This document gives a concise **project aim and objectives** and a **detailed summary of the implemented design and functionality** of the framework.

---

## 1. Project aim and objectives

### Aim

The project aims to **design, implement, and evaluate** an LLM-driven ontology engineering framework for the **Neuro-ICU / neuro-intensive care domain**, using **BrainIT literature** as the primary input. The central question is how well language models can reconstruct a high-quality domain ontology in a **low-resource clinical setting**, and which factors—prompting, constraints, post-processing, and choice of LLM—improve or degrade that outcome.

### Objectives

- **End-to-end pipeline:** Build a reproducible pipeline from raw text (paste or upload) to a generated ontology and evaluation report.
- **Multiple LLM providers:** Support and compare extraction across several APIs (OpenAI, Anthropic, Google, Groq, Hugging Face, DeepSeek) so that method and provider can be varied independently.
- **Several prompting methods:** Compare zero-shot (baseline), one-shot, few-shot (phased 2-step), and few-shot III (phased 3-step) extraction to assess the impact of examples and task decomposition.
- **Progressive pipeline modes:** Four clearly named modes (Strict → Guided → Schema-Completed → Fully Reasoned) group the optional post-processing features into meaningful experimental conditions for ablation.
- **Enhancement strategies:** Implement and evaluate optional post-processing improvements—schema-guided completion, an LLM reasoning layer, and a rule-based reasoning layer—as orthogonal toggles for ablation.
- **Evaluation against gold:** Evaluate generated ontologies against the **BrainIT gold standard** (ontology/schema) using coverage, precision, recall, **relation recall**, structural metrics, error taxonomy, and a new **per-stage ablation table**.
- **Usability and reproducibility:** Provide a web UI to run experiments, save artifacts per run, and compare runs side-by-side.

The framework is positioned as a **modular hybrid ontology engineering system** (prompt strategies + optional LLM and symbolic layers), not only a set of prompt experiments.

---

## 2. Implemented design and functionality

### 2.1 High-level architecture

The system is structured in **four conceptual layers**:

1. **Prompt strategy (extraction)** — How the LLM extracts from text: choice of prompting method and, where applicable, retrieval of examples.
2. **Merge and optional vocabulary filter** — Chunk-level extractions are merged into one ontology; optionally restricted to a gold vocabulary when evaluation controls are enabled.
3. **Post-processing** — Built-in cleanup (always when gold available; dedupe, scope/evidence/hierarchy pruning, axioms), then optional improvements: Schema-guided completion, LLM Reasoning Layer, and Rule-based Reasoning Layer, in that order.
4. **Evaluation** — Alignment to the gold standard, computation of metrics (including per-stage snapshots), and run summary.

Pipeline order:

```
Corpus (paste/upload) → Load (strip control chars → normalize → [scope filter if enabled])
    → Chunking (control chars stripped per chunk; candidates with section context)
    → Extraction (per chunk, by strategy; evidence required; Phase 2 output filtered)
    → Merge (build_ontology: evidence required; canonical alias map + singular/plural; dedupe by canonical key / (label, domain, range); stratum on entities; duplicate merge of provenance/synonyms/aliases)
    → [Vocabulary filter if filter_to_gold_vocabulary enabled]
    ──── STAGE SNAPSHOT: extraction ────
    → [Schema-guided completion]
    ──── STAGE SNAPSHOT: after_sgc ────
    → [Built-in cleanup]
    ──── STAGE SNAPSHOT: after_cleanup ────
    → [LLM Reasoning Layer (on clean ontology)]
    ──── STAGE SNAPSHOT: after_llm_reasoning ────
    → [Rule-based Reasoning Layer (optional: schema completion + orphan pruning)]
    ──── STAGE SNAPSHOT: after_rule_based ────
    → [Gold-vocab filter if eval_restrict_to_gold]
    ──── STAGE SNAPSHOT: after_gold_filter ────
    → Validation → Evaluation (class + relation + hierarchy metrics) → Artifacts & summary
```

Stage snapshots are written to `metrics["by_stage"]` in `metrics.json` and displayed in the **per-stage ablation table** in `summary.txt`.

![Pipline](screenshots/1.jpg)

---

### 2.2 Corpus input, normalization, scope filter, and chunking

- **Input:** Users can **paste text** or **upload one or more files** (.txt or .pdf). Multiple files are saved to `data/corpus_ui` and loaded as a multi-document corpus. **Use default paper** can be enabled so the run uses a built-in default corpus when no text is pasted or uploaded.

- **Load pipeline** (`src/corpus/ingest.py`): For each document: **raw text** → **strip control chars** (`clean_chars`) → **normalize** → **[scope filter if enabled]** → stored as `doc["text"]`. Optionally `doc["raw_text"]` (when `keep_raw_text` in config) and, for PDFs, `doc["pages"]` (page-level segments for provenance and header/footer detection).

- **Normalization** (`src/corpus/normalize.py`): Fixes broken hyphenation across line breaks (e.g. intra-\ncranial → intracranial), normalises symbols (Unicode fractions, fancy dashes, curly quotes, OCR repeats), removes repeated header/footer lines across pages/blocks, then strips form-feed and other control chars.

- **Scope filter (optional):** When enabled, scope filter runs **inside load_corpus** after normalize:
  1. **Document-level:** paragraph-level (drop whole paragraphs if admin-heavy), then line-level removal of admin headings/phrases and blacklist terms.
  2. **Chunk-level** (after chunking): **dual-score router** (`filter_chunks_to_clinical`):
     - `chunk_scores()` returns (admin_score, clinical_score).
     - **Drop** if section matches governance sections (Part A, group formation, database access, ethics approval, etc.); if `governance_dominance ≥ 2` and `clinical_score ≤ 1`; if `admin_score ≥ 2` and `clinical_score ≤ 1`.
     - **Keep** if `clinical_score ≥ 3`; **always keep** if section matches clinical dataset sections.
     - **Reorder**: clinical dataset sections ordered first.
  - **Blacklist terms** (`_SCOPE_BLACKLIST_TERMS`): compound phrases only — `"membership"`, `"centres"`, `"centers"`, `"quality control"`, `"steering group"`, `"ethics committee"`, `"consortium"`, `"data validation staff"`, `"validation workflow"`, `"publication criteria"`. Single generic words are **not** in the blacklist to avoid destroying legitimate clinical lines.
  - **Clinical scoring**: strong terms (heart rate, ICP, GCS, sedation, **sedation levels**, ventilation, **monitoring**, CPP, MAP, **fluids**, **fluid input and output**, **nutrition**, **condition**, etc.) count **1.0**; broad terms (core dataset, outcome) count 0.5. **"monitoring" is a strong term** (promoted from weak) because it is a core BrainIT clinical concept. **"fluids"**, **"nutrition"**, and **"condition"** were promoted to strong to prevent chunks containing only these brief therapy-type terms from being filtered out before reaching the LLM.

- **Scope filter config validation:** When a gold standard is loaded (benchmark mode), `scope_filter` **must be set explicitly** in the config. If the key is absent, `run_experiments.run_one()` raises a `ValueError` immediately. This enforces that UI runs (`scope_filter=True` by default) and CLI runs produce comparable results.

- **Chunking:** Documents are split into **semantic chunks** so that each chunk fits within prompt limits and stays coherent. Chunk text is passed through **strip control chars** again before candidate extraction and storage.

---

### 2.3 Prompting methods (extraction strategies)

**Four active** prompting strategies are exposed in the UI; one legacy strategy is retained for backward compatibility but hidden. Strategy order and IDs are in `src/prompting/strategy_order.py`.

| Strategy (config id) | Display name (UI) | Description |
|----------------------|-------------------|-------------|
| `baseline` | Zero-Shot | No examples; model extracts classes, relations, and hierarchy from text only. |
| `one_shot` | One-Shot (MMR-1) | **One** best concept example per chunk from `pool_strict_concepts.json` (MMR k=1). Optional hierarchy phase runs **only when** the chunk contains hierarchy cues and has ≥2 class labels from the **allowed clinical vocabulary**. |
| `phased_3step` | **Few-Shot** | **Phase 1:** 3 concept examples → extract classes. **Phase 2:** 3 relation examples → extract relations only (hierarchy deferred to Phase 3). **Phase 3:** 3 hierarchy examples → extract hierarchy only where lexical cues exist. Three LLM calls, then merge. Phase 2 and Phase 3 outputs are vocabulary-filtered before merge. |

**Legacy strategies (hidden, backward-compatible):**

| Strategy (config id) | Legacy display name | Status |
|----------------------|---------------------|--------|
| `simple_fewshot` | Few-Shot I (legacy) | Dominated by `phased_2step`; hidden from UI. |

**Task-specific example pools** (see `docs/task_pool_wiring.md` for the full wiring map):

- **`pool_strict_concepts.json`** — 6 class-first examples from BrainIT core dataset; all labels gold-schema aligned. All `Patient` evidences use direct source-text quotes.
- **`pool_strict_relations.json`** — 6 strong, source-faithful relation examples with gold class labels for domain/range. Includes `brainit_monitoring_indicates_condition` — a dedicated example teaching the LLM to extract the `monitoring indicates condition` relation (Monitoring Data → Condition) and the `targets condition` relation (Therapy → Condition).
- **`pool_strict_hierarchy.json`** — 5 gold-aligned hierarchy examples. Includes two new examples using **verbatim BrainIT paper sentences** with the `"include:"` enumeration pattern: one for monitoring parameters (Heart Rate, Respiration Rate, MAP, ICP, SaO2, Temperature ⊑ Monitoring Data) and one for ICU management types (Ventilation, Sedation, Fluids, Nutrition, Vasopressors, Antibiotics ⊑ Intensive Care Management). All hierarchy edge evidences contain valid lexical triggers that pass `filter_hierarchy_to_lexical_cues`.

**Hierarchy phase class label filtering:** For both `one_shot` and `phased_2step` hierarchy sub-calls, the `known_classes` list is **filtered to the allowed clinical vocabulary** before being passed to the prompt.

**Vocabulary and filtering:** When "Prompt vocab guardrails" is on, the prompt injects class labels from the gold vocabulary and **only paper-wording relation labels** (`EXTRACTION_RELATION_LABELS`). CamelCase gold relation labels are **not** injected. `RELATION_ALIASES_CORE` maps paper-wording output to gold labels during post-extraction filtering.

---

### 2.4 LLM providers

#### Extraction LLM

The following providers are selectable as the **extraction LLM** for all strategies:

| Provider | Model |
|----------|-------|
| **OpenAI** (default) | GPT-4o-mini |
| **Anthropic** | Claude Haiku 4.5 |
| **Google** | Gemini 2.5 Flash |
| **Groq** | Llama 3.1 8B (free tier) |
| **Hugging Face** | Mistral 7B via Router API (free) |
| **DeepSeek** | deepseek-chat |

#### Reasoning LLM (Advanced)

When Schema-Guided Completion or the LLM Reasoning Layer is enabled, a separate **Reasoning LLM** is used for those improvement steps. Configurable independently of the extraction LLM:

| Option | Model |
|--------|-------|
| **OpenAI** (default) | GPT-4o-mini (same as extraction) |
| **DeepSeek Reasoner** | deepseek-reasoner (R1) — stronger reasoning, slower |

This separation allows the extraction and reasoning steps to use different models for ablation. The Reasoning LLM choice is reported in `summary.txt` and the run label.

---

### 2.5 Pipeline modes

The UI presents **four progressive pipeline modes** that group the post-processing feature flags into clearly named experimental conditions. Each mode is a superset of the previous one.

| Mode | Config flags enabled | Use case |
|------|---------------------|----------|
| **Strict** | `prompt_vocab_guardrails`, `eval_restrict_to_gold` | Extraction with gold vocab guardrails; no NER, no post-processing |
| **Guided** | Strict + `medical_ner_anchor`, `candidate_terms` | Adds Medical NER anchor and candidate term injection |
| **Schema-Completed** | Guided + `schema_guided_completion`, `symbolic_reasoner` | Adds LLM schema gap-filling + rule-based hierarchy completion |
| **Fully Reasoned** | Schema-Completed + `llm_reasoning` | Adds LLM Reasoning Layer (PROPOSE/VERIFY hierarchy inference) |

`Schema-Completed` is the **default** mode in the UI, balancing capability and speed.

Pipeline mode is reported in the run label (format: `Strategy - Mode - LLM - EvalSettings - Advanced`) and in `summary.txt`.

---

### 2.6 Advanced / Experimental features

The collapsible **Advanced** section of the UI provides:

- **Reasoning LLM override**: Selects the LLM for improvement steps (see §2.4). Defaults to OpenAI; DeepSeek Reasoner is the alternative. When DeepSeek Reasoner is used, `DSR` appears in the run label.

**Medical NER anchor** (`medical_ner_anchor`) is now a **built-in component of Guided Extraction mode** (Mode 2) and cascades to Modes 3 and 4. It injects biomedical entities extracted by ScispaCy (`en_ner_bc5cdr_md`) into the prompt as "Suggested concepts," anchoring the LLM's extraction on verified clinical entities. It is no longer exposed as a separate Advanced toggle.

---

### 2.7 Post-processing improvements

Post-processing improvements run **after** merge, in fixed order: SGC → Cleanup → LLM Reasoning → Rule-based.

#### 2.7.1 Schema-guided completion

- **What it does:** Compares the merged ontology to the **gold schema**, identifies **missing** classes, relations, and hierarchy edges, and asks the LLM: "Which of these missing items are supported by the **corpus**?" Adds only items that (1) are in the schema, (2) have **corpus evidence**, and (3) have valid domain/range for relations. Uses canonical-aware matching (`_canonical_norm`) for class and relation validation.
- **Counts recorded:** `classes_added`, `relations_added`, `hierarchy_added`.
- **Debugging artifacts:** `prompts/sgc_prompt.txt`, `prompts/sgc_response.txt`.

#### 2.7.2 Built-in cleanup (always on when gold is available)

`apply_builtin_cleanup()` in `src/ontology/reasoner.py`. Runs **before** the LLM Reasoning Layer so the LLM operates on a clean, deduplicated ontology. Order:

1. **Dedupe** — classes, relations, hierarchy (canonical key, gold labels; merges provenance and aliases when merging duplicates).
2. **Out-of-scope class pruning** — compound governance patterns only (e.g. `\bsteering group\b`, `\bcommittee\b`, `\bdatabase\b`). Compound-phrase patterns prevent false positives on legitimate clinical labels.
3. **Abstract data label pruning** — specific patterns (`\braw data\b`, `\bcore data\b`, `\bdataset\b`). Extended allowlist includes `Monitoring Data`, `Demographic Data`.
4. **Broad contextual label pruning** — generic labels (`\bpatients?\b`, `\btherapy targets?\b`, etc.). Extended allowlist includes `Patient`, `Therapy`, `Secondary Insult Treatment`, `Intensive Care Management`, `Secondary Insults`.
5. **Class evidence pruning** — keeps only classes with evidence and label supported by evidence. **Exception:** `_EVIDENCE_PRUNING_EXEMPTIONS` is a protected set of 30+ gold-schema structural anchors (including `Condition`, `Fluids`, `Nutrition`, `Sedation`, `Session`, `Timepoint`, `Observation`, `Parameter`, `Data Quality Assessment`, and all abstract category classes) that are kept unconditionally.
6. **Hierarchy dangling endpoint pruning** — removes hierarchy edges whose subClass or superClass was pruned (not in surviving class set). Prevents orphaned edges after class removal.
7. **Relation domain/range pruning** — removes relations whose domain or range is not in the surviving class set.
8. **Relation evidence pruning** — label must be in `_ALLOWED_RELATION_LABELS_GLOBAL` (paper-wording labels + camelCase gold schema labels, normalised). Gold-schema-licensed relations are exempt from strict domain/range substring evidence check.
9. **Hierarchy fragment pruning** — removes edges where subClass/superClass contain bad tokens (verb forms, clauses) or bad phrases in the label text itself.
10. **Axiom constraints** — removes hierarchy/relations violating physiological/semantic type constraints (`neuro_axioms.py`). Expanded for v2.0 with 17 forbidden hierarchy pairs and 8 allowed relation type restrictions. Violations written to `evaluation/axiom_violations.json`.

#### 2.7.3 LLM Reasoning Layer (optional, Fully Reasoned mode)

- **What it does:** Two LLM calls (PROPOSE → VERIFY) that **infer** missing structure (mainly superclass relations) from the **current clean ontology and gold schema only** — no corpus. Only schema-licensed edges are applied.
- **Runs AFTER cleanup** — operates on a deduplicated, pruned ontology for better accuracy.
- **Matching:** Uses `canonical_key(resolve_to_canonical_label(...))` (alias-aware normalization) to match gold hierarchy endpoints to ontology labels.
- **Counts:** `classes_inferred`, `relations_inferred`, `hierarchy_inferred`.
- **Debugging artifacts:** `prompts/llm_reasoning_propose_prompt.txt`, `prompts/llm_reasoning_propose_response.txt`, `prompts/llm_reasoning_verify_prompt.txt`, `prompts/llm_reasoning_verify_response.txt`, `evaluation/llm_reasoning_patch.json`, `evaluation/llm_reasoning_patch_proposed.json`, `evaluation/llm_reasoning_patch_verified.json`.

#### 2.7.4 Rule-based Reasoning Layer (optional, UI toggle)

Acts as a **fallback after LLM Reasoning** — fills any remaining hierarchy gaps exhaustively:

1. **Schema completion** — adds hierarchy edges from gold when both endpoint classes exist in the ontology. Each added edge carries a **synthetic evidence string** (`"Schema-inferred: X is a subclass of Y."`) which satisfies `require_evidence=True` and contains `"is a"` (a recognised `HIERARCHY_LEXICAL_TRIGGER`).
2. **Orphan pruning** — removes classes not referenced by any relation or hierarchy edge. Gold-aligned classes are always preserved even when isolated.

---

### 2.8 Gold standard and evaluation

- **Gold standard:** The BrainIT ontology/schema is loaded from a configurable path. Modes: **public** (surrogate gold), **restricted** (use provided gold for alignment), **isolated** (generate gold from separate corpus).
- **Alignment (classes):** Generated classes aligned to gold using exact and semantic (TF-IDF, threshold 0.55) matching with synonym expansion.
- **Metrics (classes):**
  - **Coverage, Precision, Recall:** fraction of gold classes matched.
  - **Extraction-only:** metrics before any improvement step — used to measure the raw LLM contribution vs. post-processing contribution.
  - **Clinical-only:** governance-vocab classes excluded.
  - **By stratum:** per-chunk-type (core, governance, provenance).
  - **Errors:** hallucinations, schema violations, omissions.
  - **Structural:** `relation_domain_range_rate`, `hierarchy_edges`, `hierarchy_coverage`.

- **Relation metrics:** `compute_relation_metrics()` computes label-level precision/recall for relations against gold via `RELATION_ALIASES_CORE`. Reports `per_gold_relation` (which gold relations were found), `matched_generated` / `unmatched_generated`. Present under `metrics["relations"]` and `metrics["extraction_only"]["relations"]`.

- **Hierarchy metrics:** `compute_hierarchy_metrics()` computes precision/recall for hierarchy edges (subClass, superClass pairs) by normalized key matching. Present under `metrics["hierarchy"]`.

- **Per-stage ablation (`by_stage`):** Compact metrics snapshots (n_classes, n_relations, n_hierarchy, coverage, precision, recall, relation_recall, hierarchy_recall, hierarchy_precision) are captured after each major pipeline stage. Stages: `extraction` → `after_sgc` → `after_cleanup` → `after_llm_reasoning` → `after_rule_based` → `after_gold_filter`. Written to `metrics["by_stage"]` in `metrics.json` and rendered as a formatted table in `summary.txt`.

- **Outputs:** `evaluation/metrics.json` (includes `relations`, `extraction_only`, `clinical_only`, `by_stratum`, `by_stage`), `evaluation/table.csv`, `evaluation/hallucinated_classes.json`, `evaluation/improvement_counts.json`.

---

### 2.9 Run artifacts and summary

#### Run directory structure

Each run has a unique ID (timestamp + short hash). Under the run directory:

- **Corpus and config:** `corpus_manifest.json`, `metadata.json` (includes `input_papers` list with filename/path, `config`, `environment`, `code_version`).
- **Prompts:** Optional prompt saves under `prompts/`.
- **Generated:** `ontology.json`, `ontology_raw.json`, `summary.txt`, optionally `ontology_restricted.json`.
- **Evaluation:** `metrics.json`, `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`, optionally `axiom_violations.json` and LLM reasoning patch files.
- **Other:** `run.log`, `warnings.txt`.

#### summary.txt (enhanced)

The run summary is a single, human-readable file with the following sections:

1. **Metadata** — run ID, timestamp (UTC), prompting method, **pipeline mode**, extraction LLM, reasoning LLM (if different), improvements applied, **evaluation settings** (scope filter, gold-vocab, Medical NER).
2. **Input paper(s)** — filenames of all ingested source documents (from `docs` passed at write time).
3. **Improvement counts** — per-feature counts (classes/relations/hierarchy added, removed, inferred) plus a NOTE if LLM Reasoning Layer inferred schema-only hierarchy edges.
4. **Metrics summary (final ontology)** — coverage, precision, recall, errors, structural, clinical-only sub-metrics, relation precision/recall.
5. **Metrics at extraction only** — pre-improvement baseline (coverage, precision, recall, structural, relation metrics) for direct comparison with the final metrics above.
6. **Per-stage ablation table** — formatted table showing n_classes, n_relations, n_hierarchy, coverage, precision, recall at each stage (extraction → +SGC → +LLM Reasoning → +Cleanup → +Rule-based).
7. **Concept counts** (final ontology).
8. **Classes, Relations, Hierarchy** listings.

#### metadata.json (enhanced)

Now includes an `input_papers` field:

```json
"input_papers": [
  {"name": "corpus.txt", "stem": "corpus", "path": "data\\corpus_ui\\corpus.txt"}
]
```

This provides full traceability from run to source file, independently of `corpus_snapshots`.

---

### 2.10 Web UI

The Flask-based UI provides:

- **Run experiment form** — Three prompting toggles (Zero-Shot / One-Shot / Few-Shot), four pipeline mode cards (Strict / Guided / Schema-Completed / Fully Reasoned), LLM provider selector, evaluation settings (gold-vocab checkbox), and a collapsible Advanced section (Medical NER, Reasoning LLM). Few-Shot and Schema-Completed are the defaults.
- **Progress page** — Live progress bar with cancel. Shows extraction progress per chunk.
- **Results page** — Metrics, per-stage ablation table, ontology listing, improvement counts.
- **Run comparison** — Configuration panel mirrors the Run experiment form (same pipeline mode cards, strategy toggles, LLM selector). Runs are added to a comparison list with a **5-part standardised label**: `Strategy - PipelineMode - LLM - EvalSettings - Advanced` (e.g. `Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - None`). The comparison dashboard shows a table with these five columns. Three pre-configured comparison groups: Cross-LLM, Pipeline Modes, and Reasoning LLM.
- **Run list** — Browse all past runs; link to results/analysis.
- **Pipeline view** — Visual diagram of the current pipeline configuration.

Run label generation is consistent across Python (`web/app.py: _format_run_label`) and JavaScript (`comparison_dashboard.html: formatRunName`, `comparison_progress.html: formatRunName`).

---

## 3. Summary table (implemented functionality)

| Area | Implemented |
|------|-------------|
| **Input** | Paste text, upload .txt/.pdf; Use default paper; corpus manifest per run. Load pipeline: strip control chars → normalize → optional scope filter; optional `raw_text` and PDF `pages`. |
| **Normalization** | Hyphenation fix, symbol normalization, repeated header/footer removal, control-char strip (`normalize.py`). |
| **Scope filter** | Optional document-level (paragraph + line, compound-phrase blacklist only) inside load_corpus; chunk-level dual-score router (admin_score, clinical_score; governance section blacklist; clinical section always-keep; reorder clinical first). "monitoring" is a strong clinical term (1.0 weight). "fluids", "fluid input and output", "nutrition", "condition", "sedation levels" promoted to strong (Phase 10). |
| **Scope filter validation** | `run_one()` raises `ValueError` if `scope_filter` absent from config when gold is loaded. |
| **Chunking** | Semantic chunking; control chars stripped per chunk; candidates with section context. |
| **Prompting methods** | Four active strategies: Zero-Shot, One-Shot (MMR-1), Few-Shot (`phased_2step`), Few-Shot III (`phased_3step`). One legacy strategy hidden but backward-compatible (`simple_fewshot`). |
| **Pipeline modes (UI)** | Four progressive modes: Strict / Guided / Schema-Completed (default) / Fully Reasoned. Map to combinations of feature flags via hidden inputs and JavaScript. |
| **Hierarchy phase filtering** | `known_classes` filtered to allowed clinical vocabulary before hierarchy sub-calls. |
| **LLM providers (extraction)** | OpenAI (default), Anthropic, Google, Groq, HuggingFace, DeepSeek. |
| **Reasoning LLM (improvements)** | Configurable separately: OpenAI (default) or DeepSeek Reasoner (R1). Advanced section in UI. |
| **Vocab guardrails** | Injects gold class list + paper-wording relation labels only; `RELATION_ALIASES_CORE` maps paper-wording to gold during filtering. |
| **Medical NER** | ScispaCy NER anchor. Built into Guided mode (Mode 2) and cascades to Modes 3 and 4. Injects pre-identified clinical entities as "Suggested concepts" in the prompt. |
| **Improvements** | Built-in cleanup (always when gold: dedupe, scope/abstract/broad pruning [compound patterns + extended allowlists], evidence/hierarchy pruning [with `_EVIDENCE_PRUNING_EXEMPTIONS` whitelist for 30+ v2.0 classes], dangling hierarchy endpoint pruning, axioms [expanded for v2.0]); Schema-guided completion (canonical-aware, hierarchy-aware); LLM Reasoning Layer (PROPOSE/VERIFY, runs after cleanup); Rule-based Reasoning Layer (schema-derived edges with synthetic evidence). Pipeline order: SGC → Cleanup → LLM Reasoning → Rule-based. |
| **Relation whitelist** | `_ALLOWED_RELATION_LABELS_GLOBAL`: paper-wording + camelCase gold labels (normalised). |
| **Evaluation — class** | Coverage, precision, recall, error taxonomy, structural; `extraction_only`; `clinical_only`; `by_stratum`; `by_stage`. |
| **Evaluation — relations** | `compute_relation_metrics()`: label-level precision/recall via `RELATION_ALIASES_CORE`; `per_gold_relation`; in `metrics["relations"]` and `metrics["extraction_only"]["relations"]`. |
| **Evaluation — hierarchy** | `compute_hierarchy_metrics()`: edge-level precision/recall by normalized key matching; in `metrics["hierarchy"]`. |
| **Per-stage ablation** | `_capture_stage_metrics()` snapshots after each pipeline stage (including hierarchy metrics); `metrics["by_stage"]` in `metrics.json`; formatted ablation table in `summary.txt`. Stages: extraction → after_sgc → after_cleanup → after_llm_reasoning → after_rule_based → after_gold_filter. |
| **Canonical aliases** | `CANONICAL_ALIAS_MAP` in `canonical.py`: TBI, CVP, ICP, CPP, GCS, GOSE, MAP variants + Phase 10 extensions (Fluids ← fluid input and output/management/balance; Sedation ← sedation levels; Nutrition ← nutritional support; Condition ← clinical condition/medical condition). Evaluation synonyms in `synonyms.py` extended with Fluids/Nutrition/Sedation/Condition variants including specific condition types (systemic hypotension, intracranial hypertension → Condition). |
| **Hierarchy triggers** | `HIERARCHY_LEXICAL_TRIGGERS`: `"such as"`, `"is a"`, `"type of"`, `"kind of"`, **`"include:"`**, **`"includes:"`**. `extract_hierarchy_from_text()` handles all triggers; `"include:"` handler preserves compound nouns and strips `"use of"` prefixes. |
| **Merge / build** | `merge_parsed()`: classes by canonical key, relations by (label, domain, range); `build_ontology`: evidence required; canonical alias map; singular/plural merge; stratum on entities; dedupe and duplicate merge. |
| **Run label format** | `Strategy - PipelineMode - LLM - EvalSettings - Advanced`. Consistent: Python `_format_run_label`, JS `formatRunName`. |
| **Artifacts** | `ontology.json`, `ontology_raw.json`, `summary.txt` (enhanced: timestamp, papers, mode, eval settings, extraction baseline, by_stage table), `metrics.json` (includes `by_stage`), `improvement_counts.json`, `axiom_violations.json`, patch files, `metadata.json` (includes `input_papers`), `run.log`. |
| **Config files** | `demo.json`: all benchmark-required keys. `benchmark_template.json`: reusable template. |
| **UI** | Flask app: Run experiment (3-strategy + 4-mode + advanced); progress (with cancel); results; Run comparison (mirrors experiment form, 5-part run label, 5-column table); pipeline view; run list. |
| **CLI** | Config-driven runs; `scope_filter` must be explicit when gold is loaded. |

This implementation delivers a **modular, hybrid ontology engineering framework** with three prompting strategies, multiple LLM providers, a four-mode progressive pipeline abstraction (Strict → Guided → Schema-Completed → Fully Reasoned), complete evaluation against the BrainIT gold standard  including **relation recall**, **hierarchy precision/recall**, and **per-stage ablation**, and reproducible, validated configuration for benchmark runs.

