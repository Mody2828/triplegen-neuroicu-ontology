# Implementation Summary: LLM-Driven Ontology Engineering for Neuro-ICU

This document presents a concise summary of the **project aim and objectives** together with a detailed overview of the **implemented design and functionality** of the framework.

---

## 1. Project aim and objectives

### Aim

The project aims to **design, implement, and evaluate** an LLM-driven ontology engineering framework for the **Neuro-ICU / neuro-intensive care domain**, using **BrainIT literature** as the primary input. The central question is how well language models can reconstruct a high-quality domain ontology in a **low-resource clinical setting**, and which factors—prompting, constraints, post-processing, and choice of LLM—improve or degrade that outcome.

### Objectives

* **End-to-end pipeline:** Build a reproducible pipeline from raw text input (paste or upload) to a generated ontology and evaluation report.
* **Multiple LLM providers:** Support and compare extraction across several APIs (OpenAI (GPT‑4o‑mini), Anthropic (Claude Haiku 4.5), Google (Gemini 2.5 Flash), Groq (Llama 3.1 8B), Hugging Face (Mistral 7B), DeepSeek (deepseek-chat)).
* **Several prompting methods:** Compare zero-shot (baseline), one-shot, few-shot (phased 2-step), and few-shot III (phased 3-step) extraction to assess the impact of examples and task decomposition.
* **Progressive pipeline modes:** Provide four clearly named modes (Strict → Guided → Schema-Completed → Fully Reasoned) that group optional post-processing features into meaningful experimental conditions for ablation.
* **Enhancement strategies:** Implement and evaluate optional post-processing improvements—schema-guided completion, an LLM reasoning layer, and a rule-based reasoning layer—as orthogonal toggles for ablation.
* **Evaluation against gold:** Evaluate generated ontologies against the **BrainIT gold standard** (ontology/schema) using coverage, precision, recall, **relation recall**, structural metrics, error taxonomy, and a **per-stage ablation table**.
* **Usability and reproducibility:** Provide a web-based UI for running experiments, saving artifacts per run, and comparing runs side-by-side.

The framework is positioned as a **modular hybrid ontology engineering system** (prompt strategies plus optional LLM and symbolic layers), rather than only a set of prompt experiments.

---

## 2. Implemented design and functionality

### 2.1 High-level architecture

The system is organised into **four conceptual layers**:

1. **Prompt strategy (extraction)**
   Defines how the LLM extracts ontology elements from text, including the choice of prompting method and, where applicable, the retrieval of examples.

2. **Merge and optional vocabulary filter**
   Chunk-level extractions are merged into a single ontology. When evaluation controls are enabled, the result can optionally be restricted to a gold vocabulary.

3. **Post-processing**
   Built-in cleanup is applied when gold is available, followed by optional improvements in fixed order: Schema-guided completion, LLM Reasoning Layer, and Rule-based Reasoning Layer.

4. **Evaluation**
   The ontology is aligned to the gold standard, metrics are computed, per-stage snapshots are recorded, and the run summary is produced.

### Pipeline Order

The pipeline follows these steps:

1. **Corpus Input**
   - Paste or upload text (e.g., `.txt` or `.pdf` files).
   - Load and preprocess the text:
     - Strip control characters.
     - Normalize text.
     - Apply scope filter (if enabled).

2. **Chunking**
   - Split text into chunks.
   - Retain section context for each chunk.

3. **Extraction**
   - Extract ontology elements per chunk using the selected strategy.
   - Apply evidence requirements.
   - Filter Phase 2 output.

4. **Merge and Vocabulary Filtering**
   - Merge extracted chunks into a single ontology.
   - Apply canonical alias mapping, singular/plural normalization, and deduplication.
   - Optionally filter vocabulary to match the gold standard.

   **Stage Snapshot:** Extraction

5. **Schema-Guided Completion**
   - Apply schema-guided completion to enhance the ontology.

   **Stage Snapshot:** After Schema-Guided Completion (SGC)

6. **Built-in Cleanup**
   - Perform cleanup operations on the ontology.

   **Stage Snapshot:** After Cleanup

7. **LLM Reasoning Layer**
   - Apply reasoning to the cleaned ontology.

   **Stage Snapshot:** After LLM Reasoning

8. **Rule-Based Reasoning Layer** (Optional)
   - Apply schema completion and orphan pruning.

   **Stage Snapshot:** After Rule-Based Reasoning

9. **Gold Vocabulary Filtering** (Optional)
   - Filter ontology to match the gold vocabulary if evaluation restriction is enabled.

   **Stage Snapshot:** After Gold Filter

10. **Validation and Evaluation**
    - Validate the ontology.
    - Evaluate metrics (class, relation, hierarchy).
    - Generate artifacts and summary.

---

**Refer to the pipeline graph below for a visual representation of the methodology:**

![Pipeline](screenshots/1.jpg)

---

### 2.2 Corpus input, normalization, scope filter, and chunking

#### Input

Users can either **paste text** directly or **upload one or more files** (`.txt` or `.pdf`). Multiple uploaded files are saved to `data/corpus_ui` and loaded as a multi-document corpus. The **Use default paper** option allows a run to use a built-in default corpus when no text is pasted or uploaded.

#### Load pipeline (`src/corpus/ingest.py`)

For each document, the load sequence is:

* **raw text**
* **strip control chars** (`clean_chars`)
* **normalize**
* **[scope filter if enabled]**

The resulting cleaned text is stored as `doc["text"]`.

Optional fields include:

* `doc["raw_text"]` when `keep_raw_text` is enabled in config
* `doc["pages"]` for PDFs, providing page-level segments for provenance and header/footer detection

#### Normalization (`src/corpus/normalize.py`)

Normalization performs the following operations:

* fixes broken hyphenation across line breaks
  *(e.g. `intra-\ncranial` → `intracranial`)*
* normalises symbols such as Unicode fractions, fancy dashes, curly quotes, and OCR repeats
* removes repeated header/footer lines across pages or blocks
* strips form-feed and other control characters

#### Scope filter (optional)

When enabled, the scope filter runs **inside `load_corpus` after normalization**.

It operates at two levels:

##### 1. Document-level filtering

* paragraph-level filtering drops whole paragraphs if they are admin-heavy
* line-level filtering removes administrative headings, phrases, and blacklist terms

##### 2. Chunk-level filtering

After chunking, `filter_chunks_to_clinical` applies a **dual-score router**:

* `chunk_scores()` returns `(admin_score, clinical_score)`

Chunks are handled as follows:

* **Drop** if:

  * the section matches governance sections
    *(e.g. Part A, group formation, database access, ethics approval, etc.)*
  * `governance_dominance ≥ 2` and `clinical_score ≤ 1`
  * `admin_score ≥ 2` and `clinical_score ≤ 1`

* **Keep** if:

  * `clinical_score ≥ 3`
  * the section matches clinical dataset sections

* **Reorder**:

  * clinical dataset sections are ordered first

##### Blacklist terms (`_SCOPE_BLACKLIST_TERMS`)

Only compound phrases are blacklisted:

* `membership`
* `centres`
* `centers`
* `quality control`
* `steering group`
* `ethics committee`
* `consortium`
* `data validation staff`
* `validation workflow`
* `publication criteria`

Single generic words are intentionally excluded from the blacklist to avoid removing legitimate clinical content.

##### Clinical scoring

Clinical scoring assigns weights as follows:

* **Strong terms (1.0)** include:
  `heart rate`, `ICP`, `GCS`, `sedation`, `sedation levels`, `ventilation`, `monitoring`, `CPP`, `MAP`, `fluids`, `fluid input and output`, `nutrition`, `condition`, etc.
* **Broad terms (0.5)** include:
  `core dataset`, `outcome`

The following were explicitly promoted to strong terms:

* `monitoring`
* `fluids`
* `nutrition`
* `condition`
* `sedation levels`

This was done to avoid filtering out chunks that contain only brief but clinically relevant therapy-type terms before they reach the LLM.

#### Scope filter config validation

When a gold standard is loaded in benchmark mode, `scope_filter` **must be explicitly set** in the config. If the key is missing, `run_experiments.run_one()` raises a `ValueError` immediately. This ensures that UI runs (`scope_filter=True` by default) and CLI runs remain comparable.

#### Chunking

Documents are split into **semantic chunks** so that each chunk:

* remains coherent
* fits within prompt limits

Chunk text is passed through **strip control chars** again before candidate extraction and storage.

---

### 2.3 Prompting methods (extraction strategies)

**Four active** prompting strategies are exposed in the UI. One legacy strategy is retained for backward compatibility but hidden. Strategy order and IDs are defined in `src/prompting/strategy_order.py`.

| Strategy (config id) | Display name (UI) | Description                                                                                                                                                                                                                                                                                                                       |
| -------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baseline`           | Zero-Shot         | No examples; model extracts classes, relations, and hierarchy from text only.                                                                                                                                                                                                                                                     |
| `one_shot`           | One-Shot (MMR-1)  | **One** best concept example per chunk from `pool_strict_concepts.json` (MMR k=1). Optional hierarchy phase runs **only when** the chunk contains hierarchy cues and has ≥2 class labels from the **allowed clinical vocabulary**.                                                                                                |
| `phased_3step`       | **Few-Shot**      | **Phase 1:** 3 concept examples → extract classes. **Phase 2:** 3 relation examples → extract relations only (hierarchy deferred to Phase 3). **Phase 3:** 3 hierarchy examples → extract hierarchy only where lexical cues exist. Three LLM calls, then merge. Phase 2 and Phase 3 outputs are vocabulary-filtered before merge. |

#### Task-specific example pools

* **`pool_strict_concepts.json`**
  6 class-first examples from the BrainIT core dataset; all labels are gold-schema aligned. All `Patient` evidences use direct source-text quotes.

* **`pool_strict_relations.json`**
  6 strong, source-faithful relation examples with gold class labels for domain/range. Includes `brainit_monitoring_indicates_condition`, a dedicated example teaching the model to extract:

  * the `monitoring indicates condition` relation (Monitoring Data → Condition)
  * the `targets condition` relation (Therapy → Condition)

* **`pool_strict_hierarchy.json`**
  5 gold-aligned hierarchy examples. Includes two examples using **verbatim BrainIT paper sentences** with the `"include:"` enumeration pattern:

  * one for monitoring parameters
    *(Heart Rate, Respiration Rate, MAP, ICP, SaO2, Temperature ⊑ Monitoring Data)*
  * one for ICU management types
    *(Ventilation, Sedation, Fluids, Nutrition, Vasopressors, Antibiotics ⊑ Intensive Care Management)*

All hierarchy-edge evidences contain valid lexical triggers that pass `filter_hierarchy_to_lexical_cues`.

#### Hierarchy phase class label filtering

For both `one_shot` and `phased_2step` hierarchy sub-calls, the `known_classes` list is filtered to the allowed clinical vocabulary before being passed to the prompt.

#### Vocabulary and filtering

When **Prompt vocab guardrails** is enabled, the prompt injects:

* class labels from the gold vocabulary
* only paper-wording relation labels (`EXTRACTION_RELATION_LABELS`)

CamelCase gold relation labels are **not** injected.

`RELATION_ALIASES_CORE` maps paper-wording output to gold labels during post-extraction filtering.

---

### 2.4 LLM providers

#### Extraction LLM

The following providers are available for the **extraction LLM**:

| Provider             | Model                            |
| -------------------- | -------------------------------- |
| **OpenAI** (default) | GPT-4o-mini                      |
| **Anthropic**        | Claude Haiku 4.5                 |
| **Google**           | Gemini 2.5 Flash                 |
| **Groq**             | Llama 3.1 8B (free tier)         |
| **Hugging Face**     | Mistral 7B via Router API (free) |
| **DeepSeek**         | deepseek-chat                    |

#### Reasoning LLM (Advanced)

When Schema-Guided Completion or the LLM Reasoning Layer is enabled, a separate **Reasoning LLM** is used for those improvement steps.

Available options:

| Option                | Model                                               |
| --------------------- | --------------------------------------------------- |
| **OpenAI** (default)  | GPT-4o-mini (same as extraction)                    |
| **DeepSeek Reasoner** | deepseek-reasoner (R1) — stronger reasoning, slower |

This separation allows the extraction and reasoning steps to be varied independently for ablation. The selected Reasoning LLM is reported in `summary.txt` and the run label.

---

### 2.5 Pipeline modes

The UI presents **four progressive pipeline modes** that group post-processing feature flags into clearly named experimental conditions. Each mode is a superset of the previous one.

| Mode                 | Config flags enabled                                     | Use case                                                          |
| -------------------- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| **Strict**           | `prompt_vocab_guardrails`, `eval_restrict_to_gold`       | Extraction with gold vocab guardrails; no NER, no post-processing |
| **Guided**           | Strict + `medical_ner_anchor`, `candidate_terms`         | Adds Medical NER anchor and candidate term injection              |
| **Schema-Completed** | Guided + `schema_guided_completion`, `symbolic_reasoner` | Adds LLM schema gap-filling + rule-based hierarchy completion     |
| **Fully Reasoned**   | Schema-Completed + `llm_reasoning`                       | Adds LLM Reasoning Layer (PROPOSE/VERIFY hierarchy inference)     |

`Schema-Completed` is the **default** mode in the UI, chosen as a balance between capability and speed.

Pipeline mode is reported in the run label using the format:

`Strategy - Mode - LLM - EvalSettings - Advanced`

and is also shown in `summary.txt`.

---

### 2.6 Advanced / Experimental features

The collapsible **Advanced** section of the UI currently provides:

* **Reasoning LLM override**
  Selects the LLM used for improvement steps (see §2.4). OpenAI is the default; DeepSeek Reasoner is the alternative. When DeepSeek Reasoner is used, `DSR` appears in the run label.

---

### 2.7 Post-processing improvements

Post-processing improvements run **after merge**, in the fixed order:

**Schema-guided completion → Cleanup → LLM Reasoning → Rule-based reasoning**

#### 2.7.1 Schema-guided completion

* **Purpose:** Compares the merged ontology to the **gold schema**, identifies **missing** classes, relations, and hierarchy edges, and asks the LLM which of these missing items are supported by the **corpus**.
* **Addition criteria:** Only items that:

  1. are present in the schema,
  2. have corpus evidence,
  3. and have valid domain/range for relations
     are added.
* **Matching:** Uses canonical-aware matching (`_canonical_norm`) for validating classes and relations.
* **Recorded counts:** `classes_added`, `relations_added`, `hierarchy_added`
* **Debugging artifacts:**
  `prompts/sgc_prompt.txt`, `prompts/sgc_response.txt`

#### 2.7.2 Built-in cleanup (always on when gold is available)

`apply_builtin_cleanup()` is defined in `src/ontology/reasoner.py`. It runs **before** the LLM Reasoning Layer so that the LLM operates on a clean, deduplicated ontology.

The cleanup order is:

1. **Dedupe**
   Classes, relations, and hierarchy are deduplicated using canonical keys and gold labels. Provenance and aliases are merged when duplicates are collapsed.

2. **Out-of-scope class pruning**
   Removes labels matching compound governance patterns only
   *(e.g. `steering group`, `committee`, `database`)*
   Compound patterns are used to avoid false positives on legitimate clinical labels.

3. **Abstract data label pruning**
   Removes labels matching specific patterns such as `raw data`, `core data`, `dataset`.
   The allowlist includes `Monitoring Data` and `Demographic Data`.

4. **Broad contextual label pruning**
   Removes generic labels such as `patients`, `therapy targets`, etc.
   The allowlist includes `Patient`, `Therapy`, `Secondary Insult Treatment`, `Intensive Care Management`, and `Secondary Insults`.

5. **Class evidence pruning**
   Retains only classes whose label is supported by evidence.
   **Exception:** `_EVIDENCE_PRUNING_EXEMPTIONS` contains a protected set of 30+ gold-schema structural anchors, including `Condition`, `Fluids`, `Nutrition`, `Sedation`, `Session`, `Timepoint`, `Observation`, `Parameter`, `Data Quality Assessment`, and all abstract category classes. These are always preserved.

6. **Hierarchy dangling endpoint pruning**
   Removes hierarchy edges whose subclass or superclass no longer exists after class pruning.

7. **Relation domain/range pruning**
   Removes relations whose domain or range is not present in the surviving class set.

8. **Relation evidence pruning**
   Relation labels must be in `_ALLOWED_RELATION_LABELS_GLOBAL` (paper-wording labels plus camelCase gold schema labels, normalised). Gold-schema-licensed relations are exempt from strict domain/range substring evidence checks.

9. **Hierarchy fragment pruning**
   Removes hierarchy edges whose subclass or superclass contains bad tokens such as verb forms or clause fragments.

10. **Axiom constraints**
    Removes hierarchy and relation structures that violate physiological or semantic type constraints defined in `neuro_axioms.py`.
    This was expanded in v2.0 to include:

    * 17 forbidden hierarchy pairs
    * 8 allowed relation type restrictions

    Violations are written to `evaluation/axiom_violations.json`.

#### 2.7.3 LLM Reasoning Layer (optional, Fully Reasoned mode)

* **Purpose:** Uses two LLM calls (PROPOSE → VERIFY) to infer missing structure, mainly superclass relations, from the **current clean ontology and gold schema only**.
* **Important:** This stage does **not** use corpus text.
* **Application rule:** Only schema-licensed edges are applied.
* **Matching:** Uses `canonical_key(resolve_to_canonical_label(...))` for alias-aware matching between ontology labels and gold hierarchy endpoints.
* **Recorded counts:** `classes_inferred`, `relations_inferred`, `hierarchy_inferred`
* **Debugging artifacts:**
  `prompts/llm_reasoning_propose_prompt.txt`
  `prompts/llm_reasoning_propose_response.txt`
  `prompts/llm_reasoning_verify_prompt.txt`
  `prompts/llm_reasoning_verify_response.txt`
  `evaluation/llm_reasoning_patch.json`
  `evaluation/llm_reasoning_patch_proposed.json`
  `evaluation/llm_reasoning_patch_verified.json`

#### 2.7.4 Rule-based Reasoning Layer (optional, UI toggle)

This acts as a **fallback after LLM Reasoning** and fills remaining hierarchy gaps exhaustively.

It performs two operations:

1. **Schema completion**
   Adds hierarchy edges from the gold ontology when both endpoint classes exist in the ontology.
   Each added edge carries a synthetic evidence string of the form:

   `"Schema-inferred: X is a subclass of Y."`

   This satisfies `require_evidence=True` and includes `"is a"`, which is a recognised `HIERARCHY_LEXICAL_TRIGGER`.

2. **Orphan pruning**
   Removes classes not referenced by any relation or hierarchy edge. Gold-aligned classes are always preserved even if isolated.

---

### 2.8 Gold standard and evaluation

#### Gold standard

The BrainIT ontology/schema is loaded from a configurable path.

Supported modes:

* **public** — surrogate gold
* **restricted** — provided gold used for alignment
* **isolated** — gold generated from a separate corpus

#### Alignment (classes)

Generated classes are aligned to gold using:

* exact matching
* semantic TF-IDF matching with threshold 0.55
* synonym expansion

#### Metrics (classes)

The framework reports:

* **Coverage, Precision, Recall**
* **Extraction-only** metrics, measured before improvement stages
* **Clinical-only** metrics, excluding governance-vocabulary classes
* **By stratum** metrics, grouped by chunk type (`core`, `governance`, `provenance`)
* **Errors** such as hallucinations, schema violations, and omissions
* **Structural metrics** including:

  * `relation_domain_range_rate`
  * `hierarchy_edges`
  * `hierarchy_coverage`

#### Relation metrics

`compute_relation_metrics()` computes label-level precision and recall for relations against gold via `RELATION_ALIASES_CORE`.

It reports:

* `per_gold_relation`
* `matched_generated`
* `unmatched_generated`

These are stored in:

* `metrics["relations"]`
* `metrics["extraction_only"]["relations"]`

#### Hierarchy metrics

`compute_hierarchy_metrics()` computes precision and recall for hierarchy edges using normalised `(subClass, superClass)` key matching.

These are stored in:

* `metrics["hierarchy"]`

#### Per-stage ablation (`by_stage`)

Compact metric snapshots are captured after each major pipeline stage. Each snapshot includes:

* `n_classes`
* `n_relations`
* `n_hierarchy`
* `coverage`
* `precision`
* `recall`
* `relation_recall`
* `hierarchy_recall`
* `hierarchy_precision`

Stages recorded are:

* `extraction`
* `after_sgc`
* `after_cleanup`
* `after_llm_reasoning`
* `after_rule_based`
* `after_gold_filter`

These are written to `metrics["by_stage"]` in `metrics.json` and rendered as a formatted ablation table in `summary.txt`.

#### Outputs

Evaluation outputs include:

* `evaluation/metrics.json`
* `evaluation/table.csv`
* `evaluation/hallucinated_classes.json`
* `evaluation/improvement_counts.json`

---

### 2.9 Run artifacts and summary

#### Run directory structure

Each run receives a unique ID consisting of a timestamp and short hash.

The run directory includes:

* **Corpus and config**

  * `corpus_manifest.json`
  * `metadata.json`
    Includes `input_papers`, `config`, `environment`, and `code_version`

* **Prompts**

  * optional prompt saves under `prompts/`

* **Generated**

  * `ontology.json`
  * `ontology_raw.json`
  * `summary.txt`
  * optionally `ontology_restricted.json`

* **Evaluation**

  * `metrics.json`
  * `table.csv`
  * `hallucinated_classes.json`
  * `improvement_counts.json`
  * optionally `axiom_violations.json`
  * LLM reasoning patch files when relevant

* **Other**

  * `run.log`
  * `warnings.txt`

#### `summary.txt` (enhanced)

The run summary is a single human-readable file containing:

1. **Metadata**
   Run ID, timestamp (UTC), prompting method, pipeline mode, extraction LLM, reasoning LLM (if different), improvements applied, evaluation settings (scope filter, gold-vocab, Medical NER)

2. **Input paper(s)**
   Filenames of all ingested source documents

3. **Improvement counts**
   Per-feature counts for classes, relations, and hierarchy items added, removed, or inferred, plus a note if the LLM Reasoning Layer inferred schema-only hierarchy edges

4. **Metrics summary (final ontology)**
   Coverage, precision, recall, errors, structural metrics, clinical-only sub-metrics, relation precision, relation recall

5. **Metrics at extraction only**
   Pre-improvement baseline for direct comparison with the final metrics

6. **Per-stage ablation table**
   Formatted table showing:

   * `n_classes`
   * `n_relations`
   * `n_hierarchy`
   * `coverage`
   * `precision`
   * `recall`
     across all pipeline stages

7. **Concept counts**
   Final ontology counts

8. **Classes, Relations, Hierarchy**
   Final listings

#### `metadata.json` (enhanced)

`metadata.json` now includes an `input_papers` field such as:

```json
"input_papers": [
  {"name": "corpus.txt", "stem": "corpus", "path": "data\\corpus_ui\\corpus.txt"}
]
```

This provides full traceability from a run back to its source file independently of corpus snapshots.

---

### 2.10 Web UI

The Flask-based UI provides the following features:

* **Run experiment form**
  Three prompting toggles (Zero-Shot / One-Shot / Few-Shot), four pipeline mode cards (Strict / Guided / Schema-Completed / Fully Reasoned), LLM provider selector, evaluation settings (gold-vocab checkbox), and a collapsible Advanced section (Medical NER, Reasoning LLM). Few-Shot and Schema-Completed are the defaults.

* **Progress page**
  Live progress bar with cancel support. Extraction progress is shown per chunk.

* **Results page**
  Displays metrics, per-stage ablation table, ontology listing, and improvement counts.

* **Run comparison**
  The comparison configuration mirrors the Run experiment form. Runs are added to a comparison list with a **5-part standardised label**:

  `Strategy - PipelineMode - LLM - EvalSettings - Advanced`

  Example:
  `Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - None`

  The comparison dashboard displays these in a 5-column table.
  Three pre-configured comparison groups are available:

  * Cross-LLM
  * Pipeline Modes
  * Reasoning LLM

* **Run list**
  Browse all past runs and access results/analysis.

* **Pipeline view**
  Visual diagram of the current pipeline configuration.

Run label generation is consistent across:

* Python (`web/app.py: _format_run_label`)
* JavaScript (`comparison_dashboard.html: formatRunName`, `comparison_progress.html: formatRunName`)

---

## 3. Summary table (implemented functionality)

| Area                             | Implemented                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**                        | Paste text, upload .txt/.pdf; Use default paper; corpus manifest per run. Load pipeline: strip control chars → normalize → optional scope filter; optional `raw_text` and PDF `pages`.                                                                                                                                                                                                                                             |
| **Normalization**                | Hyphenation fix, symbol normalization, repeated header/footer removal, control-char strip (`normalize.py`).                                                                                                                                                                                                                                                                                                                        |
| **Scope filter**                 | Optional document-level (paragraph + line, compound-phrase blacklist only) inside load_corpus; chunk-level dual-score router (admin_score, clinical_score; governance section blacklist; clinical section always-keep; reorder clinical first). `"monitoring"` is a strong clinical term (1.0 weight). `"fluids"`, `"fluid input and output"`, `"nutrition"`, `"condition"`, `"sedation levels"` are also strong terms (Phase 10). |
| **Scope filter validation**      | `run_one()` raises `ValueError` if `scope_filter` is absent from config when gold is loaded.                                                                                                                                                                                                                                                                                                                                       |
| **Chunking**                     | Semantic chunking; control chars stripped per chunk; candidates with section context.                                                                                                                                                                                                                                                                                                                                              |
| **Prompting methods**            | Four active strategies: Zero-Shot, One-Shot (MMR-1), Few-Shot (`phased_2step`), Few-Shot III (`phased_3step`). One legacy strategy remains hidden but backward-compatible (`simple_fewshot`).                                                                                                                                                                                                                                      |
| **Pipeline modes (UI)**          | Four progressive modes: Strict / Guided / Schema-Completed (default) / Fully Reasoned. These map to combinations of feature flags via hidden inputs and JavaScript.                                                                                                                                                                                                                                                                |
| **Hierarchy phase filtering**    | `known_classes` are filtered to the allowed clinical vocabulary before hierarchy sub-calls.                                                                                                                                                                                                                                                                                                                                        |
| **LLM providers (extraction)**   | OpenAI (default), Anthropic, Google, Groq, HuggingFace, DeepSeek.                                                                                                                                                                                                                                                                                                                                                                  |
| **Reasoning LLM (improvements)** | Configurable separately: OpenAI (default) or DeepSeek Reasoner (R1). Available through the Advanced UI section.                                                                                                                                                                                                                                                                                                                    |
| **Vocab guardrails**             | Injects gold class list plus paper-wording relation labels only; `RELATION_ALIASES_CORE` maps paper-wording to gold during filtering.                                                                                                                                                                                                                                                                                              |
| **Medical NER**                  | ScispaCy NER anchor. Included in Guided mode (Mode 2) and cascades to Modes 3 and 4. Injects pre-identified clinical entities as “Suggested concepts” in the prompt.                                                                                                                                                                                                                                                               |
| **Improvements**                 | Built-in cleanup (always when gold: dedupe, scope/abstract/broad pruning, evidence/hierarchy pruning, dangling hierarchy endpoint pruning, axioms); Schema-guided completion (canonical-aware, hierarchy-aware); LLM Reasoning Layer (PROPOSE/VERIFY, after cleanup); Rule-based Reasoning Layer (schema-derived edges with synthetic evidence). Order: SGC → Cleanup → LLM Reasoning → Rule-based.                                |
| **Relation whitelist**           | `_ALLOWED_RELATION_LABELS_GLOBAL`: paper-wording plus camelCase gold labels (normalised).                                                                                                                                                                                                                                                                                                                                          |
| **Evaluation — class**           | Coverage, precision, recall, error taxonomy, structural metrics; `extraction_only`; `clinical_only`; `by_stratum`; `by_stage`.                                                                                                                                                                                                                                                                                                     |
| **Evaluation — relations**       | `compute_relation_metrics()`: label-level precision/recall via `RELATION_ALIASES_CORE`; `per_gold_relation`; stored in `metrics["relations"]` and `metrics["extraction_only"]["relations"]`.                                                                                                                                                                                                                                       |
| **Evaluation — hierarchy**       | `compute_hierarchy_metrics()`: edge-level precision/recall by normalized key matching; stored in `metrics["hierarchy"]`.                                                                                                                                                                                                                                                                                                           |
| **Per-stage ablation**           | `_capture_stage_metrics()` snapshots after each pipeline stage (including hierarchy metrics); written to `metrics["by_stage"]` in `metrics.json`; formatted in `summary.txt`. Stages: extraction → after_sgc → after_cleanup → after_llm_reasoning → after_rule_based → after_gold_filter.                                                                                                                                         |
| **Canonical aliases**            | `CANONICAL_ALIAS_MAP` in `canonical.py`: TBI, CVP, ICP, CPP, GCS, GOSE, MAP variants plus Phase 10 extensions (Fluids, Sedation, Nutrition, Condition). Evaluation synonyms in `synonyms.py` are extended accordingly.                                                                                                                                                                                                             |
| **Hierarchy triggers**           | `HIERARCHY_LEXICAL_TRIGGERS`: `"such as"`, `"is a"`, `"type of"`, `"kind of"`, `"include:"`, `"includes:"`. `extract_hierarchy_from_text()` handles all triggers; `"include:"` preserves compound nouns and strips `"use of"` prefixes.                                                                                                                                                                                            |
| **Merge / build**                | `merge_parsed()`: classes by canonical key, relations by `(label, domain, range)`; `build_ontology`: evidence required; canonical alias map; singular/plural merge; stratum on entities; dedupe and duplicate merge.                                                                                                                                                                                                               |
| **Run label format**             | `Strategy - PipelineMode - LLM - EvalSettings - Advanced`. Consistent in both Python and JavaScript.                                                                                                                                                                                                                                                                                                                               |
| **Artifacts**                    | `ontology.json`, `ontology_raw.json`, `summary.txt` (enhanced), `metrics.json` (including `by_stage`), `improvement_counts.json`, `axiom_violations.json`, patch files, `metadata.json` (including `input_papers`), `run.log`.                                                                                                                                                                                                     |
| **Config files**                 | `demo.json`: includes all benchmark-required keys. `benchmark_template.json`: reusable template.                                                                                                                                                                                                                                                                                                                                   |
| **UI**                           | Flask app with Run experiment, progress, results, run comparison, pipeline view, and run list.                                                                                                                                                                                                                                                                                                                                     |
| **CLI**                          | Config-driven runs; `scope_filter` must be explicit when gold is loaded.                                                                                                                                                                                                                                                                                                                                                           |

This implementation delivers a **modular, hybrid ontology engineering framework** with multiple prompting strategies, multiple LLM providers, a four-mode progressive pipeline abstraction (Strict → Guided → Schema-Completed → Fully Reasoned), complete evaluation against the BrainIT gold standard including **relation recall**, **hierarchy precision/recall**, and **per-stage ablation**, together with reproducible and validated configuration for benchmark runs.
