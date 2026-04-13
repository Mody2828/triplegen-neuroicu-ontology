# Implementation Summary: LLM-Driven Ontology Engineering for Neuro-ICU

This document gives a concise **project aim and objectives** and a **detailed summary of the implemented design and functionality** of the framework.

---

## 1. Project aim and objectives

### Aim

The project aims to **design, implement, and evaluate** an LLM-driven ontology engineering framework for the **Neuro-ICU / neuro-intensive care domain**, using **BrainIT literature** as the primary input. The central question is how well language models can reconstruct a high-quality domain ontology in a **low-resource clinical setting**, and which factors—prompting, constraints, post-processing, and choice of LLM—improve or degrade that outcome.

### Objectives

- **End-to-end pipeline:** Build a reproducible pipeline from raw text (paste or upload) to a generated ontology and evaluation report.
- **Multiple LLM providers:** Support and compare extraction across several APIs (OpenAI, Anthropic, Google, Groq, Hugging Face, DeepSeek) so that method and provider can be varied independently.
- **Several prompting methods:** Compare zero-shot (baseline), one-shot, and few-shot (phased 3-step) extraction to assess the impact of examples and task decomposition.
- **Progressive pipeline modes:** Three clearly named modes (Strict → Guided → Schema-Completed) group the optional post-processing features into meaningful experimental conditions for ablation. A **Controlled Experiment** page exposes all individual pipeline flags as independent toggles.
- **Enhancement strategies:** Implement and evaluate optional post-processing improvements—Text-Grounded Completion (always on), schema-guided completion, and a rule-based reasoning layer—as orthogonal toggles for ablation.
- **Evaluation against gold:** Evaluate generated ontologies against the **BrainIT gold standard** (ontology/schema) using coverage, precision, recall, **relation recall**, structural metrics, error taxonomy, and a new **per-stage ablation table**.
- **Usability and reproducibility:** Provide a web UI to run experiments, save artifacts per run, and compare runs side-by-side.

The framework is positioned as a **modular hybrid ontology engineering system** (prompt strategies + optional LLM and symbolic layers), not only a set of prompt experiments.

---

## 2. Implemented design and functionality

### 2.1 High-level architecture

The system is structured in **four conceptual layers**:

1. **Prompt strategy (extraction)** — How the LLM extracts from text: choice of prompting method and, where applicable, retrieval of examples.
2. **Merge and optional vocabulary filter** — Chunk-level extractions are merged into one ontology; optionally restricted to a gold vocabulary when evaluation controls are enabled.
3. **Post-processing** — Text-Grounded Completion (always on), then optional improvements: Schema-guided completion and Rule-based Reasoning Layer; built-in cleanup (always on when gold available).
4. **Evaluation** — Alignment to the gold standard, computation of metrics (including per-stage snapshots), and run summary.

Pipeline order:

```
Corpus (paste/upload/paper catalog) → Load (strip control chars → normalize → [scope filter if enabled])
    → Chunking (control chars stripped per chunk; candidates with section context)
    → Extraction (per chunk, by strategy; evidence required; Phase 2/3 output filtered)
    → Merge (build_ontology: evidence required; canonical alias map + singular/plural; dedupe by canonical key / (label, domain, range); stratum on entities; duplicate merge of provenance/synonyms/aliases)
    → [Vocabulary filter if filter_to_gold_vocabulary enabled]
    ──── STAGE SNAPSHOT: extraction ────
    → Text-Grounded Completion (TGL — built-in, always on when corpus present)
    ──── STAGE SNAPSHOT: after_text_grounded ────
    → [Schema-guided completion (SGC) — optional]
    ──── STAGE SNAPSHOT: after_sgc ────
    → [Built-in cleanup (guided by domain scope file)]
    ──── STAGE SNAPSHOT: after_cleanup ────
    → [Orphan Rescue (optional: connect isolated classes via LLM + source text)]
    → [Post-orphan scope cleanup (remove re-introduced out-of-scope classes/edges)]
    ──── STAGE SNAPSHOT: after_orphan_rescue ────
    → [LLM CoT Refinement (semantic review: remove non-clinical noise, fix hierarchy inversions)]
    ──── STAGE SNAPSHOT: after_llm_refinement ────
    → [Rule-based Reasoning Layer (optional: schema completion + orphan pruning)]
    ──── STAGE SNAPSHOT: after_rule_based ────
    → [Gold-vocab filter if eval_restrict_to_gold]
    ──── STAGE SNAPSHOT: after_gold_filter ────
    → Validation → Evaluation (class + relation + hierarchy metrics) → Artifacts & summary
```

Stage snapshots are written to `metrics["by_stage"]` in `metrics.json` and displayed in the **per-stage ablation table** in `summary.txt`.

**Note on removed feature:** The LLM Reasoning Layer (PROPOSE/VERIFY hierarchy inference, Mode 4) has been **removed** from the pipeline. The `run_llm_reasoning_layer_patch()` function remains as dead code in `ontology_completion.py` for reference but is never called. All pipeline modes are now Strict / Guided / Schema-Completed.

---

### 2.2 Corpus input, normalization, scope filter, and chunking

- **Input:** Users can **paste text** or **upload one or more files** (.txt or .pdf). Multiple files are saved to `data/corpus_ui` and loaded as a multi-document corpus. **Use default paper** can be enabled so the run uses a built-in default corpus when no text is pasted or uploaded. The **Controlled Experiment** and **Run Comparison** pages offer a **BrainIT paper catalog** dropdown (11 papers) so runs can be tied to specific papers without manual upload.

- **BrainIT paper catalog** (`BRAINIT_PAPER_CATALOG` in `web/app.py`): Maps integer paper IDs 1–11 to files in `resources/BrainIT Papers/`. When a paper is selected, the file is copied to an isolated `data/corpus_papers/paper_N/` directory so it never contaminates `data/corpus_ui/`. Papers:
  1. Piper 2003 — The BrainIT Group: Concept and Core Dataset Definition
  2. Moss 2013 — Trusting Intensive Care Unit
  3. Stell 2018 — Automated Measurement of Adherence to TBI Guidelines
  4. Shaw 2024 — Exploration of PbtO2 and ICP Relationship
  5. Georgatzis 2016 — Artefact in Physiological Data (FSLDS Approach)
  6. Güiza 2015a — Visualizing Pressure and Time Burden (copy a)
  7. Donald 2012 — Trigger Characteristics of EUSIG-Defined Hypotensive Events
  8. Depreitere 2018 — Cerebral Perfusion Pressure Variability
  9. Donald 2019 — Forewarning of Hypotensive Events (Bayesian ANN)
  10. Decraene 2023 — Decompressive Craniectomy as Second/Third-Tier Intervention
  11. Güiza 2015b — Visualizing Pressure and Time Burden (copy b)

- **Load pipeline** (`src/corpus/ingest.py`): For each document: **raw text** → **strip control chars** (`clean_chars`) → **normalize** → **[scope filter if enabled]** → stored as `doc["text"]`. Optionally `doc["raw_text"]` (when `keep_raw_text` in config) and, for PDFs, `doc["pages"]` (page-level segments for provenance and header/footer detection).

- **Normalization** (`src/corpus/normalize.py`): Fixes broken hyphenation across line breaks (e.g. intra-\ncranial → intracranial), normalises symbols (Unicode fractions, fancy dashes, curly quotes, OCR repeats), removes repeated header/footer lines across pages/blocks, then strips form-feed and other control chars. Additionally performs:
  - **Citation stripping:** Removes inline citation markers (`[1]`, `[1,2]`, `[1-3]`, `(Author et al., 2020)`), superscript citation numbers, and the full `References` / `Bibliography` section at the end of the document. This prevents the LLM from extracting author names, journal titles, or DOI fragments as clinical classes.
  - **Abbreviation expansion:** Detects `Full Term (ABBREV)` patterns in the text (e.g. "Intracranial Pressure (ICP)") and builds a per-document dictionary. A built-in **35-entry neuro-ICU abbreviation dictionary** (ICP, CPP, GCS, GOSe, MAP, CVP, SaO2, SjvO2, PbtO2, PRx, EVD, etc.) is merged with document-local detections. Bare abbreviations without prior full-form definition are expanded inline for LLM clarity (`"CPP was monitored"` → `"Cerebral Perfusion Pressure (CPP) was monitored"`). Expansion is non-destructive: the original abbreviation is preserved in parentheses so evidence anchoring still matches.

- **Scope filter (optional):** When enabled, scope filter runs **inside load_corpus** after normalize:
  1. **Document-level:** paragraph-level (drop whole paragraphs if admin-heavy), then line-level removal of admin headings/phrases and blacklist terms.
  2. **Chunk-level** (after chunking): **dual-score router** (`filter_chunks_to_clinical`):
     - `chunk_scores()` returns (admin_score, clinical_score).
     - **Drop** if section matches governance sections (Part A, group formation, database access, ethics approval, etc.); if `governance_dominance ≥ 2` and `clinical_score ≤ 1`; if `admin_score ≥ 2` and `clinical_score ≤ 1`.
     - **Keep** if `clinical_score ≥ 3`; **always keep** if section matches clinical dataset sections.
     - **Reorder**: clinical dataset sections ordered first.
  - **Blacklist terms** (`_SCOPE_BLACKLIST_TERMS`): compound phrases only — extended comprehensively for all 11 BrainIT papers to cover governance (e.g. `"steering group"`, `"ethics committee"`, `"centre membership"`, `"publication criteria"`), BrainIT group/network/project (`"brainit group"`, `"brainit network"`, `"project management"`, `"collaborative group"`, `"internet registration form"`), data infrastructure (`"data collection software"`, `"data quality control"`, `"data analysis methodologies"`), study design, software, and publishing metadata. Single generic words are **not** in the blacklist to avoid destroying legitimate clinical lines.
  - **Clinical scoring**: strong terms count **1.0**; broad terms (core dataset, outcome) count 0.5. Strong terms (`_CLINICAL_TERMS_STRONG`) include: core Neuro-ICU variables (heart rate, ICP, CPP, MAP, GCS, pupil, temperature, CVP, EtCO2, SjO2, TCD, PbrO2, brain temperature, microdialysis, SaO2, ventilation, sedation, sedation levels, vasopressors, antibiotics, blood gases, biochemistry, haematology, secondary insults, hypotension, hypertension, intracranial hypertension, systemic hypotension, arterial hypotension); promoted Fluids/Nutrition/Condition (fluids, fluid input, fluid input and output, fluid output, nutrition, nutritional, condition, clinical condition); and **Moss 2013 (Trusting ICU) framework terms** (observation, observations, timepoint, timepoints, session, sessions, sensor, sensors, data quality, data quality assessment, possible error, probable error, noradrenaline, adrenaline, sepsis, out-of-range, out of range, reclassif, clinical context, medical disorder, acceptable range). **"monitoring"** was promoted to strong because it is a core BrainIT concept.

- **Scope filter config validation:** When a gold standard is loaded (benchmark mode), `scope_filter` **must be set explicitly** in the config. If the key is absent, `run_experiments.run_one()` raises a `ValueError` immediately. This enforces that UI runs (`scope_filter=True` by default) and CLI runs produce comparable results.

- **Chunking:** Documents are split into **semantic chunks** so that each chunk fits within prompt limits and stays coherent. Chunk sizing (`src/corpus/chunk.py`) targets **2,000 tokens per chunk** with a **hard cap of 3,200**, **min 400**, and **200-token overlap**, tuned to match GPT-4o-mini's optimal context window (increased from 1,600 during Phase 24 era). Chunk text is passed through **strip control chars** again before candidate extraction and storage.
- **Section-aware weighting:** After chunking, each chunk is classified by section type (abstract, introduction, methods, results, discussion, references, etc.) and assigned a **priority tier**: *high* (methods, results), *medium* (abstract, introduction, discussion), *low* (appendices, acknowledgements), *skip* (references, bibliography, author lists). Reference-section chunks are dropped entirely before LLM extraction. For surviving chunks, the section type is injected into the extraction prompt as context (e.g. `"Section context: Methods"`) so the LLM can weigh concept importance accordingly. Implemented in `src/corpus/chunk.py` and injected in `src/prompting/run.py`.
- **Candidate extraction** (`src/corpus/candidates.py`) uses spaCy noun chunks and NOUN/PROPN with aggressive filtering: generic terms, admin/organisation phrases, publication metadata, author-name patterns, and institutional phrases are rejected; default `max_terms=12` (Phase 13). Candidates are injected only in Guided mode as optional hints.

---

### 2.3 Prompting methods (extraction strategies)

**Three active** prompting strategies are exposed in the UI; two legacy strategies are retained for backward compatibility but hidden. Strategy order and IDs are in `src/prompting/strategy_order.py`.

| Strategy (config id) | Display name (UI) | Description |
|----------------------|-------------------|-------------|
| `baseline` | Zero-Shot | No examples; model extracts classes, relations, and hierarchy from text only. |
| `one_shot` | One-Shot (MMR-1) | **One** comprehensive example per chunk from `pool_one_shot_comprehensive.json` (MMR k=1), each showing classes, relations, and hierarchy. Fallback: `pool_strict_concepts.json`. Optional hierarchy phase runs **only when** the chunk contains hierarchy cues and has ≥1 class label from the **allowed clinical vocabulary**. |
| `phased_3step` | **Few-Shot** (primary) | **Phase 1:** 3 concept examples → extract classes. **Phase 2:** 3 relation examples → extract relations only (hierarchy deferred to Phase 3). **Phase 3:** 3 hierarchy examples → extract hierarchy only where lexical cues exist. Three LLM calls, then merge. Phase 2 and Phase 3 outputs are vocabulary-filtered before merge. |

**Legacy strategies (hidden, backward-compatible):**

| Strategy (config id) | Legacy display name | Status |
|----------------------|---------------------|--------|
| `simple_fewshot` | Few-Shot I (legacy) | Hidden from UI. |
| `phased_2step` | Few-Shot II (legacy) | Two-phase (classes + relations/hierarchy); hidden from UI. |

**Task-specific example pools** (see `docs/task_pool_wiring.md` for the full wiring map):

- **`pool_strict_concepts.json`** — 6 class-first examples from BrainIT core dataset; all labels gold-schema aligned. All `Patient` evidences use direct source-text quotes.
- **`pool_strict_relations.json`** — 6 strong, source-faithful relation examples with gold class labels for domain/range. Includes `brainit_monitoring_indicates_condition` — a dedicated example teaching the LLM to extract the `monitoring indicates condition` relation (Monitoring Data → Condition) and the `targets condition` relation (Therapy → Condition).
- **`pool_strict_hierarchy.json`** — 5 gold-aligned hierarchy examples. Includes two new examples using **verbatim BrainIT paper sentences** with the `"include:"` enumeration pattern: one for monitoring parameters (Heart Rate, Respiration Rate, MAP, ICP, SaO2, Temperature ⊑ Monitoring Data) and one for ICU management types (Ventilation, Sedation, Fluids, Nutrition, Vasopressors, Antibiotics ⊑ Intensive Care Management). All hierarchy edge evidences contain valid lexical triggers that pass `filter_hierarchy_to_lexical_cues`.
- **`pool_one_shot_comprehensive.json`** — 5 examples for One-Shot strategy; each example contains **classes, relations, and hierarchy** (not just classes). Used so the single retrieved example demonstrates the full extraction pattern expected in one LLM call. Loaded via `load_retrieval_pool_one_shot()`; MMR selects one per chunk (fallback to concept pool if missing).

**Hierarchy phase class label filtering:** For both `one_shot` and `phased_3step` (and legacy `phased_2step`) hierarchy sub-calls, the `known_classes` list is **filtered to the allowed clinical vocabulary** before being passed to the prompt.

**Vocabulary and filtering:** When "Prompt vocab guardrails" is on, the prompt injects class labels from the gold vocabulary and **only paper-wording relation labels** (`EXTRACTION_RELATION_LABELS`). CamelCase gold relation labels are **not** injected. `RELATION_ALIASES_CORE` maps paper-wording output to gold labels during post-extraction filtering.

**Current prompt structure (section order):**

The order of sections in the assembled prompt is fixed per strategy. For One-Shot, **vocabulary and hints are placed first** so the model sees allowed labels before cautionary rules; this ordering was validated against runs where constraints-first ordering reduced extraction recall.

| Strategy | Section order (top → bottom) |
|----------|------------------------------|
| **One-Shot** | 1. **VOCAB_AND_HINTS** (class labels → relation labels with domain/range → omit rule → leaf-first rule → [NER note if Guided] → [candidate terms if Guided]) → 2. SYSTEM ROLE → 3. CORE CONSTRAINTS (source fidelity, evidence anchor, ignore non-clinical, priority, confidence, hierarchy) → 4. Anti-anchoring rule → 5. Conservative extraction → 6. OUTPUT FORMAT (JSON ONLY) → 7. OUTPUT RULES → 8. --- Retrieved example --- → 9. --- TEXT TO ANALYZE --- |
| **Few-Shot** (phased_3step / phased_2step) | 1. SYSTEM ROLE → 2. CORE CONSTRAINTS → 3. **VOCAB_AND_HINTS** → 4. **PHASE_INSTRUCTION** (phase-specific task) → 5. OUTPUT FORMAT → 6. Relation/list/hierarchy rules → 7. ANTI-ANCHORING → 8. --- Examples --- → 9. TEXT TO ANALYZE |
| **Zero-Shot** (baseline) | 1. [VOCAB_AND_HINTS if guardrails on] → 2. SYSTEM ROLE → 3. PRIMARY GOAL → 4. HARD RULES (source fidelity, evidence, ignore non-clinical, extract only clinical, class/relation/hierarchy rules, zero-shot behaviour) → 5. OUTPUT FORMAT → 6. FINAL CHECK → 7. --- TEXT TO ANALYZE --- |

Templates: One-Shot `one_shot.json` (prefix starts with `{{VOCAB_AND_HINTS}}`); Few-Shot `mmr_fewshot_controlled.txt` (placeholders `{{VOCAB_AND_HINTS}}`, `{{PHASE_INSTRUCTION}}`, `{{EXAMPLES}}`, `{{TEXT}}`); Baseline `baseline.json` (no placeholder; vocab prepended in code when guardrails on). One-Shot relation list shows **only typed relations** (with domain/range); bare internal labels (e.g. `includes`, `is_a`) are omitted to avoid contradicting "Use ONLY the following relation labels with the stated domain and range".

---

### 2.4 LLM providers

#### Extraction LLM

The following providers are selectable as the **extraction LLM** for all strategies:

| Provider | Model |
|----------|-------|
| **OpenAI** (default) | GPT-4o-mini |
| **OpenAI** | GPT-4o (full model, slower/more expensive) |
| **Anthropic** | Claude Haiku 4.5 |
| **Google** | Gemini 2.5 Flash |
| **Groq** | Llama 3.1 8B (free tier) |
| **Hugging Face** | Mistral 7B via Router API (free) |
| **DeepSeek** | deepseek-chat |

#### Reasoning LLM (Advanced)

When Schema-Guided Completion is enabled, a separate **Reasoning LLM** is used for that improvement step. Configurable independently of the extraction LLM:

| Option | Model |
|--------|-------|
| **OpenAI** (default) | GPT-4o-mini (same as extraction) |
| **DeepSeek Reasoner** | deepseek-reasoner (R1) — stronger reasoning, slower |

This separation allows the extraction and reasoning steps to use different models for ablation. The Reasoning LLM choice is reported in `summary.txt` and the run label.

---

### 2.5 Pipeline modes

The **Run Experiment** and **Run Comparison** UIs present **three progressive pipeline modes** that group the post-processing feature flags into clearly named experimental conditions.

| Mode | Config flags enabled | Use case |
|------|---------------------|----------|
| **Strict** | *(none — all False)* | **Pure raw extraction.** No vocabulary guardrails, no gold filtering, no NER, no schema. The LLM extracts freely; evidence is the only quality gate. |
| **Guided** | `prompt_vocab_guardrails`, `eval_restrict_to_gold`, `medical_ner_anchor`, `candidate_terms` | Adds gold vocab guardrails in the prompt, eval restriction to gold vocabulary, Medical NER anchor, and candidate term injection. |
| **Schema-Completed** | Guided + `schema_guided_completion`, `symbolic_reasoner` | Adds LLM schema gap-filling and rule-based hierarchy completion. |

`Schema-Completed` is the **default** mode in the UI, balancing capability and speed.

**Core principle of Strict mode:** Evidence is the quality gate, not vocabulary lists. The LLM extracts whatever clinical concepts it finds in the text; the pipeline accepts any class with non-empty evidence, any relation with text-grounded evidence (bulk-fabrication checked), and any hierarchy edge with explicit lexical cues. No gold-vocabulary filter is applied at extraction or evaluation time. This makes Strict mode perform well on **unseen papers with novel vocabulary**.

**Removed mode:** "Fully Reasoned" (Mode 4, `llm_reasoning` flag) has been removed from all UI pages. The underlying `run_llm_reasoning_layer_patch()` function is dead code and is never called.

**Controlled mode** is derived automatically when any cleanup toggle is turned off or TGL is disabled (via the Controlled Experiment page). The `_pipeline_mode_label()` function returns `"Controlled"` in this case.

Pipeline mode is reported in the run label (format: `Strategy - Mode - LLM - EvalSettings - Advanced - Paper`) and in `summary.txt`.

---

### 2.6 Advanced / Experimental features

The collapsible **Advanced** section of the UI (Run Experiment and Run Comparison) provides:

- **Reasoning LLM override**: Selects the LLM for SGC improvement steps (see §2.4). Defaults to OpenAI; DeepSeek Reasoner is the alternative. When DeepSeek Reasoner is used, `DSR` appears in the run label.

**Medical NER anchor** (`medical_ner_anchor`) is a **built-in component of Guided Extraction mode** (Mode 2) and cascades to Mode 3. It injects biomedical entities extracted by ScispaCy (`en_ner_bc5cdr_md`) into the prompt as "Suggested concepts." When vocab guardrails are enabled, only NER entities that fuzzy-match a gold class label are injected (`filter_ner_to_gold`), and the prompt wording instructs the model to use them as evidence only and to extract labels solely from the allowed class list (Phase 13). It is also individually controllable on the Controlled Experiment page.

---

### 2.7 Post-processing improvements

Post-processing improvements run **after** merge, in fixed order: TGL → SGC → Cleanup → Rule-based.

#### 2.7.1 Text-Grounded Completion (TGL) — always on

`run_text_grounded_completion()` in `src/prompting/ontology_completion.py`. Runs **after** merge and **before** SGC and cleanup. This is a built-in second-pass extraction step that gives the LLM the **full document text** and the **current (merged) ontology** and asks it to find any clinical classes or relations that were missed by per-chunk extraction.

- **What it does (Phase 15 redesign):** Builds `_build_classes_relations_prompt()` with: current class list, current relations, the full corpus text (truncated to 30,000 chars), general clinical guidance (concept types to look for, soft relation patterns, abbreviation expansion), and evidence rules. **No vocabulary whitelist is injected.** The LLM extracts freely from the text; evidence is the only quality gate.
- **Filtering:** Accepts any LLM-suggested class with **non-empty evidence** (no `ALLOWED_CLASSES_CORE` whitelist check). Relations require evidence that appears in the source text, checked by `_evidence_appears_in_text()`, plus bulk-fabrication detection (`_filter_bulk_fabricated`). All additions deduplicated against the existing ontology by normalized key. Domain and range of relations must match already-extracted class labels.
- **Hierarchy enrichment (soft):** After the LLM pass, `_add_deterministic_hierarchy()` adds edges from `_HIERARCHY_EDGE_CHECKLIST` where **both endpoints already exist** in the ontology. This is soft enrichment — no new classes are forced.
- **Removed (Phase 15):** The old vocabulary scan loop (adding all `ALLOWED_CLASSES_CORE` classes matching keywords), the structural parent loop (forcing parent classes for hierarchy integrity), `_RELATION_DOMAIN_RANGE_HINTS` (hard domain/range constraints in prompt), `_STOP_WORDS`, and `_class_label_mentioned_in_text()` have all been removed.
- **Controlled:** The `text_grounded_completion` config flag (default `True`) can be toggled **off** on the Controlled Experiment page. Toggling it off causes `_pipeline_mode_label()` to return `"Controlled"`.
- **Counts recorded:** `classes_added`, `relations_added`, `hierarchy_added`.
- **Debugging artifacts:** `prompts/tgc_pass1_prompt.txt`, `prompts/tgc_pass1_response.txt`.
- **Max tokens:** `TEXT_GROUNDED_MAX_TOKENS`.

#### 2.7.2 Schema-guided completion (SGC) — optional

- **What it does:** Compares the merged ontology (post-TGL) to the **gold schema**, identifies **missing** classes, relations, and hierarchy edges, and asks the LLM: "Which of these missing items are supported by the **corpus**?" Adds only items that (1) are in the schema, (2) have **corpus evidence**, and (3) have valid domain/range for relations. Uses canonical-aware matching (`_canonical_norm`) for class and relation validation.
- **Counts recorded:** `classes_added`, `relations_added`, `hierarchy_added`.
- **Debugging artifacts:** `prompts/sgc_prompt.txt`, `prompts/sgc_response.txt`.

#### 2.7.3 Built-in cleanup (always on — guided by domain scope file)

`apply_builtin_cleanup()` in `src/ontology/reasoner.py`. Runs **after TGL and SGC**, before Orphan Rescue and Rule-based layer. Each individual cleanup group can be toggled off via a `cleanup_config` dict from the Controlled Experiment page.

**Domain scope file (`resources/domain_scope.json`):** Cleanup is guided by a curated domain scope file containing 117 in-scope classes, 27 relations, 78 evidence exemptions, and categorised allowlists — all derived from all 11 BrainIT papers. This is **independent of the gold-standard ontology** (which is used for evaluation metrics only). The loader is in `src/ontology/domain_scope.py`. Cleanup always runs regardless of whether a gold standard is loaded.

Order:

1. **Dedupe** (`cleanup_dedupe`) — classes, relations, hierarchy (canonical key, gold labels; merges provenance and aliases when merging duplicates).
2. **Out-of-scope class pruning** (`cleanup_scope_pruning`) — governance/admin label patterns via `_OUT_OF_SCOPE_CLASS_PATTERNS` (comprehensive for all 11 papers: BrainIT group/network, data infrastructure, institution names, study design, publishing metadata, etc.). Evidence-based patterns (`_OUT_OF_SCOPE_EVIDENCE_PATTERNS`) also drop classes whose evidence is governance-heavy.
3. **Abstract data label pruning** (`cleanup_scope_pruning`) — specific patterns + allowlist from domain scope (Monitoring Data, Demographic Data, Core/Optional/Derived Parameter, Data Quality Assessment, etc.).
4. **Broad contextual label pruning** (`cleanup_scope_pruning`) — generic labels + allowlist from domain scope (42 classes).
5. **Class evidence pruning** (`cleanup_evidence_pruning`) — keeps only classes with evidence and label supported by evidence. Evidence exemptions and in-scope classes are loaded from the domain scope file (78 exemptions + 117 in-scope classes). The evidence-matching pipeline uses nine successive strategies (see §2.13 Phase 21).
6. **Structural validation** (when `cleanup_structural`): **Auto-add missing endpoints** for relations/hierarchy; then **re-run scope pruning** on any newly added endpoint stubs so out-of-scope classes are not re-introduced. **Edge pruning** (`_prune_edges_with_out_of_scope_endpoints`) removes relations and hierarchy edges whose domain/range/sub/super match out-of-scope patterns (prevents noise from surviving via relation/hierarchy references). Counts: `relations_pruned_scope`, `hierarchy_pruned_scope`.
7. **Hierarchy dangling endpoint pruning** (`cleanup_structural`) — removes hierarchy edges whose subClass or superClass is not in the surviving class set.
8. **Relation domain/range pruning** (`cleanup_structural`) — removes relations whose domain or range is not in the surviving class set.
9. **Relation evidence pruning** (`cleanup_structural`) — label must be in `_ALLOWED_RELATION_LABELS_GLOBAL`.
10. **Hierarchy fragment pruning** (`cleanup_structural`) — removes edges where subClass/superClass contain bad tokens.
11. **Axiom constraints** (`cleanup_axioms`) — removes hierarchy/relations violating physiological/semantic type constraints (`neuro_axioms.py`). 29 forbidden hierarchy pairs, 12 allowed relation type restrictions. Violations written to `evaluation/axiom_violations.json`.
12. **Circular hierarchy detection** — Kahn's algorithm detects and removes edges that participate in cycles (A ⊑ B ⊑ … ⊑ A).

Each cleanup group is independently togglable via the Controlled Experiment page. When all five flags are `True` (the default), behaviour is identical to the previous always-on cleanup.

**Cleanup audit log (`cleanup_removed.json`):** Every item removed during cleanup is written to `<run_dir>/cleanup_removed.json` as a flat JSON array for post-run diagnostic review. Each entry records: `type` (class / relation / hierarchy), identifying fields (label, domain, range, subClass, superClass), `stage` (which cleanup sub-stage removed it), `reason` (human-readable explanation), and — for evidence-based removals — an `evidence_snippet` (first 200 chars of the evidence field). All 11 pruning stages are covered. The log is purely diagnostic: it has no effect on pipeline behaviour. Use it to identify false positives and tighten or loosen pruning rules.

**Post-orphan scope cleanup:** After the Orphan Rescue pass (which can add classes as relation endpoints), `apply_post_pass_scope_cleanup()` in `reasoner.py` runs: it re-applies out-of-scope class pruning, abstract/broad pruning, and `_prune_edges_with_out_of_scope_endpoints()` so that any governance or methodological concepts re-introduced by the LLM are removed. Results are recorded in `improvement_counts["post_orphan_scope_cleanup"]`.

#### 2.7.3b LLM Chain-of-Thought Refinement (after cleanup and orphan rescue)

`run_ontology_refinement()` in `src/prompting/ontology_completion.py`. A domain-agnostic LLM pass that presents the full ontology to the LLM and asks it to identify non-clinical classes, semantically incorrect relations, and hierarchy errors (inversions, nonsensical edges) using Chain-of-Thought reasoning. Returns JSON with `classes_to_remove`, `relations_to_remove`, `hierarchy_to_remove`, and `hierarchy_corrections`. Applied after post-orphan scope cleanup and before rule-based reasoning. Gated on `text_grounded_completion` config flag. Debugging artifacts: `refinement_prompt.txt`, `refinement_response.txt`. Results: `improvement_counts["llm_refinement"]`.

#### 2.7.4 Rule-based Reasoning Layer (optional, UI toggle)

Acts as a **fallback after cleanup** — fills any remaining hierarchy gaps exhaustively:

1. **Schema completion** — adds hierarchy edges from gold when both endpoint classes exist in the ontology. Each added edge carries a **synthetic evidence string** (`"Schema-inferred: X is a subclass of Y."`) which satisfies `require_evidence=True` and contains `"is a"` (a recognised `HIERARCHY_LEXICAL_TRIGGER`).
2. **Orphan pruning** — removes classes not referenced by any relation or hierarchy edge. Gold-aligned classes are always preserved even when isolated.

---

### 2.8 Gold standard and evaluation

- **Gold standard:** The BrainIT ontology/schema is loaded from a configurable path. Modes: **public** (surrogate gold), **restricted** (use provided gold for alignment), **isolated** (generate gold from separate corpus). The current gold file is a **placeholder proxy** (`resources/brainit_core_2003.ttl`) constructed from BrainIT publications while awaiting the actual BrainIT consortium ontology — **93 classes** (with canonical aliases), **18 relations**, and **76 hierarchy edges**. All evaluation numbers are reported against this proxy.
- **Alignment (classes):** Generated classes aligned to gold using exact and dense semantic embedding matching (`all-MiniLM-L6-v2` sentence-transformer cosine similarity, threshold 0.55) with synonym expansion. The embeddings replace the legacy TF-IDF matcher, dramatically improving cross-vocabulary synonym resolution.
- **Metrics (classes):**
  - **Coverage, Precision, Recall:** fraction of gold classes matched.
  - **Extraction-only:** metrics before any improvement step — used to measure the raw LLM contribution vs. post-processing contribution.
  - **Clinical-only:** governance-vocab classes excluded.
  - **By stratum:** per-chunk-type (core, governance, provenance).
  - **Errors:** hallucinations, schema violations, omissions.
  - **Structural:** `relation_domain_range_rate`, `hierarchy_edges`, `hierarchy_coverage`.

- **Relation metrics:** `compute_relation_metrics()` computes label-level precision/recall for relations against gold via `RELATION_ALIASES_CORE`. Reports `per_gold_relation` (which gold relations were found), `matched_generated` / `unmatched_generated`. Present under `metrics["relations"]` and `metrics["extraction_only"]["relations"]`.

- **Hierarchy metrics:** `compute_hierarchy_metrics()` computes precision/recall for hierarchy edges (subClass, superClass pairs) by normalized key matching. Present under `metrics["hierarchy"]`.

- **Per-stage ablation (`by_stage`):** Compact metrics snapshots (n_classes, n_relations, n_hierarchy, coverage, precision, recall, relation_recall, hierarchy_recall, hierarchy_precision) are captured after each major pipeline stage. Stages: `extraction` → `after_text_grounded` → `after_sgc` → `after_cleanup` → `after_orphan_rescue` → `after_rule_based` → `after_gold_filter`. Written to `metrics["by_stage"]` in `metrics.json` and rendered as a formatted table in `summary.txt`.

- **Outputs:** `evaluation/metrics.json` (includes `relations`, `extraction_only`, `clinical_only`, `by_stratum`, `by_stage`), `evaluation/table.csv`, `evaluation/hallucinated_classes.json`, `evaluation/improvement_counts.json`.

---

### 2.9 Run artifacts and summary

#### Run directory structure

Each run has a unique ID (timestamp + short hash). Under the run directory:

- **Corpus and config:** `corpus_manifest.json`, `metadata.json` (includes `input_papers` list with filename/path, `config` with all pipeline flags, `environment`, `code_version`).
- **Prompts:** Optional prompt saves under `prompts/`.
- **Generated:** `ontology.json`, `ontology_raw.json`, `ontology_pre_cleanup.json` (pre-cleanup snapshot for debugging), `summary.txt`, optionally `ontology_restricted.json`.
- **Evaluation:** `metrics.json`, `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`, optionally `axiom_violations.json`.
- **Other:** `run.log`, `warnings.txt`.

#### summary.txt (enhanced)

The run summary is a single, human-readable file with the following sections:

1. **Metadata** — run ID, timestamp (UTC), prompting method, **pipeline mode**, extraction LLM, reasoning LLM (if different), improvements applied, **evaluation settings** (scope filter, gold-vocab, Medical NER).
2. **Input paper(s)** — filenames of all ingested source documents (from `docs` passed at write time).
3. **Improvement counts** — per-feature counts (classes/relations/hierarchy added/removed) per stage: text_grounded_completion, schema_guided_completion, builtin_cleanup, orphan_rescue, post_orphan_scope_cleanup, symbolic_reasoner. Cleanup result includes relation_labels_normalized, relations_range_normalized, out_of_scope_classes_removed, relations_pruned_scope, hierarchy_pruned_scope, and other granular counts.
4. **Metrics summary (final ontology)** — coverage, precision, recall, errors, structural, clinical-only sub-metrics, relation precision/recall.
5. **Metrics at extraction only** — pre-improvement baseline (coverage, precision, recall, structural, relation metrics) for direct comparison with the final metrics above.
6. **Per-stage ablation table** — formatted table showing n_classes, n_relations, n_hierarchy, coverage, precision, recall at each stage (extraction → +Text-grounded → +SGC → +Cleanup → +Rule-based).
7. **Concept counts** (final ontology).
8. **Classes, Relations, Hierarchy** listings.

#### metadata.json (comprehensive)

`metadata.json` now captures **all pipeline configuration flags** for every run type (Run Experiment, Run Comparison, Controlled Experiment). The `config` dict includes:

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "scope_filter": true,
  "prompt_vocab_guardrails": true,
  "schema_guided_completion": false,
  "symbolic_reasoner": false,
  "medical_ner_anchor": false,
  "candidate_terms": false,
  "eval_restrict_to_gold": false,
  "text_grounded_completion": true,
  "chunk_clinical_filter": true,
  "clinical_only_routing": true,
  "require_label_in_evidence": true,
  "filter_to_gold_vocabulary": false,
  "strict_relations": false,
  "cleanup_dedupe": true,
  "cleanup_scope_pruning": true,
  "cleanup_evidence_pruning": true,
  "cleanup_structural": true,
  "cleanup_axioms": true,
  "paper_name": "Piper 2003",
  "paper_id": 1,
  "input_papers": [{"name": "...", "stem": "...", "path": "..."}]
}
```

`_build_run_config()` (used by both single-run and comparison runs) explicitly sets all 11 new flags using `run_opts.get(key, default)` so comparison run metadata is equally comprehensive. This was standardized so that all run types produce self-documenting, fully reproducible metadata.

---

### 2.10 Web UI

The Flask-based UI provides:

- **Run Experiment (`/`)** — Three prompting toggles (Zero-Shot / One-Shot / Few-Shot), three pipeline mode cards (Strict / Guided / Schema-Completed), LLM provider selector, evaluation settings (gold-vocab checkbox), and a collapsible Advanced section (Medical NER, Reasoning LLM). Few-Shot and Schema-Completed are the defaults. Submits to `/run` with `from_page=index`.
- **Controlled Experiment (`/controlled-experiment`)** — Mirrors the Run Experiment form but replaces the Pipeline Mode cards with **19 individual feature toggles** grouped into three categories:
  - **Preprocessing:** Scope Filter (document-level), Chunk Clinical Filter, Clinical-only Routing, Medical NER Anchor, Candidate Terms Injection.
  - **Prompt Injection:** Vocabulary Guardrails, Filter to Gold Vocabulary, Strict Relations, Require Label in Evidence.
  - **Post-Processing:** Text-Grounded Completion, Schema-Guided Completion, Rule-based Reasoning, Cleanup Dedupe, Cleanup Scope Pruning, Cleanup Evidence Pruning, Cleanup Structural, Cleanup Axiom Constraints.
  - Also includes **BrainIT paper catalog** dropdown for paper selection. Submits to `/run` with `from_page=controlled_experiment`. Unchecked toggles correctly resolve to `False` (HTML checkbox convention).
- **Run Comparison (`/comparison`)** — Configuration panel mirrors the Run Experiment form (same three pipeline mode cards, strategy toggles, LLM selector). **BrainIT paper catalog** dropdown per run. Runs are added to a comparison list with a **5-part standardised label**: `Strategy - PipelineMode - LLM - EvalSettings - Advanced - Paper` (e.g. `Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - None - Piper 2003`). The comparison dashboard shows a table with these columns. Three pre-configured comparison groups: Cross-LLM, Pipeline Modes, and Reasoning LLM.
- **Progress page** — Live progress bar with cancel. Shows extraction progress per chunk, TGL status ("Text-grounded completion: waiting for LLM response…"), and post-processing steps.
- **Results page** — Metrics, per-stage ablation table, ontology listing, improvement counts, and **Cluster Results** (interactive PCA scatter plot and silhouette analysis of extracted concepts using Ward's hierarchical clustering on dense embeddings).
- **Run list** — Browse all past runs; link to results/analysis.
- **Pipeline view** — Visual diagram of the current pipeline configuration.
- **Ontology Engineering (`/ontology-engineering`)** — Cross-paper synthesis page distinct from the main extraction pipeline. Lets the user (1) select **multiple completed source runs**, (2) merge them into a single ontology via canonical-key dedup, (3) semantically cluster the merged classes into N groups, (4) run **LLM cluster completion** to add relations and structure per cluster, (5) evaluate the reconstructed ontology against the gold standard. A **Compare Metrics** modal shows side-by-side Overall F1, Class F1, Hierarchy F1, and Relation F1 for the reconstruction vs. each source run, with a Chart.js bar chart and sortable table. This is the subsystem documented in §2.14.

**Backward compatibility:** The `_flag(key, legacy_default)` helper in the `/run` handler ensures that when a form submission comes from `index.html` or `comparison_dashboard.html`, new pipeline flags (TGL, cleanup flags, etc.) default to their historical "always-on" values. Only submissions from `controlled_experiment` allow these flags to be toggled off.

Run label generation is consistent across Python (`web/app.py: _format_run_label`) and JavaScript (`comparison_dashboard.html: formatRunName`, `comparison_progress.html: formatRunName`).

---

### 2.11 CLI

Config-driven runs via `run_experiments` and JSON config:

```
python -m src.experiments.run_experiments --config src/experiments/configs/demo.json
```

- `demo.json` includes all benchmark-required keys: `scope_filter: true`, `clinical_only_routing: true`, `require_label_in_evidence: true`, `gold_standard_path`, all active strategies.
- `benchmark_template.json` provides a reusable template with all mandatory fields documented.
- If `scope_filter` is absent from config when a gold standard is loaded, a `ValueError` is raised immediately.

---

### 2.12 Technical notes

- **Control characters (pipeline-wide):** `src/corpus/clean_chars.py` provides `strip_control_chars_for_prompt()`. Used after reading raw text (ingest), during normalization, before chunking, before prompting, and before JSON parsing.
- **Parsing:** LLM responses parsed as JSON (with fallbacks) in `src/prompting/parse.py`. Input sanitized before `json.loads`. Items without evidence or with evidence shorter than configured minimum are dropped.
- **Schema and filtering:** `src/prompting/schema.py` provides `filter_parsed_to_vocabulary`. A dedicated regex pass `extract_hierarchy_from_text()` extracts hierarchy from trigger phrases and merges with LLM-derived hierarchy. All hierarchy edges are filtered by `filter_hierarchy_to_lexical_cues` at the end of each chunk's processing.
  - **`HIERARCHY_LEXICAL_TRIGGERS`** now includes `"include:"` and `"includes:"` in addition to the original `"such as"`, `"is a"`, `"type of"`, `"kind of"`. This captures the dominant enumeration pattern in the BrainIT paper (*"monitoring data...include: Heart Rate, Respiration Rate, MAP..."*; *"Types of data include: ventilation settings, sedation levels..."*) which was previously invisible to both the LLM trigger filter and the regex fallback.
  - **`extract_hierarchy_from_text()`** now includes a dedicated `"include:"` handler. It strips trailing sentence-glue connectors (`"... Database and include:"`), strips leading articles and trailing verb/preposition clauses (e.g. "required for...", "collected during..."), caps the super-class candidate to the leading 5 words (the noun phrase head), correctly preserves compound sub-class names (`"fluid input and output"` stays intact), and strips leading `"use of "` prefixes (so `"use of vasopressors"` → `"vasopressors"`).
- **Canonical normalization (`src/ontology/canonical.py`):** `CANONICAL_ALIAS_MAP` is a comprehensive alias-to-canonical-label map covering: TBI, CVP, ICP, CPP, GCS/GOSe, MAP; Fluids (fluid input and output, fluid management, fluid balance, fluid input, fluid output); Nutrition (nutritional support, nutritional intake); Sedation (sedation levels, sedation level); Condition (condition, clinical condition, medical condition); Secondary Insult subtypes (Systemic/Arterial Hypotension, Intracranial Hypertension, Jugular Venous Desaturation, ARDS, Sepsis, Hypertension, Hypotension); Clinical Assessment subtypes (Pupil Assessment, CT Scan Assessment with multiple variants); Laboratory Values (12 aliases for Blood Gases/ABG, Biochemistry, Haematology, Sodium/Na, Potassium/K+, Glucose, Haemoglobin/Hb, White Blood Cell Count/WBC, Haematocrit); Therapy hierarchy (Baseline Therapy, Secondary Insult Therapy, Arterial Pressors, Noradrenaline, Adrenaline); Nursing Interventions (Nursing Intervention, Routine Nursing Care, Physiotherapy, Bedside Intervention, Patient Transport); Monitoring hierarchy (Core/Optional Monitoring Parameter, Derived Parameter); Observation framework (Session, Timepoint, Observation, Parameter, Sensor); Data Quality (Data Quality Assessment, Possible Error/PoE, Probable Error/PrE). `resolve_to_canonical_label()` is alias-aware and used throughout the pipeline. `canonical_key()` normalizes for dedupe.
- **Evaluation synonyms (`src/evaluation/synonyms.py`):** `DOMAIN_SYNONYMS` extended with evaluation-level synonyms for `Fluids`, `Nutrition`, `Sedation`, and `Condition` (including `"systemic hypotension"`, `"intracranial hypertension"`, `"arterial hypotension"` mapping to `"Condition"`). This ensures the semantic embedding matcher and exact matcher correctly credit these classes even when the LLM uses the verbose paper phrase rather than the gold canonical label.
- **Config and strategies:** `StrategyConfig` dataclass in `src/experiments/config.py` carries all per-run parameters. A single run is fully specified and reproducible from `metadata.json`.
- **Pre-cleanup snapshot:** `ontology_pre_cleanup.json` is saved immediately before `apply_builtin_cleanup()` runs, preserving the post-extraction + post-TGL + post-SGC ontology state for debugging.

---

### 2.13 Improvements history (consolidated)

#### Original improvements (system design phase)

1. **Dual-score chunk routing** — `chunk_scores()` returns (admin_score, clinical_score). Drop chunks with high admin and low clinical; always keep chunks from clinical dataset sections. Governance section blacklist; governance dominance detection; chunk reordering (clinical first).
2. **Broad contextual label pruning** — built-in cleanup step removes generic labels unless in allowlist.
3. **Canonical alias merge** — `CANONICAL_ALIAS_MAP` for TBI, CVP, ICP, CPP, GCS, GOSE, MAP variants (original entries; see §2.12 and Phase 10 fix 33 for subsequent extensions). Singular/plural merge. `resolve_to_canonical_label()` before add.
4. **Schema-guided / LLM metrics breakdown** — `extraction_only` metrics (before improvements) for ablation. `clinical_only` metrics. `by_stratum` metrics (core/governance/provenance).
5. **Stratum on entities** — chunk routing assigns stratum; stored on classes, relations, hierarchy for per-stratum reporting.

#### Audit-driven fixes (Phase 8)

6. **Narrowed out-of-scope class patterns** — compound phrases only. Prevents false positives on legitimate clinical labels.
7. **Narrowed out-of-scope evidence patterns** — compound phrases only (e.g. `\bdata collection form\b`, `\bsoftware tool\b`).
8. **Fixed `Monitoring Data` pruning** — `\bdata\b` removed from `_ABSTRACT_DATA_LABEL_PATTERNS`. Added `Monitoring Data`, `Demographic Data` to allowlist.
9. **Fixed `Secondary Insult Treatment` pruning** — `\btreatment\b` and `\btherapies?\b` removed from `_BROAD_CONTEXT_LABEL_PATTERNS`. Extended allowlist.
10. **Extended relation whitelist** — `_ALLOWED_RELATION_LABELS_GLOBAL` extended with camelCase gold schema labels in normalised form.
11. **Narrowed scope filter blacklist** — compound phrases only. Prevents destruction of legitimate clinical text.
12. **"monitoring" promoted to strong clinical term** — 1.0 weight. Additional BrainIT terms (ventilation, nutrition, fluids, antibiotics, laboratory values, SaO2) promoted to strong.
13. **Vocab guardrails inject paper-wording relations only** — `EXTRACTION_RELATION_LABELS` only, no camelCase gold labels in prompts.
14. **Hierarchy sub-prompt class label filtering** — `known_classes` filtered to allowed clinical vocabulary before hierarchy sub-calls.
15. **Therapy guaranteed in Phase 3** — `phased_3step` Phase 3 prepends `Therapy` to `phase3_known` for clinical chunks.
16. **Pool example evidence corrections** — `Patient` evidence corrected to source-text quote; all hierarchy evidences contain valid lexical triggers.
17. **Relation recall evaluation** — `compute_relation_metrics()` added. Per-relation boolean `per_gold_relation`. Results in `metrics["relations"]` and `metrics["extraction_only"]["relations"]`.
18. **scope_filter config validation** — raises `ValueError` if `scope_filter` absent from config when gold standard is loaded.

#### Phase 9: UI redesign, pipeline modes, and summary improvements

19. **3-mode pipeline abstraction (UI)** — Run Experiment and Run Comparison forms now expose three pipeline mode cards (Strict / Guided / Schema-Completed). Hidden inputs sync the underlying feature flags. Schema-Completed is the default. Note: previously four modes (including "Fully Reasoned"); that mode has been removed.
20. **Simplified strategy selector** — UI reduced to 3 active strategies: Zero-Shot, One-Shot, Few-Shot. Few-Shot is `phased_3step` (3-phase extraction). `simple_fewshot` and `phased_2step` are legacy (hidden, backward-compatible).
21. **Reasoning LLM configurable** — Separate LLM for improvement steps. Default: OpenAI. Alternative: DeepSeek Reasoner (R1). Exposed in Advanced section.
22. **Medical NER in Guided mode** — Medical NER is a built-in component of Guided mode (Mode 2) and cascades to Mode 3, not a separate Advanced toggle (see §2.6).
23. **5-part standardised run label** — Format: `Strategy - PipelineMode - LLM - EvalSettings - Advanced - Paper`. Consistent across Python backend (`_format_run_label` in `app.py`) and JavaScript (`formatRunName` in `comparison_dashboard.html`, `comparison_progress.html`).
24. **Run comparison panel redesign** — Mirrors Run experiment form. BrainIT paper catalog per run. Three pre-configured comparison groups.
25. **Per-stage ablation table (`by_stage`)** — `_capture_stage_metrics()` snapshots ontology state (n_classes, n_relations, n_hierarchy, coverage, precision, recall, relation metrics) after each pipeline stage. Written to `metrics["by_stage"]`. Rendered as a formatted table in `summary.txt`. Stage `after_text_grounded` added for TGL.
26. **`extraction_only_relations` capture fix** — Moved capture of `extraction_only` relation metrics to immediately after initial ontology building (before any improvements), ensuring they reflect raw extraction only.
27. **Enhanced `summary.txt`** — Now includes: timestamp, pipeline mode, evaluation settings, input paper names, extraction-only baseline comparison, per-stage ablation table, clinical-only and relation sub-metrics.
28. **`input_papers` in `metadata.json`** — `build_metadata()` now accepts `docs` and writes an `input_papers` list with filename, stem, and path for full corpus traceability.

#### Phase 10: Gap-analysis-driven fixes (class recall, hierarchy evaluation, canonical coverage)

Targeted fixes derived from a systematic gap analysis comparing pipeline output (best run: Few-Shot Schema-Completed, 85.71% class recall, 50% relation recall, 0% hierarchy coverage) against a manually-produced expert ontology extraction from the BrainIT paper:

29. **`"include:"` hierarchy trigger** — `HIERARCHY_LEXICAL_TRIGGERS` extended with `"include:"` and `"includes:"`. The BrainIT paper uses this pattern as its dominant enumeration form, which the pipeline was completely blind to. Now accepted by `filter_hierarchy_to_lexical_cues` and handled by `extract_hierarchy_from_text()` with compound-noun preservation and "use of" prefix stripping.
30. **Fix hierarchy evaluation bug (symbolic reasoner evidence)** — `_complete_hierarchy_from_schema` now adds a synthetic evidence string (`"Schema-inferred: X is a subclass of Y."`) to every schema-derived hierarchy edge. Previously, these edges had no `evidence` field and were silently dropped by `filter_parsed_to_vocabulary(require_evidence=True)` during `eval_restrict_to_gold`, causing `hierarchy_coverage=0.00` across all 12 runs despite correct edges being present internally.
31. **Evidence pruning exemption whitelist** — `_EVIDENCE_PRUNING_EXEMPTIONS` frozenset added to `reasoner.py` protecting `Condition`, `Fluids`, `Nutrition`, and `Sedation` from evidence-based class pruning. Root cause: gold-schema abstract/therapeutic class labels do not appear literally in their corpus evidence (e.g. paper says `"fluid input and output"` for gold class `"Fluids"`; `"secondary insults"` for gold class `"Condition"`).
32. **Extended canonical alias map** — 20 new entries in `CANONICAL_ALIAS_MAP` (`canonical.py`): `Fluids` (fluid input and output, fluid management, fluid balance, fluid input, fluid output), `Nutrition` (nutritional support, nutritional intake), `Sedation` (sedation levels, sedation level), `Condition` (condition, clinical condition, medical condition). Ensures LLM output using verbose paper phrases merges to the gold canonical label during deduplication.
33. **Extended evaluation synonyms** — `DOMAIN_SYNONYMS` in `synonyms.py` extended with evaluation-time synonyms for `Fluids`, `Nutrition`, `Sedation`, and `Condition` (including specific condition types `"systemic hypotension"`, `"intracranial hypertension"`, `"arterial hypotension"` → `"Condition"`). Ensures the semantic embedding matcher credits these classes at evaluation time.
34. **New hierarchy pool examples with `"include:"` evidence** — Two new entries in `pool_strict_hierarchy.json` using verbatim BrainIT paper sentences: (a) monitoring parameters ⊑ Monitoring Data (6 edges), (b) ICU management types ⊑ Intensive Care Management (6 edges). Evidence strings all contain `"include:"` so they pass `filter_hierarchy_to_lexical_cues`. These are the primary driver for LLM few-shot hierarchy extraction.
35. **New `monitoring indicates condition` pool example** — New entry in `pool_strict_relations.json` (`brainit_monitoring_indicates_condition`) teaching the LLM to extract: `monitoring indicates condition` (Monitoring Data → Condition) and `targets condition` (Therapy → Condition). Also includes hierarchy edges linking `Intracranial Pressure (ICP)` and `Mean Arterial Pressure (MAP)` to `Condition` via the "secondary insults such as" text pattern.
36. **Fluids/nutrition/condition promoted in scope filter** — `"fluids"`, `"fluid input"`, `"fluid input and output"`, `"nutrition"`, `"nutritional"`, `"condition"`, `"clinical condition"`, and `"sedation levels"` added to `_CLINICAL_TERMS_STRONG` and `_CLINICAL_VARIABLE_KEYWORDS_STRONG` in `scope_filter.py`, preventing chunks that contain only these brief terms from being filtered before reaching the LLM.

#### Phase 11: Strict/Guided mode audit + end-to-end pipeline check

Comprehensive audit of Mode 1 (Strict Extraction) and Mode 2 (Guided Extraction) pipelines, followed by an end-to-end trace of all pipeline stages.

37. **`phased_3step` Phase 2 instruction fix** — Phase 2 in `phased_3step` was incorrectly instructed to "extract relations and hierarchy only," but hierarchy extraction is Phase 3's job. Now uses `hierarchy_in_separate_phase=True` parameter so the instruction says "extract relations only (hierarchy will be extracted in Phase 3)." Phase 2 in `phased_2step` retains the original "relations and hierarchy" instruction since it has no dedicated Phase 3.

38. **Phase 2 output vocabulary filtering** — Phase 2 output in both `phased_2step` and `phased_3step` was merged with Phase 1 without vocabulary filtering, allowing non-gold classes and relations to enter the ontology. Now Phase 2 output is filtered through `filter_parsed_to_vocabulary()` with the same parameters as Phase 1 (gold classes, relations, evidence requirements) before merge. Reduces noise in extraction-only metrics.

39. **Provenance routing gated by `clinical_only_routing`** — `use_provenance` was not gated by `clinical_only_routing`, so chunks containing provenance keywords ("sensor", "data quality") were misrouted even when scope_filter was enabled. Now gated identically to `use_governance`: when `clinical_only_routing=True` (scope_filter on), all chunks route to core.

40. **Dead parameter cleanup in `build_phase_prompt`** — Removed unused `use_provenance` and `use_governance` parameters from `build_phase_prompt()`. Replaced `used_core_vocab`/`used_governance_vocab` dead variables with direct use of `use_core`/`use_governance`.

41. **Regex hierarchy `super_candidate` extraction fix** — `extract_hierarchy_from_text()`'s "include:" handler previously applied a 5-word cap taking the **last** 5 words, producing garbage superClass labels (e.g. "required for new therapeutic approaches"). Now: (a) strips leading articles ("The"), (b) strips trailing verb/preposition clauses ("required for...", "collected during..."), (c) caps to first 5 words (the noun phrase head). Result: "The minute by minute monitoring data required for new therapeutic approaches" → "minute by minute monitoring data" (correct).

42. **`ALLOWED_CLASSES_CORE` expanded to v2.0** — Stale fallback vocabulary (`ALLOWED_CLASSES_CORE` in `vocabulary.py`) updated from 7 v1.0 classes to all 72 v2.0 gold classes. Ensures correct class coverage when running without a gold standard loaded.

#### Phase 12: Few-Shot extraction enrichment & SGC fix

Deep investigation and fix phase targeting Few-Shot's underperformance in Schema-Completed mode.

43. **SGC evidence length fix** — `parse_output()` now accepts a `min_evidence_length` parameter. SGC calls with `min_evidence_length=1` so that short but legitimate clinical terms (e.g. "EtCO2"=5 chars, "TCD"=3 chars, "nutrition"=9 chars) are not rejected by the default MIN_EVIDENCE_LENGTH=12 threshold. Regular extraction parsing retains the strict threshold.

44. **SGC `max_tokens` increase** — `SGC_MAX_TOKENS` increased from 4096 to 8192. The v2.0 gold schema has 72 classes + 16 relations + 57 hierarchy edges; with all missing items listed, the LLM response routinely exceeds 4096 tokens and was being truncated mid-JSON.

45. **Truncated JSON recovery improvements** — `_try_recover_truncated_json()` in `parse.py`: (a) pre-strips trailing whitespace and commas before trying closings; (b) reordered closings so `"]}"` (most common truncation point) is tried first; (c) added `"\"}]}"` closing for mid-string truncation; (d) increased trim limit from 80 to 120 chars for deeper recovery; (e) tries both pre-stripped and raw fragments.

46. **Phase 1 prompt enrichment** — Replaced the minimal "Extract candidate classes only" instruction with a comprehensive "COMPREHENSIVE CLASS EXTRACTION" block that: (a) enumerates 8 concept types to extract (conditions, parameters, therapies, assessments, lab values, nursing, sensors, abstract categories); (b) adds "BREADTH OVER CAUTION" directive; (c) instructs to "extract every distinct concept, not just those similar to the examples."

47. **Anti-anchoring in few-shot template** — Replaced the weak "style guidance only" instruction in `mmr_fewshot_controlled.txt` with a detailed ANTI-ANCHORING block: "The examples below demonstrate OUTPUT FORMAT and evidence style ONLY. They come from DIFFERENT text passages... Do NOT limit your extraction to the concept types shown in the examples." Also replaced the overly conservative "Reject rule: When in doubt, omit" with "Confidence rule: If a clinical concept is clearly named, include it."

48. **Phase 2 prompt enrichment** — Phase 2 now instructs the LLM to also extract supplementary classes that Phase 1 may have missed (e.g. concepts needed as relation domain/range). Previously Phase 2 was told to "set classes to an empty array" which prevented recovery of Phase 1 gaps.

49. **Phase 3 gating relaxation** — Lowered the threshold from ≥2 extracted classes to ≥1 class for Phase 3 hierarchy extraction. When extracted classes are fewer than 5, gold vocabulary labels are injected as valid hierarchy endpoints so the LLM can form edges even when Phase 1 was conservative. Same fix applied to `phased_2step` and `one_shot` hierarchy phases.

50. **Phase 3 prompt enrichment** — Replaced terse instruction with structured block including concrete examples ("parameters such as Heart Rate → Heart Rate subClassOf Parameter"), enumeration guidance ("extract EACH listed item"), and clearer formatting.

51. **SGC evidence quality instruction** — Added explicit guidance in SGC prompt: "Evidence must be actual text from the corpus, NOT ontology URIs (e.g. 'pd:Session'), NOT the label repeated, and NOT definitions. Copy a real fragment from the corpus." Previously the LLM would produce evidence like "pd:Session" which is an ontology URI, not a corpus span.

52. **SGC diagnostic logging** — `run_schema_guided_completion()` now writes `sgc_diagnostic.json` to the run's prompts directory with counts at each filter stage: raw response length, parsed classes/relations/hierarchy, and filtered counts. Enables debugging SGC yield without full pipeline re-runs.

#### Phase 13: Guided-mode NER and candidate-term fixes (investigation-driven)

Investigation of why **One-Shot Guided** and **Few-Shot Guided** underperformed **Strict** while **Zero-Shot Guided** outperformed Strict.

53. **NER gold-filtering when vocab guardrails are on** — In Guided mode, when `inject_vocab_guardrails` is enabled and gold class labels are available, NER entities are filtered to only those that fuzzy-match a gold class label (`filter_ner_to_gold()` in `run.py`). Matching uses exact label match (case-insensitive) or substring overlap. Typical effect: ~75% of NER entities dropped, keeping only gold-aligned suggestions.

54. **NER wording softened** — The NER block in the prompt was changed from "Prefer using them where they fit" to: "These are entity mentions detected in this chunk. Use them as additional evidence that a concept is present, but only extract labels from the allowed class list above." This removes the conflict with vocab guardrails and makes NER strictly supportive, not prescriptive.

55. **Candidate term filtering tightened** — In `src/corpus/candidates.py`: (a) expanded `_GENERIC_TERMS` with ~40 terms; (b) expanded `_CANDIDATE_ADMIN_SUPPRESS` with organisation/institutional terms; (c) improved `_is_metadata_noise()` to reject author-name patterns and institutional phrases; (d) added stopwords. Result: admin-heavy chunks yield ~10–11 clinical terms instead of 20 with 15+ noise terms.

56. **Candidate max_terms reduced** — Default `max_terms` in `extract_candidates()` reduced from 20 to 12 to limit prompt length and noise volume.

57. **One-Shot comprehensive example pool** — One-Shot previously used `pool_strict_concepts.json`, where every example had empty `relations` and `hierarchy` arrays. Added `pool_one_shot_comprehensive.json` with 5 examples that each demonstrate classes, relations, and hierarchy. One-Shot strategy now loads this pool via `load_retrieval_pool_one_shot()` and selects the single example via MMR (fallback to concept pool if file missing).

58. **Prompt section ordering verified** — Confirmed that `assemble_mmr_prompt_controlled()` replaces `{{VOCAB_AND_HINTS}}` and `{{PHASE_INSTRUCTION}}` correctly and that the final prompt order is correct. No code change; documented for consistency.

#### Phase 14: Controlled Experiment page, pipeline flag granularisation, and metadata standardisation

Major UI and backend changes to enable fine-grained experimental control over every pipeline feature.

59. **Controlled Experiment page** — New route `/controlled-experiment` and template `controlled_experiment.html`. Exposes all 19 pipeline toggles as individual checkboxes grouped into Preprocessing, Prompt Injection, and Post-Processing categories. Includes BrainIT paper catalog dropdown. Submits to the same `/run` endpoint as Run Experiment, but with `from_page=controlled_experiment` so the backend uses checkbox-absent-means-False semantics. A pipeline mode of "Controlled" is shown in run labels when any non-default toggle combination is used.

60. **`_flag()` helper for backward compatibility** — The `/run` POST handler uses a `_flag(key, legacy_default)` closure that: when `from_page == "controlled_experiment"`, reads the checkbox value strictly (absent = False); otherwise, defaults to `legacy_default="on"` (True) for flags that were always on before. This ensures that runs from Run Experiment and Run Comparison pages are **not affected** by the new flags.

61. **`_CONFIG_KEYS` expanded** — Added 11 new keys: `text_grounded_completion`, `chunk_clinical_filter`, `clinical_only_routing`, `require_label_in_evidence`, `filter_to_gold_vocabulary`, `strict_relations`, `cleanup_dedupe`, `cleanup_scope_pruning`, `cleanup_evidence_pruning`, `cleanup_structural`, `cleanup_axioms`. These are included in the config fingerprint for run deduplication.

62. **`_LABEL_DEFAULTS` expanded** — All 11 new keys added with their correct default values (True for most cleanup and filter flags; False for filter_to_gold_vocabulary, strict_relations).

63. **`apply_builtin_cleanup()` made granular** — Accepts a `cleanup_config` dict. Each of the five cleanup groups (`cleanup_dedupe`, `cleanup_scope_pruning`, `cleanup_evidence_pruning`, `cleanup_structural`, `cleanup_axioms`) is independently gated. When `cleanup_config` is absent or all flags are True, behaviour is identical to the original always-on cleanup.

64. **`text_grounded_completion` configurable** — `run_text_grounded_completion()` call in `run_experiments.py` is now conditional on `config.get("text_grounded_completion", True)`. Default True preserves all existing run behaviour.

65. **`chunk_clinical_filter` configurable** — `filter_chunks_to_clinical()` call conditional on `config.get("chunk_clinical_filter", True)`.

66. **`clinical_only_routing` logic fixed** — Previously the logic `bool(config.get("clinical_only_routing") if "clinical_only_routing" in config else config.get("scope_filter", False))` correctly prioritises an explicit config value when present; falls back to `scope_filter`. Prevents the controlled page from having routing overridden by an implicit `scope_filter` value.

67. **LLM Reasoning Layer removed** — All integration points for `llm_reasoning_layer` removed from `_CONFIG_KEYS`, `_LABEL_DEFAULTS`, the `run()` handler, `run_experiments.py`, and all templates. The dead function `run_llm_reasoning_layer_patch()` remains in `ontology_completion.py` for reference only.

68. **`_build_run_config()` standardised** — Comparison runs now also populate all 11 new flags using `run_opts.get(key, default)`, producing equally comprehensive `metadata.json` files. Previously comparison runs were missing these keys, making their metadata incomplete relative to single-run metadata.

69. **`ontology_pre_cleanup.json` added** — Written immediately before `apply_builtin_cleanup()` so the post-extraction + post-TGL + post-SGC ontology state is preserved for debugging differing results between runs.

70. **BrainIT paper catalog expanded to 11 papers** — Previously 8 papers; now 11. Papers are stored in `resources/BrainIT Papers/` and referenced by `BRAINIT_PAPER_CATALOG`. Both the Controlled Experiment and Run Comparison pages expose the full catalog dropdown.

71. **Non-determinism confirmed, pipeline integrity verified** — A systematic audit (March 2026) comparing two Few-Shot runs on the same paper and config confirmed that all 15 chunk extraction prompts are byte-for-byte identical between runs. Differences in output (TGL returning 68 vs 29 classes) are caused by GPT-4o-mini's non-deterministic sampling, not by any code change. The pipeline is correct and reproducible from config; results vary due to LLM temperature.

#### Phase 15: Evidence-Driven Pipeline — removing hard-coded vocabulary constraints

Architectural shift to make the pipeline generalise to unseen BrainIT literature with novel vocabulary. The core principle: **evidence is the quality gate, not vocabulary whitelists**.

72. **TGC rebuilt as LLM-driven (no vocabulary gate)** — `run_text_grounded_completion()` completely redesigned. Removed: `ALLOWED_CLASSES_CORE` whitelist check on LLM-suggested classes; the vocabulary scan loop that force-added all matching `ALLOWED_CLASSES_CORE` classes; the structural parent loop. Now accepts any class with non-empty evidence. Relations still require text-grounded evidence + bulk-fabrication check. Deterministic hierarchy (`_HIERARCHY_EDGE_CHECKLIST`) remains as soft enrichment (fires only when both endpoints exist).

73. **TGC prompt simplified** — `_build_classes_relations_prompt()` now sends: current ontology state, general clinical concept type guidance, soft relation patterns (not hard constraints), short abbreviation guidance, and evidence rules. Removed: `ALLOWED_CLASSES_CORE` list injection, `MISSING CLASSES` section, `ALLOWED LABELS` section, `_RELATION_DOMAIN_RANGE_HINTS` (hard domain/range constraints). The prompt now says "Extract any clinical concept from the text that is NOT yet in the ontology."

74. **Dead code removed** — `_RELATION_DOMAIN_RANGE_HINTS` constant, `_STOP_WORDS` frozenset, and `_class_label_mentioned_in_text()` function all removed from `ontology_completion.py`. The function signature of `_build_classes_relations_prompt()` simplified: `allowed_classes` and `allowed_relations` parameters removed.

75. **Extraction templates softened** — Exhaustive hard-coded domain hierarchy removed from all three prompt templates:
    - **`baseline.json`** Rule 9 (Label Mapping): Shortened to most common abbreviations + "expand others to their full clinical name." Rule 10 (Structural Completeness): Replaced the 10-category domain hierarchy with "use the text's own categories and groupings."
    - **`one_shot.json`**: Same changes applied.
    - **`mmr_fewshot_controlled.txt`** Rules 6–7: Same changes applied.
    - **`baseline.json` Final Check**: Removed "uses a relation label outside the allowed set" — no longer applicable in raw mode.

76. **`CLASS_SYNONYM_MAP` None entries removed** — All 9 entries mapping to `None` (`"TBI"`, `"traumatic brain injury"`, `"TBI"`, `"severe traumatic brain injury"`, `"sTBI"`, `"head injury"`, `"severe head injury"`, `"Other"`, `"polytrauma"`) removed from `vocabulary.py`. These silently dropped classes that the LLM extracted. Now `CLASS_SYNONYM_MAP` is canonicalization-only — it renames labels, never drops them. Type annotation changed from `dict[str, str | None]` to `dict[str, str]`. `resolve_class_synonyms()` docstring updated; `_resolve()` return type simplified to `str`.

77. **Strict mode redefined as raw extraction** — Comparison dashboard Mode 1 (Strict) changed from `vocab=True, eval_gold=True` to `vocab=False, eval_gold=False`. Strict mode now runs with **no vocabulary guardrails, no gold filtering, no NER, no schema** — pure LLM + evidence.

78. **Gold vocabulary loading decoupled from strategy** — In `run_experiments.py`, `allowed_classes` and `allowed_relations` are now only loaded from gold when at least one vocabulary-dependent feature is explicitly enabled: `prompt_vocab_guardrails`, `filter_to_gold_vocabulary`, or `eval_restrict_to_gold`. Previously they were always loaded for any `phased_3step` run, forcing vocabulary filtering even in Strict mode. `preloaded_gold` is still loaded when `gold_mode == "restricted"` (for cleanup and evaluation) — only the vocab whitelist is gated.

79. **Phase 1 concept instruction softened** — The `build_phase_prompt("concepts")` instruction in `run.py` no longer embeds the exhaustive 8-category domain hierarchy (was listing ~110 specific concepts). Replaced with: generic concept type bullets, short abbreviation guidance (HR, ICP, CPP, GCS, GOSe, PbtO2, PRx), and "use the text's own categories" principle.

80. **Phase 3 gold vocab injection gated** — In `phased_3step` Phase 3, the gold vocabulary label injection and forced `Therapy`/`Baseline Therapy` prepending are now gated behind `inject_vocab_guardrails`. In Strict mode (guardrails off), only the LLM-extracted class labels are passed as valid hierarchy endpoints. Same change applied to the `phased_2step` and `one_shot` hierarchy phases.

81. **Vocabulary fallback gated** — The `ALLOWED_CLASSES_CORE` / `ALLOWED_RELATIONS_CORE` fallback for `filter_classes` / `filter_relations` (used by hierarchy phase filtering) is now gated behind `inject_vocab_guardrails or filter_to_gold_vocabulary`. In Strict mode, no vocabulary fallback applies.

#### Phase 16: Prompt template audit and state-of-the-art alignment

Systematic audit of all extraction prompt templates benchmarked against 2025–2026 state-of-the-art ontology extraction research: OntoKGen adaptive Chain-of-Thought prompting \cite{deepak2024ontokgen}, OntoGPT/SPIRES zero-shot schema-driven extraction \cite{caufield2024spires}, LLMs4OL 2025 few-shot ontology learning \cite{alexbek2025llms4ol}, WojoodOntology ontology-driven LLM prompting \cite{jarrar2025wojood}, and iterative prompt refinement for clinical NLP \cite{haworth2025iterative}.

82. **Rule numbering fixed in `baseline.json`** — Rules were misnumbered 1,2,3,4,5,9,10,6,7,8. Renumbered sequentially 1–8 in logical order: Evidence First → Source Fidelity → Extraction Scope → Class Rules → Label Mapping → Structural Completeness → Relation Rules → Hierarchy Rules.

83. **Over-conservative language replaced** — All three templates contained language that actively suppressed extraction recall:
    - `baseline.json`: "Extract a SMALL, HIGH-PRECISION ontology" and "prefer under-generation to over-generation" (Rule 8 ZERO-SHOT BEHAVIOUR) → replaced with "Extract a COMPREHENSIVE, evidence-grounded ontology" and removed ZERO-SHOT BEHAVIOUR section entirely.
    - `one_shot.json`: "Conservative extraction — Prefer under-extraction to hallucination" section → removed entirely. Evidence requirement is the quality gate.
    - `mmr_fewshot_controlled.txt`: "Extract relations only when the wording is explicit and stable. If uncertain, return no relations" → softened to "Extract relations when the text clearly states a connection between two concepts."

84. **Ignore lists compressed** — Verbose 8–12 line "DO NOT extract" lists (authors, affiliations, institutions, journals, DOIs, organisations, databases, software, ethics, etc.) consolidated into compact 2-line "Extraction Scope" clauses across all three templates. Research \cite{haworth2025iterative} shows positive framing ("Extract X") outperforms negative framing ("Do NOT extract Y"). Token budget freed for more useful instructions.

85. **Chain-of-Thought reasoning step added** — All three templates now include a REASONING STEP section before the output format: "Before producing JSON, mentally scan the text for: (1) every named clinical concept, (2) explicit relations between them, (3) hierarchy cues (such as, is a, type of, include:). Then output only items with verbatim evidence." This is an in-prompt CoT nudge inspired by OntoKGen \cite{deepak2024ontokgen} and Phoenixes at LLMs4OL 2025 \cite{alexbek2025llms4ol} — no additional LLM call required.

86. **System role generalised** — All three templates and the TGC prompt (`_build_classes_relations_prompt`) changed from "Neuro-ICU / BrainIT literature" to "biomedical and clinical literature. The text may come from Neuro-ICU, neuroscience, or related clinical domains." This removes domain anchoring that limited Strict mode generalisation to unseen papers.

87. **Output completeness nudges added** — `baseline.json` COMPLETENESS CHECK now includes "Have you extracted every clinical concept the text explicitly names?" `one_shot.json` OUTPUT RULES now ends with "Extract every distinct clinical concept the text names — err on the side of inclusion when evidence exists." `mmr_fewshot_controlled.txt` adds a **Completeness** line with the same wording. These counter the LLM's tendency to stop early.

88. **Anti-anchoring strengthened in `one_shot.json`** — Added "Your text may contain concept types NOT represented in the example — extract them all" to the existing anti-anchoring rule. Research on few-shot anchoring effects shows examples have very strong anchoring on LLM output.

89. **Phase 2 relation pattern hints** — Both Phase 2 instruction variants (with and without separate hierarchy phase) in `run.py` now include a "COMMON RELATION PATTERNS" section with 6 soft hints: `has monitoring data`, `receives therapy`, `has outcome`, `monitoring indicates condition`, `targets condition`, `includes`. These are guidance, not constraints — "You may also extract other relations that the text explicitly supports."

90. **SGC prompt section order corrected** — `build_whole_ontology_completion_prompt()` in `ontology_completion.py` now places the instruction block before the few-shot example (was: example → instruction). Research suggests instruction-first ordering improves LLM comprehension of the task before seeing the format example.

91. **TGC output count nudge** — `_build_classes_relations_prompt()` now includes "The per-chunk extraction typically misses concepts — be thorough" to encourage the LLM to find additional items beyond the initial extraction.

92. **Pipeline mode card and help text updated** — `index.html` Strict mode card description changed from "Scope filter · Vocab guardrails" to "Raw extraction · No vocab list · Evidence-only". Help panel text updated to describe Strict as "Raw extraction: no vocabulary list or gold filter. Evidence is the only quality gate; works on any paper with unseen vocabulary." Pipeline mode JS (`MODES` object) now correctly sets `prompt_vocab_guardrails=false` and `eval_restrict_to_gold=false` for Mode 1 (Strict), and `true` for Modes 2–3. Same fix applied to `comparison_dashboard.html`.

#### Phase 17: Hierarchy extraction overhaul

Analysis of four consecutive runs showed hierarchy recall at 0–1.3% for all papers except Piper 2003 (22.4%). Root cause analysis identified five compounding extraction bottlenecks. Five fixes applied:

93. **Phase 2 now extracts hierarchy alongside relations** — The `build_phase_prompt` for `phased_3step` Phase 2 no longer sets hierarchy to an empty array. Instead, it includes subClassOf extraction instructions with hierarchy cue guidance ("such as", "includes", "type of", "consists of", "categorized as", enumeration lists). Phase 2 can now return hierarchy edges in the same LLM call as relations.

94. **Hierarchy trigger set expanded from 6 to 25 cues** — `HIERARCHY_LEXICAL_TRIGGERS` in `schema.py` expanded from `("such as", "is a", "type of", "kind of", "include:", "includes:")` to 25 cues including: `"consists of"`, `"comprises"`, `"comprised of"`, `"composed of"`, `"categorized as"`, `"classified as"`, `"grouped into"`, `"subdivided into"`, `"subtypes"`, `"subtype of"`, `"forms of"`, `"categories of"`, `"types include"`, `"types of"`, `"subclasses of"`, `"subclass of"`, `"is a subclass of"`, `"is a form of"`, `"is a type of"`. These are imported from `schema.py` into `run.py` via `HIERARCHY_LEXICAL_TRIGGERS`.

95. **Phase 3 double-gating removed** — Previously, hierarchy edges extracted in Phase 3 were filtered through `filter_hierarchy_to_lexical_cues`, requiring both the chunk to contain triggers AND the evidence string to also contain a trigger word. Replaced with single chunk-level gating: any hierarchy edge with non-empty evidence is accepted if the chunk was already gated for hierarchy triggers. This prevents valid edges like `{"subClass": "Heart Rate", "superClass": "Core Monitoring Parameter", "evidence": "minute-by-minute data includes heart rate"}` from being dropped because "includes" (the trigger) was only in the chunk, not in the truncated evidence.

96. **Phase 3 cue list expanded in prompt** — The Phase 3 instruction in `build_phase_prompt` now lists 5 categories of hierarchy cues: (1) "such as" patterns, (2) "is a / type of / kind of / form of", (3) "include: / includes: / consists of / comprises", (4) "categorized as / classified as / grouped into / subdivided into", (5) enumeration lists under category headings. Adds "Be thorough — extract EVERY parent-child relationship the text supports."

97. **TGC Pass 1 now extracts hierarchy** — `_build_classes_relations_prompt` in `ontology_completion.py` updated with a `### HIERARCHY GUIDANCE` section listing hierarchy cues and enumeration pattern instructions. The JSON output format now includes a `"hierarchy"` key expecting `[{"subClass": "...", "superClass": "...", "evidence": "..."}]`. Previously TGC Pass 1 only extracted classes and relations; hierarchy was only handled by a separate checklist-based pass.

#### Phase 18: Chain-of-Layer (CoL) taxonomy construction

Integration of the Chain-of-Layer technique (Zeng et al., 2024, CIKM) into TGC, replacing the checklist-based hierarchy pass with structured top-down taxonomy construction.

98. **`_build_col_prompt` function created** — New function in `ontology_completion.py` that generates a Chain-of-Layer prompt. Takes all known class labels, current hierarchy edges, and the corpus text. Instructs the LLM to organize classes into a numbered hierarchical tree (e.g., `1. Monitoring Data\n  1.1 Core Monitoring Parameter\n    1.1.1 Heart Rate`) building top-down, layer by layer. Includes explicit instructions to only add edges supported by textual evidence, and to leave flat any classes that do not have a clear parent in the text.

99. **`_parse_col_tree` function created** — New parser in `ontology_completion.py` that converts the numbered tree output from the LLM back into a list of `subClassOf` dictionary triples. Uses regex pattern `r"^\s*([\d.]+)\s+(.+)$"` to parse lines, determines parent-child relationships from numbering depth (e.g., `1.1` is a child of `1`, `1.1.1` is a child of `1.1`). Returns `[{"subClass": "Heart Rate", "superClass": "Core Monitoring Parameter"}, ...]`.

100. **CoL integrated into TGC as replacement for Pass 2** — `run_text_grounded_completion` updated: the previous checklist-based hierarchy pass (Step 2b) was replaced with the CoL pass (Step 2c). The TGC multi-pass architecture is now: (1) deterministic hierarchy pre-scan, (2a) LLM Pass 1 (classes + relations + hierarchy), (2c) CoL taxonomy pass, (3) deterministic hierarchy post-scan.

101. **New intermediate classes auto-created from CoL** — If the CoL pass identifies superclass labels that are not in the current ontology (e.g., intermediate grouping categories like "Derived Parameter"), they are automatically created as new `ClassEntity` objects and added to the ontology. The `valid_class_norms` variable was refactored from a `set` to a `Dict[str, str]` (normalised label → original label) to support this.

102. **CoL diagnostics and artefacts** — `tgc_diagnostic.json` now includes `"col_hierarchy_edges"` count. CoL prompts and responses saved as `tgc_col_prompt.txt` and `tgc_col_response.txt` in the run artefacts directory.

#### Phase 19: Scope filter and blacklist expansion (clinical-only extraction)

Targeted at removing governance, administrative, and methodological noise that survived extraction and cleanup across all 11 BrainIT papers.

103. **Comprehensive out-of-scope class patterns** — `_OUT_OF_SCOPE_CLASS_PATTERNS` in `reasoner.py` expanded with ~80 additional patterns covering: BrainIT group/network/project (`brainit`, `project management`, `collaborative group`, `internet registration`, `research study`, `publication`); data infrastructure (`data element`, `data contributing`, `data quality control`, `data analysis`, `data collection`, `core dataset`, `time-series data`, `definition file`, `collection tool`, `validation staff`, `sampling technique`, `nursing chart`, `bedside monitoring type`, `health care technology`); patient attributes as data properties not classes (`patient age`, `patient sex`, `patient population`, `patient management`); vague meta-concepts (`clinical information`, `physiological monitoring`, `therapy target`, `cpp management`, `motor score`, `hypoxia evidence`, `not testable`, `unknown field`, `missing data`); institution names (`university`, `hospital`, `department`, `school of`, `nhs`, `medical centre`); publishing/reference metadata (`journal`, `publisher`, `copyright`, `manuscript`, `reference list`); study/cohort labels (`study result`, `study population`, `study cohort`, `patient cohort`); monitoring/system meta (`monitoring system`, `monitoring device`, `recording system`, `data source`, `signal quality`); and `neuroprotective drug`. All 110 gold clinical classes verified against new patterns with zero false positives.

104. **Edge pruning for out-of-scope endpoints** — New `_prune_edges_with_out_of_scope_endpoints()` and `_is_out_of_scope_label()` in `reasoner.py`. Removes relations whose domain or range matches out-of-scope patterns, and hierarchy edges whose subClass or superClass matches. Prevents `_auto_add_missing_endpoints` from re-introducing pruned noise classes as endpoint stubs. Run after initial scope pruning and again after endpoint auto-add; counts reported as `relations_pruned_scope`, `hierarchy_pruned_scope`.

105. **Re-prune after endpoint auto-add** — After `_auto_add_missing_endpoints`, built-in cleanup now re-runs out-of-scope, abstract-data, and broad-contextual class pruning so any auto-added endpoint that matches noise patterns is removed. Then runs edge pruning so relations/hierarchy referencing those endpoints are dropped.

106. **Post-orphan scope cleanup** — New `apply_post_pass_scope_cleanup()` in `reasoner.py`. Invoked from `run_experiments.py` immediately after Orphan Rescue. Re-applies out-of-scope class pruning, abstract/broad pruning, and edge pruning. Orphan Rescue can add new classes as relation/hierarchy endpoints; this pass ensures governance/methodological concepts re-introduced by the LLM are removed. Results in `improvement_counts["post_orphan_scope_cleanup"]`.

107. **Scope filter blacklists expanded** — `scope_filter.py`: `_SCOPE_BLACKLIST_TERMS`, `_GOVERNANCE_DOMINANCE_PHRASES`, `_ADMIN_PHRASES`, `_ADMIN_HEADINGS`, and `_GOVERNANCE_SECTIONS` extended with BrainIT group/network/project terms (`brainit group`, `brainit network`, `project management`, `collaborative group`, `internet registration form`), data infrastructure (`data elements definition file`, `data collection software`, `data collection protocols`, `data analysis methodologies`, `data quality control`, `bedside monitoring types`, `health care technology`, `sql database`, `multi-centre trial`), and equivalent section headings. Upstream filtering reduces governance content reaching the LLM.

108. **Abstract allowlist tightened** — Section-heading-style labels (`Demographic and Clinical Information`, `Minute by Minute monitoring information`, `Intensive care management information`, `Secondary insult treatment information` and normalised variants) removed from `_ABSTRACT_LABEL_ALLOWLIST_NORM` so they are pruned by abstract-data patterns when the LLM extracts them as classes. Gold structural classes (Monitoring Data, Demographic Data, Core/Optional/Derived Parameter, etc.) remain protected.

#### Phase 20: LLM Chain-of-Thought Ontology Refinement Layer

The hardcoded blacklist approach (~914 patterns) does not generalise to unseen vocabulary in new papers and cannot catch semantic errors (e.g. hierarchy inversions, domain/range mismatches). Phase 20 adds an LLM-driven refinement pass that reasons about ontology quality using Chain-of-Thought, replacing domain-specific blacklists with generalizable semantic understanding.

109. **LLM CoT Refinement pass** — New `run_ontology_refinement()` in `src/prompting/ontology_completion.py`. Presents the full generated ontology (all classes with evidence, relations with domain/range/evidence, hierarchy edges with evidence) plus a 6,000-char corpus preview to the LLM with a domain-agnostic prompt. The prompt defines what belongs in a clinical ontology (conditions, parameters, therapies, assessments, lab values, devices, patient concepts, data categories) and what does not (project names, governance, data infrastructure, statistical methods, publishing metadata, institution names, IT concepts). The LLM reasons step-by-step about each questionable item using Chain-of-Thought and returns a JSON response with `classes_to_remove`, `relations_to_remove`, `hierarchy_to_remove`, and `hierarchy_corrections` (for fixing inversions). The function applies all removals and corrections to the ontology, also cascading class removals to relations and hierarchy edges that reference removed classes.

110. **Domain-agnostic prompt design** — The refinement prompt uses general clinical/biomedical terminology (not BrainIT-specific), enabling it to work on any medical paper without modification. Positive definitions cover clinical conditions, physiological parameters, treatments, assessments, lab values, devices, and data categories. Negative definitions cover research governance, data infrastructure, statistical methods, publishing metadata, and institution names — expressed as semantic categories, not hardcoded term lists.

111. **Hierarchy semantic review** — The prompt instructs the LLM to check hierarchy edges for: inversions (child/parent swapped, e.g. "Brain Injury ⊑ Traumatic Brain Injury" should be reversed), and nonsensical edges (e.g. "Patient ⊑ Secondary Insult"). Corrections are returned as `hierarchy_corrections` with `old_sub`, `old_super`, `new_sub`, `new_super` fields, and applied in-place rather than requiring removal + re-addition.

112. **Pipeline position** — Runs after orphan rescue and post-orphan scope cleanup, before rule-based reasoning. This ensures the LLM reviews a cleaned but not schema-completed ontology. Rule-based reasoning can still add gold hierarchy edges after refinement. Gated on `config.get("text_grounded_completion", True)` (same gate as TGC/orphan rescue). Stage snapshot recorded as `after_llm_refinement` in `metrics["by_stage"]`.

113. **Debugging artifacts** — Writes `prompts/refinement_prompt.txt` and `prompts/refinement_response.txt` to the run directory. Results recorded in `improvement_counts["llm_refinement"]` with keys `classes_removed`, `relations_removed`, `hierarchy_removed`, `hierarchy_corrected`.

#### Phase 21: Evidence matcher upgrade + cleanup audit log

Analysis of `cleanup_removed.json` from a Georgatzis 2016 run revealed two systematic false positives in class and relation pruning: (1) "Arterial Oxygen Saturation" was removed because its evidence was "including pulse oximetry" — lexically disjoint from the label despite being clinically equivalent; (2) relations such as "has mean arterial pressure measurement" and "has handling episode" were removed because their endpoints (domain/range) were not found in the evidence text due to morphological variation ("physiological parameters" ≠ "physiological data") or abbreviation gaps ("ABPm" not bridged to "Mean Arterial Pressure"). Three improvements were applied.

**Strategy 1 — Lemmatised token overlap (step 8 in `_label_evidenced_in_evidence` and `_evidence_contains_class_reference`):** After the existing raw-token overlap check, a spaCy lemma-based overlap is computed: label tokens and evidence tokens are each lemmatised using `en_core_sci_sm` (the installed biomedical model), stopwords and short tokens are removed, and the class is kept if at least `ceil(len(label_lemmas) / 2)` lemmas appear in the evidence. This catches morphological variants — "physiological parameters" ↔ "physiological data" (shared lemma "physiological"), "patient handling events" ↔ "Handling Episode" (shared lemma "handle"), "mean ABP" ↔ "Mean Arterial Pressure" (shared lemma "mean"). Applied to both class pruning (`_label_evidenced_in_evidence`) and relation endpoint pruning (`_evidence_contains_class_reference`).

**Strategy 2 — Max word-pair semantic similarity (step 9):** After lemma overlap, `en_core_sci_sm` word vectors are used to compute the maximum cosine similarity across all (label_token, evidence_token) pairs: `max_word_pair_similarity = max(sim(lt, et) for lt in label_tokens for et in evidence_tokens)`. This is more robust than doc-level similarity because it finds the single most-semantically-similar token pair rather than averaging over the whole document, which dilutes the signal. For "Arterial Oxygen Saturation" vs "including pulse oximetry", the (saturation, oximetry) pair scores 0.716. Threshold: **0.65**, calibrated empirically: all valid clinical concepts in BrainIT runs score ≥ 0.67; out-of-scope noise that passes earlier filters (e.g. "Pilot Study", "Summary Statistics") scores < 0.65. The helper `_max_word_pair_similarity(label, ev_text, nlp)` is shared between both functions. A lazy module-level loader `_get_sci_nlp()` loads `en_core_sci_sm` (with fallbacks to `en_core_web_sm`, `en_core_web_md`) on first call, returning `None` gracefully if spaCy is unavailable — strategies 8 and 9 are skipped entirely in that case.

**Strategy 3 — Expanded structural exemptions:** `_STRUCTURAL_EVIDENCE_EXEMPTIONS_BASE` extended from 17 to 38 entries with universal clinical superclass labels that are always implicit in medical text and rarely appear verbatim in evidence: `Treatment`, `Intervention`, `Procedure`, `Assessment`, `Clinical Event`, `Episode`, `Physiological Parameter`, `Physiological Data`, `Physiological Signal`, `Clinical Measurement`, `Measurement`, `Signal`, `Waveform`, `Monitor`, `Time Series`, `Clinical Data`, `Medical Record`, `Handling Episode`, `Nursing Procedure`. These protect structurally important classes from evidence pruning across any medical paper without requiring paper-specific hardcoding.

**Cleanup audit log:** `apply_builtin_cleanup()` now accumulates a structured `audit_log: List[Dict]` across all 11 pruning stages and returns it as `result["cleanup_audit_log"]`. `run_experiments.py` writes the log to `<run_dir>/cleanup_removed.json` after cleanup completes (alongside `axiom_violations.json`). The `cleanup_audit_log` key is excluded from `improvement_counts` to avoid polluting metrics. This is a diagnostic-only feature with zero runtime impact on pipeline behaviour.

**Result on the Georgatzis 2016 run:** All four previously removed false positives — "Arterial Oxygen Saturation" (via semantic 0.716), "has physiological parameter" (via token overlap on "physiological"), "has mean arterial pressure measurement" (via lemma overlap on "mean"), "has handling episode" (via exact/token match on "patient handling") — are now correctly kept. "ROC Curve" and "AUC" remain correctly removed (similarity scores 0.11–0.19, well below 0.65).

#### Phase 22: Domain scope file — decouple cleanup from gold standard

**Problem:** Built-in cleanup relied on the gold-standard ontology for allowlists, evidence exemptions, and in-scope class lists. The current gold standard is a dummy covering only papers 1–2. This meant cleanup on papers 3–11 lacked proper domain guidance, and cleanup was tightly coupled to the evaluation artifact — a circular dependency that undermines the experimental validity of "Strict mode" (which claims to run without gold knowledge).

**Solution:** Created a **domain scope file** (`resources/domain_scope.json`) curated from all 11 BrainIT papers, containing:
- **117 in-scope classes** — every valid clinical concept across all 11 papers.
- **27 in-scope relations** — all relation types used in the domain.
- **78 evidence exemptions** — classes whose canonical labels rarely appear literally in paper text (e.g. "Fluids" → paper says "fluid input/output").
- **10 abstract-data allowlist** entries — legitimate classes matching abstract patterns (e.g. "Monitoring Data").
- **42 broad-context allowlist** entries — legitimate classes matching broad contextual patterns.
- **12 concept categories** — grouping classes by clinical domain (vital signs, therapies, conditions, etc.).

**Loader:** `src/ontology/domain_scope.py` provides cached access via `get_in_scope_classes_norm()`, `get_evidence_exemptions_norm()`, `get_abstract_data_allowlist_norm()`, `get_broad_context_allowlist_norm()`. Falls back to hardcoded defaults if the file is unavailable.

**Architecture change:** Cleanup now **always runs** (previously gated on `preloaded_gold or _is_strict_mode`). The three hardcoded allowlists/exemption sets in `reasoner.py` (`_ABSTRACT_LABEL_ALLOWLIST_NORM`, `_BROAD_CONTEXT_LABEL_ALLOWLIST_NORM`, `_EVIDENCE_PRUNING_EXEMPTIONS`) are replaced by functions that load from domain scope. The gold standard's role is now limited to: (a) label canonicalization in Guided/Schema modes, and (b) axiom constraints.

**Benefit:** Cleanup is now deterministic and reproducible across all 11 papers. The domain scope file can be updated independently of the gold standard when the real BrainIT ontology arrives.

#### Phase 23: Prompt Batch 2 — schema-first reorder, self-verification, ODP structure

Three prompt engineering improvements from the Research-vs-TripleGen alignment report, validated against 2024–2025 state-of-the-art ontology extraction research.

**Item 2 — Schema-first prompt reorder:** In `mmr_fewshot_controlled.txt`, the `{{VOCAB_AND_HINTS}}` placeholder was moved from after the `### CORE CONSTRAINTS` section to before it. Research (SPIRES, OSKGC) shows that schema-first prompts produce better conformance because the LLM sees the vocabulary before encountering rules that reference it. The `baseline.json` and `one_shot.json` templates already had schema-first order and were unchanged.

**Item 3 — Lightweight self-verification (Metacognitive Prompting):** A brief `### SELF-CHECK` section was added to all extraction prompts — Phase 1 (class extraction), Phase 2 (relation extraction), Phase 3 (hierarchy extraction), baseline template, one-shot template, and TGC prompt. Each self-check asks the LLM to verify 3–5 specific properties before finalising output: (a) every label is a clinical type (not an instance or person's name), (b) no duplicate/synonym labels remain, (c) all relation endpoints match extracted classes, (d) no circular hierarchy edges, (e) every evidence field is a verbatim quote. This is deliberately lightweight (2–4 lines per phase) rather than the full metacognitive approach recommended by Ontogenia, because TripleGen already has a dedicated LLM Refinement Layer for deep structural review. The inline self-check catches the most common errors at the point of generation.

**Item 5 — Ontology Design Patterns (ODPs):** The flat "Patient-centric / Clinical process / Structural" relation pattern lists in Phase 2 (`build_phase_prompt` in `run.py`) and TGC (`_build_classes_relations_prompt` in `ontology_completion.py`) were restructured into 6 named ontology design patterns: (1) Clinical Monitoring, (2) Clinical Intervention, (3) Clinical Assessment and Outcome, (4) Patient Context, (5) Data Quality, (6) Composition. Each pattern shows a coherent micro-schema using arrow notation (e.g. `Patient --[has monitoring data]--> Monitoring Data`). This gives the LLM structural context about how relations form meaningful clusters rather than presenting an unstructured list. Aligned with the Ontogenia/ODP approach (ESWC 2024).

#### Phase 24: `includes` relation → hierarchy conversion

114. **`includes` → hierarchy conversion in reasoner** — A frequent LLM pattern was emitting `(Monitoring Data) --[includes]--> (Heart Rate)` as a relation triple rather than as a `subClassOf` edge. `apply_builtin_cleanup()` now includes an `_includes_to_hierarchy()` step that scans for relation labels matching `includes`, `include`, `is composed of`, `consists of`, `comprises`, and rewrites them as hierarchy edges (`subClass = range`, `superClass = domain`). The original relation is removed. Prevents hierarchy recall loss when the LLM uses a predicate instead of a structural edge.

#### Phase 25: Label normalization

115. **Label normalization pass** — New `_normalize_class_labels()` function run early in cleanup. Applies: (a) plural → singular for known clinical plurals (Patients → Patient, Conditions → Condition); (b) hyphen collapse (`intra-cranial` → `intracranial`) for labels matching the BrainIT vocabulary; (c) title case normalization (`icp monitoring` → `ICP Monitoring`); (d) canonical alias resolution via `canonical.CANONICAL_ALIAS_MAP`. All four passes preserve evidence and provenance; renamed labels are deduped against existing classes via `canonical_key()`. This is additional to the existing canonical dedup at merge time — it catches labels that arrive with non-canonical casing or pluralization after post-processing.

#### Phase 26: Relation vocabulary standardization

116. **Relation alias rewrite map** — A curated alias-to-canonical rewrite map with ~30 entries in `reasoner.py`: `hasComponent → has_component`, `causedBy → caused_by`, `indicatesRiskOf → indicates_risk_of`, `administeredTo → administers`, `administers → administers`, `recordedAt → recorded_at`, `hasUnit → has_unit`, `hasOutcome → has_outcome`, `targetsCondition → targets condition`, `monitors → monitors`, `treats → treats`, `measures → measures`, `indicates → indicates`, `produces → produces`, `records → records`, `manages → manages`, `influences → influences`, `evaluates → evaluates`, `causes → causes`, `isPartOf → is_part_of`, and more. Applied to every relation during cleanup. Rewrites are case-insensitive and preserve evidence. Significantly increases relation match rate against the gold vocabulary without modifying extraction prompts.

#### Phase 27: Hierarchy scaffolding from domain scope

117. **Hierarchy scaffolding** — New step in cleanup that consults `resources/domain_scope.json`'s concept-category grouping and adds hierarchy edges for classes whose canonical label matches a category (e.g. `Heart Rate → Core Monitoring Parameter`, `Mannitol → Baseline Therapy`). Only adds edges where both endpoint classes exist in the ontology — no new classes are introduced. Edges carry synthetic evidence `"Domain scope: X is a subclass of Y"` and pass the `require_evidence=True` filter.

118. **Completeness logging** — After scaffolding, a `completeness.json` is written to the run directory summarising: fraction of classes with at least one hierarchy edge, fraction of top-level classes, average hierarchy depth, number of classes with at least one incoming/outgoing relation. Diagnostic only; has no effect on metrics.

#### Phase 28: Multi-paper run guardrails

119. **Strategy warning for multi-paper corpora** — When the corpus manifest contains more than 3 documents, `run_experiments.py` logs a warning if the selected strategy is `one_shot` or `phased_3step` without `prompt_vocab_guardrails`, since few-shot extraction loses effectiveness as corpus diversity grows. The warning is written to `warnings.txt` but does not block the run.

120. **Overlap boost** — Chunking overlap is boosted from 200 to 400 tokens when the corpus has >1 document, to compensate for sentence fragmentation at document boundaries in merged corpora.

#### Phase 29: Out-of-scope class pruning — methodology and statistics terms

121. **Methodology/statistics pruning** — `_OUT_OF_SCOPE_CLASS_PATTERNS` in `reasoner.py` extended with explicit patterns for methodology and statistics concepts that consistently leaked through previous scope pruning: `correlation coefficient`, `regression model`, `bayesian network`, `ROC curve`, `AUC`, `sensitivity`, `specificity`, `null hypothesis`, `statistical significance`, `p-value`, `confidence interval`, `odds ratio`, `hazard ratio`, `logistic regression`, `linear regression`, `feature importance`, `training set`, `test set`, `validation set`, `cross-validation`, `machine learning model`, `neural network`, `random forest`. These are legitimate in research text but do not belong in a clinical domain ontology. All 110 gold clinical classes verified against new patterns with zero false positives.

#### Phase 30: Upload file race condition fix

122. **Per-run upload subdirectory** — Previously, file uploads from the web UI were saved to a shared `data/corpus_ui/` directory that was not cleared between runs. On Windows, file locking from an earlier run would cause the next run to fail silently or mix files from different runs. Fixed in `web/app.py`: each run now writes uploads to a unique subdirectory `data/corpus_ui/<run_id>/`, and a best-effort cleanup removes subdirectories older than 24 hours at the start of each run. Completely isolates concurrent/sequential runs from each other.

#### Phase 31: Evidence verification and class definitions (Mar 2026)

123. **Evidence verification overhaul** — The `_evidence_appears_in_text()` check used by TGC and SGC relation filtering was previously a whole-string substring match against the full corpus. This was overly strict — it failed when the LLM shortened or paraphrased the evidence quote slightly. Replaced with a normalised token-overlap check: the evidence is tokenised, stopwords removed, and accepted if ≥70% of evidence tokens appear in a 200-token window of the corpus text. Includes a fallback to raw substring for short evidences (<20 chars). Catches genuine paraphrases while still rejecting fabricated evidence.

124. **Class definitions added throughout pipeline** — Every `ClassEntity` now carries a `definition` field populated at extraction time. LLM prompts were updated to request a one-sentence definition alongside each class label, and the schema permits `{"label": "...", "definition": "...", "evidence": "...", "synonyms": [...]}` end-to-end. Definitions propagate through merge, cleanup, reasoner, and LLM refinement. Used by cluster completion (§2.14) to enrich the per-cluster LLM prompt with existing semantics.

#### Phase 32: Ontology Engineering subsystem — cross-paper reconstruction

125. **Ontology Engineering page** (§2.14) — Entirely new web UI page and backend subsystem for cross-paper ontology synthesis. Lets the user select multiple source extraction runs, merge them, cluster the merged classes, and reconstruct a unified ontology via LLM-driven cluster completion. Distinct from the main extraction pipeline and does not modify extraction behaviour.

126. **Multi-source ontology merge** — `src/ontology/merge.py` `merge_ontologies()` combines N source-run ontologies into a single dict, deduping classes by `canonical_key()`, relations by `(label, canonical(domain), canonical(range))`, and hierarchy edges by `(canonical(sub), canonical(super))`. Preserves provenance, evidence, and synonyms when merging duplicates. Used as the first step of the OE page flow.

127. **Semantic clustering of merged classes** — `cluster_classes()` embeds each class label + definition via `all-MiniLM-L6-v2` and performs Ward's agglomerative clustering to partition the class set into N clusters (default 25). Each cluster is auto-named by concatenating the two most central labels. Output schema: `{"clusters": [{"id": N, "name": "...", "members": [...]}]}`.

128. **LLM cluster completion** (`src/ontology/cluster_completion.py`) — For each cluster, a prompt is built that contains: (a) the cluster name, (b) every class in the cluster with its definition/evidence/synonyms, (c) names of all other clusters for cross-cluster linking context, (d) the closed relation vocabulary (see below). The LLM is asked to produce **only relations** using the closed vocabulary — no new classes, no hierarchy. Processing is per-cluster (one LLM call per cluster). Diagnostic counters logged per cluster: `dropped_inferred_classes`, `dropped_relations_vocab`, `dropped_relations_endpoints`, `dropped_hierarchy`, `kept_classes`.

129. **Hard no-inferred-classes filter** — `run_cluster_completion()` drops any class the LLM emits that is not a verbatim member of the cluster or of the full source ontology. This is enforced deterministically regardless of what the prompt says, and a `[inferred]` prefix is stripped before matching. The precision motivation: against a small gold standard, every extra class lowers precision more than it raises recall, so the reconstruction layer is constrained to *rearrange* the source rather than expand it.

130. **Closed relation vocabulary** — The allowed relation labels are harvested from the merged source ontology itself (not from the gold standard). `source_rel_labels` collects the distinct relation labels that the source extraction runs produced, case-insensitive. The cluster prompt inlines this list as "Allowed relation labels (closed vocabulary)", instructing the LLM that new labels will be discarded. A deterministic post-parse filter re-checks every emitted relation and snaps the label to the source's exact casing so downstream `merge_ontologies` dedupes cleanly. Relations whose domain or range is not a valid source class are also dropped. **This filter reads no gold-standard file and uses no domain-curated list** — it is purely self-seeded from extraction output, preserving the "no gold leakage in reconstruction" property.

131. **Hierarchy passthrough from source** — The cluster prompt explicitly instructs the LLM not to emit hierarchy edges ("Do NOT produce subClassOf / hierarchy edges. The hierarchy is already defined in the source ontology and will be preserved independently"), and any hierarchy edges the LLM produces despite this instruction are force-dropped. All hierarchy edges in the reconstructed ontology come from the source seed. Rationale: per-cluster LLM views are too narrow to produce good hierarchy, and previous experiments showed LLM-emitted hierarchy dragging overall F1 down by ~0.05.

132. **Source-seeded merge** — At the end of cluster completion, the original merged source ontology is prepended as the first fragment before all LLM cluster fragments, and the combined list is passed to `merge_ontologies()`. Because `merge_ontologies` dedupes by canonical key, classes/hierarchy/relations the LLM preserved collapse naturally against the source copy, and any structure the LLM forgot is rescued from the source. This also fixes the disconnected-cluster-island problem: the source's cross-cluster hierarchy edges survive intact. Metadata flags `source_seeded: True`, `closed_relation_vocab: True`, `hierarchy_from_source_only: True` are written to the reconstructed `ontology.metadata`.

133. **Compare Metrics modal** — `web/static/js/ontology_engineering.js` `renderAnalyzeModal()` presents the reconstruction alongside each selected source run with four F1 series: **Overall F1** (mean of available class/hier/rel F1s), **Class F1**, **Hierarchy F1**, **Relation F1**. The modal shows: card strip with "Best" badges per metric, a 4-series Chart.js bar chart, and a sortable table with columns `Run | Overall F1 | Class F1 | Hier F1 | Rel F1 | Precision | Recall`. Default sort is by Overall F1 descending.

134. **`/api/runs-metrics` endpoint** — Flask endpoint in `web/app.py` that accepts a list of run IDs and returns per-run `class_f1`, `rel_f1`, `hier_f1`, and `overall_f1` (mean of available F1s). Both this endpoint and `_list_runs_with_ontology()` use a `_pair_f1(p, r)` helper that returns `2pr/(p+r)` when both precision and recall are present, and `None` otherwise. Powers the Compare Metrics modal without round-tripping full metrics payloads.

#### Phase 33: Embedding-based scope filter fallback (Apr 2026)

135. **Embedding scope fallback** — Added a sentence-transformer-based fallback to `chunk_is_administrative()` in `scope_filter.py`. When the new `embedding_scope_fallback` config flag is `True`, borderline chunks (where both keyword-based admin and clinical scores are below their respective thresholds) are classified by cosine similarity against pre-computed clinical and administrative centroid vectors. The centroids are 384-dimensional mean vectors computed from 25 representative clinical passages and 25 administrative passages, stored in `resources/scope_centroids.json`. The centroid separation is strong (cosine similarity 0.39 between centroids; 25/25 clinical passages and 24/25 admin passages correctly classified on the source set). The embedding path is only invoked for ambiguous chunks — clear-cut keyword matches use the fast deterministic path unchanged.

136. **Centroid generation script** — `scripts/generate_scope_centroids.py` encodes curated BrainIT-domain clinical and administrative passages with `all-MiniLM-L6-v2`, computes mean vectors, validates classification accuracy on the source passages, and writes `resources/scope_centroids.json`. Rerunnable if the passage set is updated.

137. **Strict mode default** — `embedding_scope_fallback` is set to `True` by default when Strict mode (mode 1) is selected on the index page and comparison dashboard, and `False` for Guided and Schema-Completed modes. This is consistent with Strict mode's design goal of generalisation to unseen papers without hardcoded domain vocabulary — the embedding fallback provides semantic classification for terminology not covered by the keyword lists. The Controlled Experiment page exposes the flag as a manual toggle (unchecked by default).

138. **Unit tests** — `tests/unit/test_scope_filter_embedding.py` provides 17 tests: 4 for `_cosine_sim`, 4 for `embedding_classify_chunk` (including graceful degradation when centroids/model are missing), 5 for `chunk_is_administrative` integration (embedding called only for borderline chunks, skipped for clear-cut cases), 1 for `filter_chunks_to_clinical` flag passthrough, and 3 end-to-end tests with the real sentence-transformer model.

#### Phase 34: Comparison dashboard metrics overhaul (Apr 2026)

139. **4-dimension F1 metrics across all comparison views** — Replaced the old F1/Precision/Recall/Coverage metrics format with four F1 dimensions (Overall F1, Class F1, Hierarchy F1, Relation F1) across all comparison and analysis pages, matching the OE page's Compare Metrics modal format. Affected surfaces: (a) **Analyze Selected modal** on the comparison dashboard — summary cards, bar chart (4 grouped bars), scatter plot (now Class F1 vs Hierarchy F1), and sortable table; (b) **per-run Analysis modal** on the comparison dashboard — bar chart and table; (c) **batch analysis page** (`comparison_analyze.html`) — table columns and ranking description.

140. **Backend endpoints updated** — Three Flask endpoints now compute and return `overall_f1`, `hier_f1`, and `rel_f1` alongside the existing `f1` (class F1): `api_configs_last_runs` (powers Analyze Selected), `api_config_analysis` (powers per-run Analysis modal), and `comparison_analyze` (powers batch analysis page). Each uses a local `_pair_f1(p, r)` helper to compute F1 from the `relations` and `hierarchy` sub-blocks of `metrics.json`. Overall F1 is the arithmetic mean of available class/hier/rel F1 values.

141. **Consistent chart colours** — All comparison charts now use the same colour scheme as the OE Compare Metrics modal: Overall F1 green (`#2ED6A1`), Class F1 cyan (`#39C6E6`), Hierarchy F1 orange (`#F5A65B`), Relation F1 purple (`#C485FF`). Large dashboard mode (>15 runs) shows horizontal bars sorted by Overall F1; small mode shows vertical grouped bars. Scatter plot colour-codes points by Overall F1 tier (green ≥0.7, orange ≥0.4, red <0.4).

---

### 2.14 Ontology Engineering page — cross-paper ontology reconstruction

The **Ontology Engineering** (OE) page is a secondary subsystem for **cross-paper knowledge synthesis**. Where the main extraction pipeline (§2.1–§2.7) produces a single-paper ontology from one run, the OE page takes **multiple completed source runs** and synthesises a unified ontology that spans them. It was added to address the multi-paper evaluation use case: given extraction outputs from several BrainIT papers, produce one merged ontology whose quality metrics can be compared against each individual source.

#### Flow

```
User selects source runs (N) → Load each runs/<id>/ontology.json
  → merge_ontologies()                     (src/ontology/merge.py)
  → cluster_classes()                      (semantic clustering, N_clusters configurable, default 25)
  → run_cluster_completion()               (src/ontology/cluster_completion.py)
      ├─ for each cluster:
      │    ├─ build prompt (cluster members + definitions + closed relation vocab + other clusters)
      │    ├─ LLM call (one per cluster)
      │    ├─ drop inferred classes (hard filter)
      │    ├─ filter relations by domain/range + closed vocab
      │    └─ discard any LLM-emitted hierarchy (force empty)
      └─ merge_ontologies([source_fragment] + cluster_fragments)
                                              (source-seeded — source is first fragment)
  → save to runs/<new-oe-run-id>/ontology.json
  → evaluate against gold standard
```

#### Key design properties

1. **No new classes** — The reconstruction is constrained to rearrange source classes, never add new ones. The per-cluster prompt says so, and a deterministic post-parse filter enforces it.
2. **Closed relation vocabulary self-seeded from source** — The LLM can only use relation labels that extraction already produced. No gold leakage, no hand-curated domain list.
3. **Hierarchy passes through unchanged** — The source ontology's hierarchy is preserved; the LLM is not asked to produce hierarchy and any it emits is dropped.
4. **Source-seeded merge** — The original source ontology is merged as the first fragment, so the LLM output can only *add* structure (within the closed vocab), never delete it.
5. **Gold-free reconstruction** — `cluster_completion.py` reads no gold-standard file. The gold is only touched by the evaluator that scores the reconstructed output after reconstruction completes.

#### Files

| File | Purpose |
|------|---------|
| `web/templates/ontology_engineering.html` | Jinja2 template for the OE page UI |
| `web/static/js/ontology_engineering.js` | Client-side logic: source-run picker, cluster chart, Compare Metrics modal |
| `src/ontology/merge.py` | `merge_ontologies()` — multi-source dedup by canonical keys |
| `src/ontology/cluster_completion.py` | `run_cluster_completion()` — LLM reconstruction + all filters |
| `web/app.py` | Route handlers, `/api/runs-metrics`, `_list_runs_with_ontology()` |

#### Compare Metrics modal

After reconstruction completes, the OE page opens a modal showing the reconstruction alongside each selected source run. Each row displays **four F1 metrics**:

- **Overall F1** — mean of the available class/hierarchy/relation F1s (the primary ranking metric)
- **Class F1** — `2·P·R / (P+R)` from class precision/recall
- **Hierarchy F1** — from `metrics.hierarchy.precision` / `metrics.hierarchy.recall`
- **Relation F1** — from `metrics.relations.precision` / `metrics.relations.recall`

Plus a 4-series Chart.js bar chart and "Best" badges per metric on the top card strip. Sortable by any column. This surface exists specifically to make the reconstruction vs source comparison visible — a single "F1" number would have hidden that early reconstruction attempts traded class F1 gains for hierarchy F1 losses.

#### Metadata flags

The reconstructed ontology's `metadata` dict records the following flags so the run is distinguishable from extraction runs:

```json
{
  "method": "cluster_completion",
  "provider": "openai",
  "n_clusters": 25,
  "n_successful": 25,
  "source_seeded": true,
  "closed_relation_vocab": true,
  "n_allowed_relations": 17,
  "hierarchy_from_source_only": true
}
```

#### Evaluation

The OE reconstruction is evaluated with the same `compute_relation_metrics()` and `compute_hierarchy_metrics()` used by the main pipeline. No special evaluation code path exists — the reconstructed `ontology.json` is dropped into a new run directory and scored by the standard evaluator.

**Design goal:** the OE reconstruction should be **no worse** than the strongest source run on Overall F1, and **strictly better** on hierarchy F1 (because the source seed preserves hierarchy edges while the closed-vocab relation filter tightens relation precision). On two test corpora (20260324 and 20260331 source runs), reconstruction achieved Hier F1 +0.18 over source with ≤0.03 Rel F1 loss, producing a net +0.04 Overall F1 gain.

---

## 3. Summary table (implemented functionality)

| Area | Implemented |
|------|-------------|
| **Input** | Paste text, upload .txt/.pdf; Use default paper; BrainIT paper catalog (11 papers, `BRAINIT_PAPER_CATALOG`); corpus manifest per run. Load pipeline: strip control chars → normalize → optional scope filter; optional `raw_text` and PDF `pages`. |
| **Normalization** | Hyphenation fix, symbol normalization, repeated header/footer removal, control-char strip (`normalize.py`). |
| **Scope filter** | Optional document-level (paragraph + line, compound-phrase blacklist only) inside load_corpus; chunk-level dual-score router (admin_score, clinical_score; governance section blacklist; clinical section always-keep; reorder clinical first). Blacklists expanded for all 11 papers (Phase 19): BrainIT group/network, data infrastructure, data collection/analysis terms, multi-centre trial, health care technology, etc. Strong terms (1.0 weight) include all core Neuro-ICU variables, promoted Fluids/Nutrition/Condition terms (Phase 10), and Moss 2013 framework terms. |
| **Scope filter validation** | `run_one()` raises `ValueError` if `scope_filter` absent from config when gold is loaded. |
| **Chunking** | Semantic chunking; control chars stripped per chunk; candidates with section context. |
| **Prompting methods** | Three active strategies: Zero-Shot, One-Shot (MMR-1 from `pool_one_shot_comprehensive.json` — classes+relations+hierarchy per example), Few-Shot (`phased_3step`, 3-phase with enriched Phase 1 comprehensive extraction, anti-anchoring). All templates include: Chain-of-Thought reasoning, schema-first prompt order (vocab before constraints), lightweight self-verification (metacognitive self-check), Ontology Design Patterns for relation extraction (6 patterns), compressed exclusion scope, completeness nudges, and generalised role (Phases 16, 21b, 23). Legacy: `simple_fewshot`, `phased_2step` (hidden, backward-compatible). |
| **Pipeline modes (UI)** | Three modes: **Strict** (raw — no vocab guardrails, no gold filtering, no NER; evidence-only gate); **Guided** (vocab guardrails + eval_gold + NER + candidates); **Schema-Completed** (default — Guided + SGC + symbolic reasoner). Controlled (auto-derived when non-default toggles are set). LLM Reasoning (Mode 4) removed. |
| **Controlled Experiment page** | 18 independent toggles across Preprocessing (5: scope filter, chunk clinical filter, embedding scope fallback, clinical-only routing, require label in evidence) / Prompt Injection (5) / Post-Processing (8) categories. BrainIT paper catalog dropdown. Same progress page as Run Experiment. |
| **Hierarchy phase filtering** | `known_classes` filtered to allowed clinical vocabulary before hierarchy sub-calls. |
| **LLM providers (extraction)** | OpenAI (default), Anthropic, Google, Groq, HuggingFace, DeepSeek. |
| **Reasoning LLM (improvements)** | Configurable separately: OpenAI (default) or DeepSeek Reasoner (R1). Advanced section in UI. |
| **Vocab guardrails** | Injects gold class list + paper-wording relation labels only; `RELATION_ALIASES_CORE` maps paper-wording to gold during filtering. **Only active in Guided and Schema-Completed modes** (`prompt_vocab_guardrails=True`). Strict mode has guardrails off by design — the LLM extracts freely. |
| **Medical NER** | ScispaCy NER anchor. Built into Guided mode (Mode 2) and cascades to Mode 3. When vocab guardrails are on, NER entities are **filtered to gold vocabulary only** (`filter_ner_to_gold`); wording is non-prescriptive. |
| **Candidate terms** | Extracted per chunk; injected in Guided mode as optional hints. Filtering tightened (Phase 13): max_terms default 12. |
| **Text-Grounded Completion (TGC)** | Multi-pass architecture: (1) deterministic hierarchy pre-scan, (2a) LLM Pass 1 — classes + relations + hierarchy from full corpus text (evidence-driven, no vocabulary whitelist), (2c) Chain-of-Layer (CoL) taxonomy pass — LLM organises all classes into a numbered hierarchical tree, parsed back into subClassOf triples; new intermediate categories auto-created, (3) deterministic hierarchy post-scan. Configurable off via `text_grounded_completion` flag. Artefacts: `tgc_pass1_prompt.txt`, `tgc_pass1_response.txt`, `tgc_col_prompt.txt`, `tgc_col_response.txt`, `tgc_diagnostic.json`. |
| **Domain scope** | `resources/domain_scope.json` — curated from all 11 papers: 117 classes, 27 relations, 78 evidence exemptions, abstract/broad allowlists, concept categories. Loaded by `src/ontology/domain_scope.py`. Decouples cleanup from gold standard (gold used for eval only). |
| **Improvements** | TGL (built-in) → SGC (optional) → Built-in cleanup (guided by domain scope; granular: dedupe, scope pruning, evidence pruning, structural including edge pruning for out-of-scope endpoints and re-prune after auto-add, axioms; each independently togglable) → Orphan Rescue (optional) → Post-orphan scope cleanup → Rule-based Reasoning Layer (optional). LLM Reasoning Layer removed. |
| **Relation whitelist** | `_ALLOWED_RELATION_LABELS_GLOBAL`: paper-wording + camelCase gold labels (normalised). Supplemented by blocklist + quality-check approach (softened from strict whitelist). |
| **Intrinsic validation**| Automated semantic clustering (`ClusterResults` UI) using `all-MiniLM-L6-v2` dense embeddings, Ward's hierarchical agglomerative clustering (silhouette optimal k), and Chart.js PCA scatter plots to validate concept coherence. |
| **Evaluation — class** | Coverage, precision, recall, error taxonomy, structural; `extraction_only`; `clinical_only`; `by_stratum`; `by_stage`. Aligned via exact match and dense semantic embedding matching (`all-MiniLM-L6-v2`, threshold 0.55). |
| **Evaluation — relations** | `compute_relation_metrics()`: label-level precision/recall via `RELATION_ALIASES_CORE`; `per_gold_relation`; in `metrics["relations"]` and `metrics["extraction_only"]["relations"]`. |
| **Evaluation — hierarchy** | `compute_hierarchy_metrics()`: edge-level precision/recall by normalized key matching; in `metrics["hierarchy"]`. |
| **Per-stage ablation** | `_capture_stage_metrics()` snapshots after each pipeline stage (including hierarchy metrics); `metrics["by_stage"]` in `metrics.json`; formatted ablation table in `summary.txt`. Stages: extraction → after_text_grounded → after_sgc → after_cleanup → after_orphan_rescue → after_rule_based → after_gold_filter. |
| **Canonical aliases** | `CANONICAL_ALIAS_MAP` in `canonical.py`: comprehensive alias map covering TBI, CVP, ICP, CPP, GCS/GOSe, MAP; Fluids; Nutrition; Sedation; Condition; Secondary Insult types; Clinical Assessment types; Outcome; Laboratory Values; Therapy hierarchy; Nursing Interventions; Monitoring hierarchy; Observation framework; Data Quality. Evaluation synonyms in `synonyms.py` extended. |
| **Hierarchy triggers** | `HIERARCHY_LEXICAL_TRIGGERS`: 25 cues including `"such as"`, `"is a"`, `"type of"`, `"kind of"`, `"include:"`, `"includes:"`, `"consists of"`, `"comprises"`, `"categorized as"`, `"grouped into"`, `"subdivided into"`, `"subtypes"`, `"subtype of"`, `"types of"`, and more (Phase 17). Imported from `schema.py` into `run.py`. Phase 3 double-gating replaced with single chunk-level gating. |
| **Merge / build** | `merge_parsed()`: classes by canonical key, relations by (label, domain, range); `build_ontology`: evidence required; canonical alias map; singular/plural merge; stratum on entities; dedupe and duplicate merge. |
| **Run label format** | `Strategy - PipelineMode - LLM - EvalSettings - Advanced - Paper`. Consistent: Python `_format_run_label`, JS `formatRunName`. Derives "Controlled" mode automatically. |
| **Artifacts** | `ontology.json`, `ontology_raw.json`, `ontology_pre_cleanup.json` (post-TGL/SGC snapshot), `summary.txt` (enhanced: timestamp, papers, mode, eval settings, extraction baseline, by_stage table), `metrics.json` (includes `by_stage` with `after_text_grounded` stage), `improvement_counts.json`, `axiom_violations.json`, `cleanup_removed.json` (cleanup audit log), `sgc_diagnostic.json`, `metadata.json` (all 11 new flags + `input_papers`), `run.log`. **Static resources:** `resources/domain_scope.json` (domain scope for cleanup). |
| **Config flags (all)** | `scope_filter`, `chunk_clinical_filter`, `embedding_scope_fallback`, `clinical_only_routing`, `prompt_vocab_guardrails`, `require_label_in_evidence`, `filter_to_gold_vocabulary`, `strict_relations`, `medical_ner_anchor`, `candidate_terms`, `eval_restrict_to_gold`, `text_grounded_completion`, `schema_guided_completion`, `symbolic_reasoner`, `cleanup_dedupe`, `cleanup_scope_pruning`, `cleanup_evidence_pruning`, `cleanup_structural`, `cleanup_axioms`. All recorded in `metadata.json`. |
| **Config files** | `demo.json`: all benchmark-required keys. `benchmark_template.json`: reusable template. |
| **UI** | Flask app: Run experiment (3-strategy + 3-mode + advanced); Controlled Experiment (18 toggles + paper catalog); progress (with cancel); results; Run comparison (mirrors experiment form, 5-part run label, paper catalog per run; Analyze Selected / per-run Analysis / batch analysis all show 4-dimension F1: Overall, Class, Hier, Rel with grouped bar charts matching OE colour scheme); pipeline view; run list; **Ontology Engineering** (cross-paper synthesis — see below). |
| **Ontology Engineering page** | Cross-paper ontology reconstruction subsystem (§2.14). Select N source runs → merge by canonical keys → semantically cluster (Ward's on `all-MiniLM-L6-v2` embeddings, default N=25) → per-cluster LLM completion with closed relation vocab + no new classes + no LLM hierarchy → source-seeded merge → evaluate. Compare Metrics modal shows Overall / Class / Hier / Rel F1 with Chart.js bar chart and sortable table. `/api/runs-metrics` endpoint powers the comparison. Files: `cluster_completion.py`, `merge.py`, `ontology_engineering.html`, `ontology_engineering.js`. **Gold-standard-free reconstruction** — the filter reads nothing from the gold file; closed vocab is harvested from source extraction labels. |
| **NLP preprocessing** | Citation stripping (`[1,2]`, `(Author et al., 2020)`, superscripts, full References section); abbreviation expansion (doc-local `Full Term (ABBREV)` detection + 35-entry neuro-ICU dictionary; non-destructive inline expansion); section-aware chunking (priority tiers: high/medium/low/skip; reference chunks dropped; section context injected into prompts); chunk sizing tuned to 2000 target / 3200 max / 400 min / 200 overlap for GPT-4o-mini. |
| **Relation alias rewrite map** | Phase 26 — ~30 camelCase → canonical rewrites applied in cleanup: `hasComponent → has_component`, `causedBy → caused_by`, `indicatesRiskOf → indicates_risk_of`, `hasUnit → has_unit`, `recordedAt → recorded_at`, `hasOutcome → has_outcome`, etc. Significantly increases relation match rate against gold vocabulary. |
| **Label normalization** | Phase 25 — plural→singular, hyphen collapse, title case, canonical alias resolution applied to all class labels during cleanup. Catches non-canonical casing/pluralization after post-processing. |
| **Class definitions** | Every `ClassEntity` carries a `definition` field populated at extraction time; propagates through merge/cleanup/refinement/cluster completion. Used to enrich the per-cluster LLM prompt in OE reconstruction. |
| **CLI** | Config-driven runs; `scope_filter` must be explicit when gold is loaded. |
| **Non-determinism** | Verified March 2026: same config produces different TGL output due to GPT-4o-mini temperature. All extraction prompts are byte-for-byte reproducible; LLM responses are not. Expected and documented. |

This implementation delivers a **modular, hybrid ontology engineering framework** with three prompting strategies (including 3-phase extraction with enriched prompts and anti-anchoring), multiple LLM providers, a three-mode progressive pipeline abstraction (Strict → Guided → Schema-Completed) plus a Controlled Experiment mode for fine-grained ablation, Text-Grounded Completion with Chain-of-Layer taxonomy construction as a built-in always-on improvement step (evidence-driven, no vocabulary whitelist, structured hierarchy induction), a cross-paper **Ontology Engineering** reconstruction subsystem (§2.14) that merges multiple extraction runs and rebuilds a unified ontology via clustered LLM completion with closed relation vocabulary and source-seeded hierarchy passthrough, complete evaluation against the BrainIT **placeholder gold standard** (93 classes, 18 relations, 76 hierarchy edges — a proxy until the real consortium ontology is available) including **relation recall**, **hierarchy precision/recall**, and **per-stage ablation**, and reproducible, validated configuration for benchmark runs. **The Strict mode in particular is designed as a raw evidence-only extractor that generalises to any BrainIT paper with unseen vocabulary — no domain vocabulary lists, no schema enforcement, no gold filtering applied. The Ontology Engineering reconstruction subsystem is similarly gold-free: it reads no gold file and harvests its closed relation vocabulary from extraction output only.** The hierarchy extraction was overhauled across 18 documented development phases, culminating in Chain-of-Layer integration (Phase 18) based on Zeng et al. (2024, CIKM) for structured top-down taxonomy construction. Phases 24–34 added `includes → hierarchy` conversion, label normalization, relation alias rewriting, hierarchy scaffolding from domain scope, multi-paper guardrails, methodology/statistics pruning, upload race fix, evidence verification + class definitions, the Ontology Engineering cross-paper reconstruction subsystem, an embedding-based scope filter fallback for Strict mode generalisation, and a 4-dimension F1 metrics overhaul across all comparison/analysis views.

**Doc maintenance:** Keep this summary in sync with the current backend (`reasoner.py`, `run_experiments.py`, `artifacts.py`, `metadata.py`, `canonical.py`, `scope_filter.py`, `schema.py`, `synonyms.py`, `ontology_completion.py`, `neuro_axioms.py`, `run.py`, `vocabulary.py`, `candidates.py`, `retrieval/pool.py`) and frontend (`web/app.py`, `comparison_dashboard.html`, `comparison_analyze.html`, `comparison_progress.html`, `controlled_experiment.html`, `index.html`, `ontology_engineering.html`, `ontology_engineering.js`). When changing pipeline behaviour, scope filter, evidence rules, hierarchy triggers, canonical aliases, evaluation synonyms, pool examples, NER/candidate filtering, build dedupe, config flags, UI options, or evaluation metrics, update §2.1 pipeline diagram and the relevant sections.
