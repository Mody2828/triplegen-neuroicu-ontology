# Chapter 3: Implementation of the LLM-Driven Ontology Engineering Framework for Neuro-ICU

This chapter walks through how the framework was built, explaining each part of the pipeline, why certain design choices were made, and how everything fits together. It covers the architecture, prompting strategies, post-processing layers, evaluation, and the web interface.

---

## 3.1 Project aim and objectives

### 3.1.1 Aim

The goal of this project is to build and evaluate an LLM-driven ontology engineering framework for the Neuro-ICU domain, using BrainIT clinical literature as input. The key question is: how well can language models reconstruct a domain ontology in a low-resource clinical setting, and what factors — prompting style, constraints, post-processing, and choice of LLM — help or hurt the result?

### 3.1.2 Objectives

The first objective is to build a complete, reproducible pipeline that goes from raw text (pasted or uploaded) all the way to a generated ontology with an evaluation report. The second is to support multiple LLM providers — OpenAI (GPT-4o-mini), Anthropic (Claude Haiku 4.5), Google (Gemini 2.5 Flash), Groq (Llama 3.1 8B), Hugging Face (Mistral 7B), and DeepSeek (deepseek-chat) — so the extraction method and model can be swapped independently.

Another objective is to compare different prompting approaches: zero-shot (no examples), one-shot (one example), and few-shot with a three-phase decomposition, to see how examples and task decomposition affect extraction quality. The framework also provides four progressive pipeline modes — Strict, Guided, Schema-Completed, and Fully Reasoned — that bundle post-processing features into named experimental conditions for clean ablation studies. The idea is that each mode is a superset of the previous one, so you can see exactly what each added component contributes.

On top of that, the framework supports optional enhancement layers — schema-guided completion, an LLM reasoning layer, and a rule-based reasoning layer — that can be toggled independently. Evaluation is done against the BrainIT gold standard using coverage, precision, recall, relation recall, hierarchy precision/recall, structural metrics, error taxonomy, and a per-stage ablation table. Finally, there's a web UI for running experiments, saving artifacts, and comparing results side by side.

The overall positioning is as a modular hybrid ontology engineering system that combines prompting strategies with optional LLM and symbolic reasoning layers — not just a set of prompt experiments.

---

## 3.2 Implemented design and functionality

### 3.2.1 High-level architecture

The system has four conceptual layers:

1. **Extraction layer** — how the LLM pulls ontology elements from text: the prompting method, and where applicable, retrieving few-shot examples using MMR (Maximal Marginal Relevance).
2. **Merge and vocabulary filter** — chunk-level extractions are combined into one ontology and optionally restricted to the gold vocabulary for evaluation.
3. **Post-processing** — built-in cleanup (always runs when gold is available), then optional improvements in order: schema-guided completion → LLM reasoning → rule-based reasoning.
4. **Evaluation** — alignment to the gold standard, metrics computation, per-stage snapshots, and summary generation.

### 3.2.2 Pipeline order

The pipeline flows like this:

1. **Corpus input** — text is pasted or uploaded (`.txt` / `.pdf`).
2. **Preprocessing** — control character stripping, normalisation, and optional scope filtering.
3. **Chunking** — the corpus is split into semantic chunks with section context preserved.
4. **Extraction** — each chunk is processed by the selected prompting strategy. Evidence requirements are enforced and Phase 2 output is vocabulary-filtered before merge.
5. **Merge** — chunk outputs are combined using canonical alias mapping, singular/plural normalisation, and deduplication. If evaluation controls are on, a gold vocabulary filter is applied. This is the **extraction stage snapshot**.
6. **Schema-guided completion** → snapshot.
7. **Built-in cleanup** → snapshot.
8. **LLM Reasoning Layer** (if enabled) → snapshot.
9. **Rule-based Reasoning** (if enabled) → snapshot.
10. **Gold-vocab filter** (if enabled) → snapshot.
11. **Validation and evaluation** — metrics computed, artifacts and summary generated.

Each snapshot captures class/relation/hierarchy counts and all evaluation metrics, written to `metrics["by_stage"]` in `metrics.json` and displayed as an ablation table in `summary.txt`.

The pipeline is illustrated in the methodology graph shown below.

![Pipeline](screenshots/1.jpg)

### 3.2.3 Corpus input, normalisation, scope filtering, and chunking

#### Input

The framework accepts text in three ways: paste directly, upload one or more `.txt`/`.pdf` files, or use the built-in **default paper** option. Multiple uploaded files are loaded as a multi-document corpus. When a single file is used (default paper or pasted text), the corpus path points directly to that file to make sure only the intended document gets processed.

#### Load pipeline

For each document, the load sequence is: raw text → strip control characters → normalise → optionally apply scope filter → store as `doc["text"]`. PDF documents also get `doc["pages"]` for page-level segments (useful for header/footer detection).

#### Normalisation

Normalisation handles the messy stuff that comes with real-world text: fixing broken hyphenation across line breaks (e.g. `intra-\ncranial` → `intracranial`), normalising Unicode symbols like fancy dashes and curly quotes, removing repeated headers/footers, and stripping control characters.

#### Scope filter

The scope filter is one of the more important preprocessing steps. BrainIT papers contain a lot of governance, organisational, and administrative content (steering groups, ethics committees, database access policies, etc.) that has nothing to do with clinical ontology. Without filtering, the LLM wastes extraction effort on this irrelevant material.

The filter works at two levels. At the document level, it removes paragraphs and lines that are heavily administrative. At the chunk level, it uses a dual-score router: each chunk gets an `(admin_score, clinical_score)` pair, and chunks with high admin scores but low clinical scores get dropped. Governance sections (Part A, group formation, database access, ethics approval) are always dropped. Clinical sections are always kept and reordered to appear first.

The blacklist only contains compound phrases like `steering group`, `ethics committee`, `data validation staff`, and `publication criteria` — single generic words are deliberately excluded to avoid accidentally removing clinical content.

Clinical scoring uses weighted terms. Strong terms (weight 1.0) include `heart rate`, `ICP`, `GCS`, `sedation`, `ventilation`, `monitoring`, `CPP`, `MAP`, `fluids`, `nutrition`, and `condition`. Broad terms like `core dataset` and `outcome` get 0.5. Terms like `fluids`, `nutrition`, `condition`, and `sedation levels` were specifically promoted to strong because they represent legitimate clinical concepts that happen to appear in brief mentions — without promotion, chunks containing only these terms would get filtered out before reaching the LLM.

When a gold standard is loaded, the `scope_filter` setting must be explicitly defined in the config. If it's missing, the pipeline raises an error to prevent accidentally running with different filter settings.

#### Chunking

Documents are split into semantic chunks that stay coherent while fitting within prompt limits. Each chunk gets control characters stripped again, and candidate terms are extracted from noun phrases and stored alongside the text for optional prompt injection.

### 3.2.4 Prompting methods

Three prompting strategies are available in the UI. Two legacy strategies remain for backward compatibility but are hidden.

#### Zero-Shot

No examples at all — the model extracts classes, relations, and hierarchy directly from text using only its instructions. It uses a detailed baseline prompt with strict scope rules, evidence requirements, and hierarchy constraints. This serves as the control condition for comparison.

#### One-Shot (MMR-1)

Retrieves one best concept example per chunk from the concept pool using MMR with `k=1`. There's also an optional hierarchy sub-phase that runs when the chunk text has hierarchy cues and at least one extracted class.

#### Few-Shot (three-phase, primary strategy)

This is the main strategy. It breaks the extraction task into three separate LLM calls, each focused on a different ontological concern. The idea behind this decomposition is that extracting classes, relations, and hierarchy edges are fundamentally different tasks that benefit from separate focus and phase-specific examples. Trying to do everything at once in a single prompt often leads to the model prioritising one task over others.

**Phase 1 — Class extraction.** Three concept examples (selected via MMR) prompt the LLM to extract classes. The Phase 1 instruction is deliberately comprehensive — it lists eight specific concept types to look for:

- Clinical conditions and secondary insults (e.g. Hypotension)
- Physiological parameters and monitoring variables (e.g. Heart Rate, ICP, Temperature)
- Treatments, therapies, and interventions (e.g. Sedation, Ventilation, Fluids)
- Clinical assessments and outcomes (e.g. GCS Assessment, GOSe Outcome)
- Laboratory values and biochemistry (e.g. Sodium, Glucose, Haemoglobin)
- Nursing interventions and bedside activities (e.g. Physiotherapy, Patient Transport)
- Data quality and sensor concepts (e.g. Sensor, Data Quality Assessment)
- Abstract category classes when explicitly named (e.g. Monitoring Data, Therapy, Observation)

The instruction also includes a "breadth over caution" directive: if the text mentions 15 concepts, extract all 15. Don't stop early. This was added because earlier testing showed that few-shot examples were anchoring the model too narrowly — if the examples only showed monitoring parameters, the model would only extract monitoring parameters and skip therapies, assessments, lab values, etc.

**Phase 2 — Relation extraction.** Three relation examples prompt the LLM for relations. The instruction also explicitly allows the LLM to extract supplementary classes that Phase 1 might have missed — for instance, a concept needed as domain or range for a relation. Hierarchy is deferred to Phase 3. Output is vocabulary-filtered before merge.

**Phase 3 — Hierarchy extraction.** Three hierarchy examples prompt the LLM to extract subclass/superclass edges, but only where the text contains explicit cues like "such as", "is a", "type of", "kind of", "include:", or "includes:". The prompt shows concrete examples of how enumeration cues map to edges (e.g. "parameters such as Heart Rate" → `Heart Rate subClassOf Parameter`).

Phase 3 only runs when: (a) at least one class has been extracted, (b) the chunk text contains at least one hierarchy trigger phrase, and (c) hierarchy examples are available. When fewer than five classes have been extracted, gold vocabulary labels are injected as valid endpoint labels so the LLM can still form useful hierarchy edges even when earlier phases were conservative. Output is filtered through `filter_hierarchy_to_lexical_cues` before merge.

#### Anti-anchoring

A key design element in the few-shot template. When you give an LLM a few examples, it tends to anchor on the surface features of those examples — if all three examples show monitoring parameters, the model assumes it should only extract monitoring parameters. This is a well-known problem in in-context learning.

To counter this, the prompt includes an explicit anti-anchoring instruction: "The examples below demonstrate OUTPUT FORMAT and evidence style ONLY. They come from DIFFERENT text passages and may cover different concept types than your text. Do NOT limit your extraction to the concept types shown in the examples. Extract EVERY clinical concept the current text supports, even if no example shows that type."

The traditional "Reject rule: when in doubt, omit" was also replaced with a more balanced "Confidence rule: if a clinical concept is clearly named in the text (even briefly), include it." The original reject rule was too conservative for few-shot mode and was causing the model to under-extract compared to zero-shot.

#### MMR for example selection

For all few-shot strategies, examples are selected from task pools using Maximal Marginal Relevance (MMR). The reason for using MMR rather than just picking the top-k most similar examples is that simple top-k selection tends to pick redundant examples that all cover similar content, wasting prompt space. MMR balances two things: how relevant each example is to the current chunk, and how different it is from the examples already selected.

Concretely, the score for each candidate is:

`MMR(d) = λ · sim(d, query) − (1−λ) · max_sim(d, already_selected)`

with `λ = 0.7` (favouring relevance over diversity). The first example picked is the most similar to the chunk text; subsequent picks maximise the MMR score. Similarity is cosine similarity on TF-IDF vectors.

There's also a `gold_first` option that ensures the first (gold-standard-aligned) example in each pool is always included, providing a reliable format anchor while letting MMR choose the rest for relevance and diversity.

#### Task-specific example pools

Three pools provide the examples:

- **`pool_strict_concepts.json`** — six class-first examples from the BrainIT core dataset, all aligned to gold-schema labels with direct source-text quotations.
- **`pool_strict_relations.json`** — six relation examples with gold class labels for domain/range, including one that teaches `monitoring indicates condition` and `targets condition`.
- **`pool_strict_hierarchy.json`** — five hierarchy examples, including two using verbatim BrainIT paper sentences with the `"include:"` enumeration pattern (monitoring parameters as subclasses of Monitoring Data; ICU management types as subclasses of Intensive Care Management).

#### Vocabulary guardrails

When enabled, the prompt injects gold class labels and paper-wording relation labels (not CamelCase gold relation labels, which would cause the LLM to echo labels that fail the post-extraction whitelist). A "leaf-first rule" tells the model to prefer specific labels (ICP, CPP, MAP) over broad containers (Monitoring Data, Therapy) when both appear in the text.

### 3.2.5 Medical NER anchor

The framework uses biomedical Named Entity Recognition to pre-identify clinical entities before LLM extraction. The idea is simple: run a specialised NER model over the text first, then inject the found entities into the prompt as "Suggested concepts." This gives the LLM a head start — it already knows which clinical terms are present and is less likely to miss or hallucinate them.

The NER component tries to load models in this order:

1. **ScispaCy** models — preferring `en_ner_bc5cdr_md` (trained on the BioCreative V Chemical Disease Relation corpus, recognises diseases and chemicals), then `en_ner_jnlpba_md` (biomedical named entities), then the generic scientific models.
2. **Med-spaCy** models — clinical NLP pipelines as a fallback.

ScispaCy provides spaCy-compatible models trained specifically on biomedical text. The preferred BC5CDR model is a good fit for Neuro-ICU because it picks up diseases and chemicals, which map closely to clinical conditions, therapies, and parameters in our domain.

The function processes up to 500,000 characters, extracts unique entity spans (min 2 chars, deduplicated), and returns up to 150 suggested concepts. If no NER model can be loaded (e.g. spaCy isn't installed), extraction simply continues without NER — the system degrades gracefully.

Medical NER is built into Guided mode (Mode 2) and cascades automatically into Schema-Completed and Fully Reasoned modes. It's not a separate toggle — it's always on when you use Guided or above.

### 3.2.6 LLM providers

The framework supports six extraction providers: OpenAI GPT-4o-mini (default), Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B (free tier), Hugging Face Mistral 7B (Router API), and DeepSeek deepseek-chat. An abstraction layer normalises the API differences and supports configurable `max_tokens`.

For improvement stages (SGC, LLM Reasoning), a separate reasoning LLM can be configured. Default is OpenAI GPT-4o-mini; the alternative is DeepSeek Reasoner (R1), which is stronger at reasoning but slower. This separation means you can use a fast model for extraction and a stronger one for reasoning, or use the same for both — either way it's reported in the summary and run label.

### 3.2.7 Pipeline modes

The UI has four progressive pipeline modes. Each one adds features on top of the previous mode, making it easy to see what each component contributes. This layered approach is standard in NLP evaluation — you want to isolate the effect of each component rather than testing everything at once.

| Mode | What it adds | Config flags |
|------|-------------|--------------|
| **Strict** | Extraction with gold vocab guardrails, nothing else | `prompt_vocab_guardrails`, `eval_restrict_to_gold` |
| **Guided** | + Medical NER anchor + candidate term injection | + `medical_ner_anchor`, `candidate_terms` |
| **Schema-Completed** | + LLM schema gap filling + rule-based hierarchy completion | + `schema_guided_completion`, `symbolic_reasoner` |
| **Fully Reasoned** | + LLM Reasoning Layer (PROPOSE/VERIFY) | + `llm_reasoning` |

Schema-Completed is the default because it balances capability and speed. The mode is always reported in the run label (`Strategy - Mode - LLM - EvalSettings - Advanced`) and in the summary.

### 3.2.8 Advanced features

The Advanced section provides a reasoning LLM override — you can pick a different LLM for improvement stages. Default is OpenAI; the alternative is DeepSeek Reasoner (shows as `DSR` in the run label).

### 3.2.9 Post-processing improvements

Post-processing runs after merge in a fixed order: SGC → Cleanup → LLM Reasoning → Rule-based. The ordering matters: SGC fills gaps first, cleanup removes noise, and then the reasoning layers work on a clean ontology.

#### Schema-guided completion (SGC)

SGC looks at what's in the gold schema but not yet in the generated ontology, then asks the LLM: "Which of these missing items are actually mentioned in the corpus?" It only adds items that are (a) in the schema, (b) have corpus evidence, and (c) for relations, have valid domain/range. Matching uses canonical-aware normalisation so abbreviations like "ICP" match "Intracranial Pressure (ICP)."

A few important implementation details:

- **Evidence quality instruction** — The prompt explicitly tells the LLM that evidence must be actual text copied from the corpus, not ontology URIs (like `pd:Session`) or just the label repeated. This was added after investigation showed the LLM producing fake evidence that failed downstream validation.
- **Lenient evidence threshold** — SGC parses responses with `min_evidence_length=1` instead of the default 12 characters. Short clinical terms like "EtCO2" (5 chars), "TCD" (3 chars), and "nutrition" (9 chars) are perfectly legitimate — the 12-character threshold is appropriate for regular extraction but too aggressive for SGC where the LLM is specifically asked about known gold items.
- **Large token budget** — `max_tokens=8192` to accommodate the full v2.0 gold schema (72 classes, 16 relations, 57 hierarchy edges). Previous runs with 4096 tokens were getting truncated.
- **Truncated JSON recovery** — When the response gets cut off mid-JSON, a recovery function strips trailing commas and tries multiple closing sequences to salvage whatever classes, relations, and hierarchy entries were generated.
- **Diagnostic logging** — A `sgc_diagnostic.json` file records counts at each filter stage for debugging without full re-runs.

#### Built-in cleanup

Always runs when gold is available. It cleans up the merged ontology before reasoning layers touch it. The cleanup runs in a fixed order:

1. **Deduplication** — classes, relations, hierarchy via canonical keys; merges provenance and aliases.
2. **Out-of-scope pruning** — removes governance labels (compound patterns like `steering group`, `committee`).
3. **Abstract data label pruning** — removes `raw data`, `core data`, `dataset` etc., but keeps `Monitoring Data`.
4. **Broad label pruning** — removes generic labels like `patients`, but keeps gold-aligned ones like `Patient`, `Therapy`.
5. **Evidence pruning** — removes classes without evidence, except for 30+ protected gold-schema anchors (`Condition`, `Fluids`, `Nutrition`, `Sedation`, `Session`, `Timepoint`, `Observation`, `Parameter`, etc.).
6. **Dangling hierarchy pruning** — removes hierarchy edges whose endpoints were pruned.
7. **Relation domain/range pruning** — removes relations pointing to pruned classes.
8. **Relation evidence pruning** — labels must be in the allowed relation whitelist.
9. **Hierarchy fragment pruning** — removes edges with verb forms or clause fragments.
10. **Axiom constraints** — removes structures violating physiological/semantic type constraints (17 forbidden hierarchy pairs, 8 allowed relation types).

#### LLM Reasoning Layer

Used in Fully Reasoned mode. Two LLM calls — PROPOSE then VERIFY — that infer missing hierarchy edges from the clean ontology and gold schema. No corpus text is used; only schema-licensed edges are applied. The two-step approach reduces incorrect inferences by having the model first propose edges and then independently verify them.

Matching uses canonical-aware alias resolution so "ICP" correctly matches "Intracranial Pressure (ICP)" in the gold hierarchy.

#### Rule-based Reasoning Layer

Acts as a deterministic fallback after LLM reasoning. It adds hierarchy edges from the gold schema whenever both endpoint classes exist in the ontology, each with a synthetic evidence string (`"Schema-inferred: X is a subclass of Y."` — this contains `"is a"`, which is a valid hierarchy trigger). It also prunes orphan classes that aren't connected to anything, while always preserving gold-aligned classes.

### 3.2.10 Gold standard and evaluation

#### Gold standard

The BrainIT ontology is loaded from a `.ttl` (Turtle) file using RDFLib. The loader handles multi-parent hierarchies (classes with multiple `rdfs:subClassOf` statements). Three modes: **public** (surrogate gold), **restricted** (use provided gold for alignment), **isolated** (generate gold from separate corpus).

#### Class alignment

Generated classes are aligned to gold using: exact matching, semantic TF-IDF matching (threshold 0.55, one-to-one assignment), and synonym expansion via `DOMAIN_SYNONYMS`.

#### Metrics

- **Class metrics** — coverage, precision, recall, plus extraction-only (before improvements), clinical-only (excluding governance classes), by-stratum (core/governance/provenance), and error categories (hallucinations, schema violations, omissions).
- **Relation metrics** — label-level precision and recall via `RELATION_ALIASES_CORE`, with per-gold-relation breakdown.
- **Hierarchy metrics** — edge-level precision and recall using normalised `(subClass, superClass)` key matching.
- **Per-stage ablation** — snapshots after each pipeline stage (extraction, after_sgc, after_cleanup, after_llm_reasoning, after_rule_based, after_gold_filter), written to `metrics["by_stage"]` and rendered as a table in `summary.txt`. This lets you see exactly where each metric changed and which component caused it.

Evaluation outputs: `metrics.json`, `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`.

### 3.2.11 Run artifacts and summary

Each run gets a unique ID (timestamp + short hash) and a directory with:

- **Config/corpus** — `corpus_manifest.json`, `metadata.json` (with `input_papers`, config, environment).
- **Prompts** — saved prompts for each chunk and phase (when enabled).
- **Generated ontology** — `ontology.json`, `ontology_raw.json`, `summary.txt`.
- **Evaluation** — `metrics.json` (with `by_stage`), `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`, `axiom_violations.json`, LLM reasoning patch files, `sgc_diagnostic.json`.
- **Logs** — `run.log`, `warnings.txt`.

The `summary.txt` is designed as a single human-readable file with everything: run metadata, input papers, improvement counts, final metrics, extraction-only baseline comparison, per-stage ablation table, and complete listings of classes, relations, and hierarchy.

### 3.2.12 Ontology graph visualisation

The framework includes an interactive graph visualisation that renders the generated ontology as a node-link diagram using Cytoscape.js. Classes appear as nodes, relations as directed edges, and hierarchy edges as dashed lines. Multiple layouts are available (force-directed/CoSE, breadth-first, circular, grid). Users can click nodes and edges for details, zoom/pan, and export to PNG. Accessible from the results page via "Show Ontology Graph."

### 3.2.13 Web UI

The Flask web app provides:

- **Run experiment** — three strategy toggles (Zero-Shot, One-Shot, Few-Shot), four pipeline mode cards, LLM provider selector, evaluation settings, and Advanced section. Few-Shot + Schema-Completed are selected by default.
- **Progress page** — live progress bar with cancel support, per-chunk progress.
- **Results page** — metrics, ablation table, ontology listing, improvement counts, ontology graph.
- **Run comparison** — mirrors the experiment form. Runs get five-part labels (`Strategy - Mode - LLM - EvalSettings - Advanced`), displayed in a five-column table. Three pre-configured comparison groups: Cross-LLM, Pipeline Modes, and Reasoning LLM.
- **Run list** — browse previous runs, access results and analysis.
- **Pipeline view** — visual diagram of the current pipeline.

Run label generation is consistent between Python (`_format_run_label`) and JavaScript (`formatRunName`).

---

## 3.3 Summary

The framework delivers a modular, end-to-end ontology engineering system with:

- **Three prompting strategies** — Zero-Shot, One-Shot (MMR-1), and Few-Shot (three-phase extraction with enriched prompts, anti-anchoring, and breadth-over-caution).
- **MMR-based example selection** — balancing relevance and diversity across task-specific pools.
- **Medical NER anchoring** — ScispaCy-based entity pre-identification built into Guided mode and above.
- **Six LLM providers** — with separate extraction and reasoning LLM configuration.
- **Four progressive pipeline modes** — Strict → Guided → Schema-Completed → Fully Reasoned, for structured ablation.
- **Robust post-processing** — SGC (with lenient evidence, large token budget, truncated-JSON recovery), ten-step cleanup, LLM Reasoning (PROPOSE/VERIFY), and rule-based reasoning.
- **Comprehensive evaluation** — class, relation, and hierarchy metrics with per-stage ablation against the BrainIT v2.0 gold standard (72 classes, 16 relations, 57 hierarchy edges).
- **Web interface** — Flask app with experiment running, progress tracking, results display, run comparison, ontology graph visualisation, and pipeline view.
- **Full artifact recording** — every run is reproducible with saved prompts, ontologies, metrics, configs, and diagnostic logs.
