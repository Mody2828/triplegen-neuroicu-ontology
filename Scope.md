# Chapter 3: Project Scope

## 3.1 Project Aim and Context

This project is about building and evaluating an LLM-driven ontology engineering framework for the neurointensive care (Neuro-ICU) domain. It takes BrainIT consortium publications and related Neuro-ICU literature as input and investigates how well large language models can reconstruct and align with an expert-designed domain ontology under controlled prompting and evaluation conditions.

The generated ontologies are evaluated against a BrainIT gold standard, which allows a rigorous look at alignment quality, failure modes, and what actually helps in a low-resource, safety-critical clinical domain.

## 3.2 Literature-Driven Motivation

Prior work has shown that while LLMs can extract structured knowledge from text, their performance on ontology engineering tasks depends heavily on prompt constraints, ontology guidance, and how evaluation is set up. Retrieval-augmented few-shot prompting improves relevance, but on its own it doesn't sufficiently reduce hallucinations in specialised domains. Research also shows that ontology-augmented prompting, phased extraction, and human-in-the-loop validation are important for reliability and alignment.

The Neuro-ICU domain adds another layer of difficulty. Clinical knowledge is inherently context-dependent — it relies on provenance, measurement configuration, validation status, and temporal structure. These dimensions are frequently missed by unconstrained LLM outputs. That's why this project adopts a hybrid symbolic-LLM framework rather than a fully automated approach.

## 3.3 Novelty

The project evaluates LLM-driven ontology engineering in a low-resource clinical domain using a clinician-designed gold standard ontology — not one of the well-established biomedical ontologies that most prior work relies on. This gives practical insight into how ontology constraints, retrieval strategies, phased prompting, and a four-stage progressive pipeline affect hallucination rates, structural alignment, and contextual completeness in specialised medical domains.

## 3.4 Research Objectives

Five objectives guided the project:

1. **Assess LLM capability** — determine whether LLMs can reconstruct BrainIT ontology concepts, relations, and hierarchy from Neuro-ICU literature.
2. **Compare prompting strategies** — evaluate how Zero-Shot, One-Shot, and Few-Shot approaches, including phased decomposition and anti-anchoring design, perform across multiple LLM providers.
3. **Compare source-document effects** — investigate how differences between input papers influence ontology extraction quality, including concept coverage, relation recovery, hierarchy reconstruction, and error patterns.
4. **Analyse failure modes** — identify common errors such as hallucinations, scope drift, anchoring bias, and structural inconsistencies.
5. **Evaluate pipeline components** — assess how vocabulary guardrails, Medical NER, schema-guided completion, and LLM reasoning layers influence alignment with the gold standard.

## 3.5 Delivered Methodology Pipeline

The framework implements a modular, reproducible pipeline composed of several connected stages.

### 3.5.1 Corpus Ingestion and Preprocessing

Input to the system is BrainIT publications and Neuro-ICU literature in PDF or text form, supplied through pasted text, uploaded files, or a default corpus. Multi-file uploads are supported. When using default or pasted input, the corpus path points directly to the single file rather than the directory, ensuring only the intended document gets processed.

The load pipeline follows a fixed sequence: strip control characters, normalise text, and optionally apply the clinical scope filter. Normalisation handles hyphenation repair, symbol normalisation, and removal of repeated headers/footers.

The clinical scope filter is an important piece. BrainIT papers mix clinical content with a lot of governance, organisational, and administrative material. The filter operates at two levels: document-level paragraph and line filtering using compound-phrase blacklists, and chunk-level dual-score routing based on administrative and clinical scores. Clinical sections (Monitoring, ICU Management, Secondary Insult Treatment) are always kept; governance sections get filtered out. When a gold standard is loaded in benchmark mode, the scope filter must be set explicitly so UI and CLI runs stay comparable.

Provenance is preserved throughout — each document and chunk carries source, section, and paragraph metadata, and PDF documents keep page-level segments.

### 3.5.2 Candidate Knowledge Identification

The scope filter performs lightweight clinical-versus-governance scoring using weighted term dictionaries. Strong terms (weight 1.0) include monitoring, ICP, CPP, MAP, GCS, SaO2, ventilation, fluids, nutrition, sedation, and condition. Broad terms like "core dataset" and "outcome" get 0.5. Terms like fluids, nutrition, condition, and sedation levels were specifically promoted to strong because they represent legitimate clinical concepts that appear briefly in text — without promotion they'd get filtered out before reaching the LLM.

Vocabulary guardrails kick in from Strict mode onwards. Gold class labels and paper-wording relation labels are injected into prompts to constrain what the LLM generates. CamelCase gold relation labels are deliberately excluded to avoid ambiguity. When no gold file is loaded, a fallback vocabulary of 72 v2.0 gold classes is used. A "leaf-first rule" also tells the LLM to prefer specific clinical labels (e.g. ICP, CPP) over broad containers (e.g. Monitoring Data) when both appear in the text.

Guided mode and above also include a Medical NER anchor. ScispaCy (`en_ner_bc5cdr_md`, trained on the BioCreative V Chemical Disease Relation corpus) pre-identifies biomedical entities in the text and injects them into the prompt as "Suggested concepts." The idea is to give the LLM a head start — it already knows which clinical terms are present, which reduces hallucination and improves recall. If ScispaCy isn't available, the system falls back through alternative models or simply proceeds without NER. This is a built-in part of Guided mode, not an optional toggle. On top of that, broader noun phrases extracted from the chunk are injected as candidate terms for additional hints.

### 3.5.3 Prompting and Knowledge Extraction

Three active prompting strategies are exposed in the UI. Example selection for few-shot strategies uses Maximal Marginal Relevance (MMR), which balances relevance to the current chunk against diversity among selected examples — this avoids picking redundant examples that all cover similar content and waste prompt space.

**Zero-Shot** (`baseline`) — no examples at all. The LLM extracts classes, relations, and hierarchy directly from the text using only its instructions. Serves as the control condition.

**One-Shot** (`one_shot`) — one MMR-selected concept example per chunk. May trigger a hierarchy sub-call when the chunk has hierarchy cues and at least one allowed class label.

**Few-Shot** (`phased_3step`) — the primary strategy. Breaks extraction into three separate LLM calls, each focused on a different concern:

- **Phase 1** — class extraction using three concept examples. The instruction is deliberately comprehensive, listing eight concept types to look for (conditions, parameters, therapies, assessments, lab values, nursing activities, sensor/data quality concepts, and abstract categories). It includes a "breadth over caution" directive so the model extracts everything rather than stopping early. This was added because testing showed few-shot examples were anchoring the model too narrowly.
- **Phase 2** — relation extraction using three relation examples. Also allows the LLM to extract supplementary classes that Phase 1 might have missed (e.g. concepts needed as domain/range). Hierarchy is deferred to Phase 3.
- **Phase 3** — hierarchy extraction using three hierarchy examples, but only when the chunk contains explicit cues ("such as", "is a", "type of", "kind of", "include:", "includes:") and at least one class has been extracted. When fewer than five classes are available, gold vocabulary labels are injected as valid endpoints so the LLM can still form useful edges. Output is filtered through `filter_hierarchy_to_lexical_cues`.

All phase outputs are vocabulary-filtered before merge, giving three LLM calls per chunk.

The few-shot template also includes an explicit anti-anchoring instruction: "The examples below demonstrate OUTPUT FORMAT and evidence style ONLY... Do NOT limit your extraction to the concept types shown in the examples." This counters the well-known tendency of in-context learning to anchor on the surface features of demonstrations rather than applying the task broadly.

Example pools: `pool_strict_concepts.json` (6 class examples), `pool_strict_relations.json` (6 relation examples including `monitoring indicates condition` and `targets condition`), and `pool_strict_hierarchy.json` (5 hierarchy examples, including two using verbatim BrainIT paper sentences with the "include:" pattern).

### 3.5.4 Progressive Pipeline Modes

Four pipeline modes progressively layer on functionality. Each mode is a superset of the previous one, making it easy to see what each component contributes:

| Mode | What it adds |
|------|-------------|
| **Strict** | Vocab guardrails + gold-restricted evaluation. No NER, no post-processing. |
| **Guided** | + Medical NER anchor + candidate term injection. |
| **Schema-Completed** | + Schema-guided completion (LLM gap-filling) + rule-based reasoning. |
| **Fully Reasoned** | + LLM Reasoning Layer (PROPOSE/VERIFY hierarchy inference). |

**Schema-guided completion (SGC)** identifies missing gold classes, relations, and hierarchy edges, asks the LLM which are supported by the corpus, and adds only evidence-backed items. It uses canonical-aware matching, a lenient evidence threshold (so short terms like "EtCO2" and "TCD" aren't rejected), a large token budget (8192 tokens to fit the full v2.0 schema), and truncated-JSON recovery to salvage partial responses. The prompt explicitly requires evidence to be actual corpus text, not ontology URIs or repeated labels. Diagnostic logging records counts at each filter stage.

**LLM Reasoning Layer** runs after built-in cleanup. Two LLM calls — PROPOSE then VERIFY — infer missing hierarchy edges from the clean ontology and gold schema only (no corpus). Uses alias-aware canonical matching so "ICP" correctly resolves to "Intracranial Pressure (ICP)."

**Rule-based Reasoning** acts as a deterministic fallback. It adds gold hierarchy edges wherever both endpoint classes exist, attaches synthetic evidence (containing "is a" to pass hierarchy trigger filters), and optionally prunes orphan classes while preserving gold-aligned ones.

The overall pipeline order: raw merge → vocabulary filter → SGC → cleanup → LLM reasoning → rule-based reasoning → gold-vocabulary filter for evaluation.

**Built-in cleanup** runs before the reasoning layers and handles: deduplication, out-of-scope pruning (compound governance patterns), abstract/broad label pruning (with allowlists), class evidence pruning (with 30+ exempted gold anchors), dangling hierarchy/relation pruning, and axiom constraint enforcement (17 forbidden hierarchy pairs, 8 allowed relation types).

### 3.5.5 Ontology Construction and Validation

Chunk-level extractions are merged by canonical key for classes and by `(label, domain, range)` for relations. Evidence is required for everything. A canonical alias map resolves abbreviations to their canonical labels before deduplication — so "ICP", "mean ICP", and "Intracranial Pressure (ICP)" all merge correctly. Stratum information (core, governance, or provenance) is stored per entity, and provenance routing respects clinical-only mode when the scope filter is on.

Neuro-ICU axioms define forbidden hierarchy type pairs and permitted relation types. Violations get written to `axiom_violations.json`. Structural integrity is checked, and the ontology is exported to JSON with evidence, aliases, provenance, and stratum — with raw and gold-restricted versions saved as needed.

### 3.5.6 Evaluation

The generated ontology is compared against the BrainIT v2.0 gold standard (72 classes, 16 relations, 57 hierarchy edges).

**Class evaluation** — coverage, precision, recall across overall, clinical-only, and by-stratum (core/governance/provenance) views. Extraction-only metrics are captured before any improvement stage to isolate the raw LLM contribution. Per-stage ablation records counts and metrics after each pipeline stage: extraction, after SGC, after cleanup, after LLM reasoning, after rule-based, and after gold filtering. All stored in `metrics["by_stage"]` and formatted in `summary.txt`.

**Relation evaluation** — label-level precision and recall via alias-aware matching, with per-gold-relation breakdown and extraction-only metrics for ablation.

**Hierarchy evaluation** — edge-level precision and recall using normalised key matching.

**Error taxonomy** — hallucinations, schema violations, omissions, and plausible-but-unmatched outputs (clinically reasonable but not in the gold ontology). Optional gold-vocabulary-only mode restricts the generated ontology to gold items before metrics are computed — useful as a recall control, though precision becomes 100% by design.

**Qualitative inspection** — `hallucinated_classes.json`, LLM reasoning patch files, axiom violation reports, detailed `summary.txt`, and an interactive ontology graph visualisation showing classes, relations, and hierarchy using Cytoscape.js.

## 3.6 Tools and Technology Stack

The framework is built in Python 3.10+. Here's the stack:

- **LLM extraction** — OpenAI GPT-4o-mini (default), Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek chat.
- **LLM reasoning** — OpenAI GPT-4o-mini (default), DeepSeek Reasoner R1 (alternative).
- **Retrieval** — TF-IDF via scikit-learn, cosine similarity, MMR-based example selection (λ=0.7).
- **Medical NER** — ScispaCy `en_ner_bc5cdr_md` (diseases/chemicals from BC5CDR corpus), with fallbacks to other ScispaCy and Med-spaCy models.
- **Ontology representation** — JSON internally; gold standard in OWL/RDF Turtle, loaded via `rdflib` with multi-parent hierarchy support.
- **Web interface** — Flask + Jinja2 + vanilla JavaScript. Cytoscape.js for ontology graph visualisation.
- **Evaluation** — custom alignment code in `src/evaluation/` with error analysis, per-stage ablation, TF-IDF semantic matching (threshold 0.55, one-to-one), and synonym expansion.
- **Testing** — pytest.

## 3.7 Delivered Outputs

The project delivers:

- **A reproducible pipeline** with four progressive modes (Strict → Guided → Schema-Completed → Fully Reasoned).
- **Generated Neuro-ICU ontology artefacts** aligned with the BrainIT gold standard — `ontology.json` plus raw and restricted variants.
- **Comprehensive evaluation reports** per run — `metrics.json` (with coverage, precision, recall, extraction-only, clinical-only, by-stratum, per-stage ablation, relation/hierarchy metrics), `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`, `axiom_violations.json`, and `sgc_diagnostic.json`.
- **Human-readable summary** — `summary.txt` with timestamp, input papers, pipeline mode, evaluation settings, extraction-only baseline comparison, per-stage ablation table, and all metrics.
- **Full traceability** — `metadata.json` records run configuration, input paper list (filenames, paths), environment, and code version.
- **Web interface** — run, compare, and browse experiments with standardised labels (`Strategy - PipelineMode - LLM - EvalSettings - Advanced`), interactive ontology graph, and multi-run comparison.
- **Comparative analysis** across three prompting strategies (Zero-Shot, One-Shot, Few-Shot), multiple LLM providers, and four pipeline modes.
- **Insight into LLM limitations** — hallucination patterns, anchoring bias in few-shot learning, cascading pipeline errors, and the trade-off between schema guidance and corpus grounding.
