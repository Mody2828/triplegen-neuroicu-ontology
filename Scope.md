# Project Scope

## Project Aim & Context

This project designed, implemented, and evaluated an LLM-driven ontology engineering framework for the neurointensive care (Neuro-ICU) domain. Using BrainIT consortium publications and related Neuro-ICU literature, the framework investigates the extent to which large language models (LLMs) can reconstruct and align with an expert-designed domain ontology under controlled prompting and evaluation conditions.

Generated ontologies are evaluated against a BrainIT gold standard, enabling rigorous analysis of alignment quality, failure modes, and mitigation strategies in a low-resource, safety-critical clinical domain.

## Literature-Driven Motivation

Prior work demonstrates that while LLMs can extract structured knowledge from text, performance in ontology engineering tasks depends heavily on prompt constraints, ontology guidance, and evaluation methodology. Retrieval-augmented few-shot prompting improves relevance but does not sufficiently reduce hallucinations in specialised domains. Studies show that ontology-augmented prompting, phased extraction, and human-in-the-loop validation are essential to improve reliability and alignment.

Neuro-ICU literature further highlights that clinical knowledge is inherently context-dependent, relying on provenance, measurement configuration, validation status, and temporal structure — factors often omitted by unconstrained LLM outputs. These findings motivated a **hybrid symbolic–LLM framework** rather than a fully automated approach.

## Novelty

This project evaluates LLM-driven ontology engineering in a **low-resource clinical domain** using a clinician-designed gold standard ontology, which differs from prior work relying on well-established biomedical ontologies. It contributes empirical insight into how ontology constraints, retrieval strategies, phased prompting, and a four-stage progressive pipeline affect hallucination rates, structural alignment, and contextual completeness in specialised medical domains.

## Research Objectives

- Assess the ability of LLMs to reconstruct BrainIT ontology concepts, relations, and hierarchy from domain literature.
- Compare baseline, retrieval-augmented, and ontology-guided prompting strategies across multiple LLM providers.
- Analyse common failure modes, including hallucinations, scope drift, and structural inconsistencies.
- Evaluate how ontology constraints, phased extraction, and progressive post-processing improvements affect alignment in a low-resource domain.

## Delivered Methodology Pipeline

The framework implements a modular, reproducible pipeline with the following stages:

### 1. Corpus Ingestion & Preprocessing

- **Input:** BrainIT publications and Neuro-ICU literature (PDF/text), loaded from paste, upload, or default corpus. Multi-file uploads are supported; the corpus directory is cleared when switching to default/paste so only the intended input is processed.
- **Load pipeline:** strip control characters → normalize (hyphenation fix, symbol normalization, repeated header/footer removal) → optional clinical scope filter.
- **Clinical scope filter (optional):** When enabled, document-level paragraph/line filtering using compound-phrase blacklists, then chunk-level dual-score routing (admin_score / clinical_score). Clinical dataset sections (Monitoring, ICU Management, Secondary Insult Treatment) are always retained; governance sections are filtered out. **Scope filter must be set explicitly** when a gold standard is loaded (benchmark mode) so UI and CLI runs are comparable.
- **Provenance:** Each document and chunk carries source, section, and paragraph metadata. PDF documents additionally carry page-level segments.

### 2. Candidate Knowledge Identification

- **Scope filter** (`src/corpus/scope_filter.py`): Lightweight clinical-vs-governance scoring using weighted clinical term dictionaries (strong: 1.0; broad: 0.5). "Monitoring" and BrainIT-specific terms (ventilation, ICP, CPP, MAP, GCS, SaO2, fluids, nutrition, condition, etc.) are weighted as strong terms.
- **Vocabulary guardrails (Strict mode and above):** Gold class labels and paper-wording relation labels are injected into prompts to constrain generation. CamelCase gold relation labels are deliberately excluded to avoid ambiguity. Fallback core vocabulary includes the full v2.0 gold class set (72 classes) when no gold file is loaded.
- **Medical NER anchor (Guided mode and above):** ScispaCy (`en_ner_bc5cdr_md`) extracts biomedical entities and injects them as "Suggested concepts" in the prompt. This is a **built-in part of Guided Extraction mode** (Mode 2) and cascades to Schema-Completed and Fully Reasoned; it is no longer an Advanced/experimental toggle.
- **Candidate terms (Guided mode and above):** Broader noun phrases from the chunk are extracted and injected as hints for the LLM.

### 3. Prompting & Knowledge Extraction

**Four active** prompting strategies are exposed in the UI, using Maximal Marginal Relevance (MMR) for few-shot example selection:

| Strategy | ID | Description |
|----------|----|-------------|
| Zero-Shot | `baseline` | No examples; LLM extracts classes, relations, and hierarchy from text only. |
| One-Shot | `one_shot` | One MMR-selected concept example per chunk. Optional hierarchy sub-call when chunk has hierarchy cues and ≥2 allowed-class labels. |
| Few-Shot | `phased_3step` | Phase 1: 3 concept examples → classes. Phase 2: 3 relation examples → relations only. Phase 3: 3 hierarchy examples → hierarchy only (where lexical cues exist). Phase 2 and Phase 3 outputs vocabulary-filtered before merge. Three LLM calls per chunk. |


Task-specific example pools: `pool_strict_concepts.json`, `pool_strict_relations.json`, `pool_strict_hierarchy.json`. See `docs/task_pool_wiring.md` for wiring.

### 4. Progressive Pipeline Modes (Phased Ontology Reconstruction)

The framework exposes **four pipeline modes** that progressively add features. Each mode is a superset of the previous:

| Mode | Enabled features |
|------|------------------|
| **Strict** | Prompt vocabulary guardrails; evaluation restricted to gold vocabulary. No NER, no post-processing. |
| **Guided** | Strict + Medical NER anchor + candidate term injection. |
| **Schema-Completed** | Guided + Schema-guided completion (LLM gap-filling from gold schema) + Rule-based reasoning layer (schema completion + orphan pruning). |
| **Fully Reasoned** | Schema-Completed + LLM Reasoning Layer (PROPOSE/VERIFY hierarchy inference on the clean ontology). |

**Schema-guided completion:** Identifies missing gold classes, relations, and hierarchy; asks the LLM which are supported by the corpus; adds only corpus-evidenced items. Uses canonical-aware matching.

**LLM Reasoning Layer:** Runs **after** built-in cleanup. Two LLM calls (PROPOSE → VERIFY) infer missing superclass relations from the current ontology and gold schema only (no corpus). Uses alias-aware canonical normalization (e.g. "ICP" → "Intracranial Pressure (ICP)").

**Rule-based Reasoning Layer:** Runs after LLM Reasoning. Adds gold hierarchy edges where both endpoints exist; adds synthetic evidence so edges pass evaluation. Optionally prunes orphan classes; gold-aligned classes are preserved.

Pipeline order: **Raw merge → [vocab filter] → SGC → Cleanup → LLM Reasoning → Rule-based → [gold-vocab filter for eval].**

### 5. Ontology Construction & Validation

- **Merge and build** (`src/ontology/build.py`): Chunk-level extractions merged by canonical key (classes) and (label, domain, range) (relations). Evidence is required for all elements. Canonical alias map resolves abbreviations to canonical labels before deduplication. Stratum (core/governance/provenance) stored per entity. Provenance routing respects clinical-only mode (scope filter on).
- **Built-in cleanup** (always on when gold is available): Runs **before** the LLM Reasoning Layer. Multi-step pipeline: dedupe; out-of-scope pruning (compound patterns only); abstract/broad label pruning with extended allowlists; class evidence pruning (with exemptions for gold structural anchors); hierarchy dangling-endpoint and fragment pruning; relation domain/range and evidence pruning; axiom constraint enforcement. See `src/ontology/reasoner.py` and `src/ontology/neuro_axioms.py`.
- **Neuro-ICU axioms** (`src/ontology/neuro_axioms.py`): Forbidden hierarchy type pairs and allowed relation types per (domain_type, range_type). Violations written to `evaluation/axiom_violations.json`.
- **Validation:** `src/ontology/validate.py` checks structural integrity of the exported ontology.
- **Export:** Ontology exported to JSON with `evidence`, `aliases`, `provenance`, and `stratum`. Raw and gold-restricted versions saved as needed.

### 6. Evaluation

Evaluation compares the generated ontology against the BrainIT gold standard:

**Quantitative metrics (classes):**
- Coverage, precision, recall (overall, clinical-only, by stratum: core/governance/provenance).
- **Extraction-only:** metrics captured before any improvement step — isolates raw LLM contribution.
- **Per-stage ablation:** metrics (n_classes, n_relations, n_hierarchy, coverage, precision, recall) after each stage: extraction → after_sgc → after_cleanup → after_llm_reasoning → after_rule_based → after_gold_filter. Stored in `metrics["by_stage"]` and shown in `summary.txt`.

**Quantitative metrics (relations):**
- Label-level precision and recall via `RELATION_ALIASES_CORE` mapping.
- `per_gold_relation`: which gold relations were recovered.
- Extraction-only relation metrics for ablation.

**Quantitative metrics (hierarchy):**
- Hierarchy edge counts, hierarchy coverage; structural metrics.

**Error taxonomy:**
- Hallucinations (generated, unmatched to gold).
- Schema violations.
- Omissions (gold classes/relations not recovered).
- Plausible-out-of-ontology (generated but clinically reasonable).

**Gold-vocabulary-only evaluation (optional):** Restricts generated ontology to gold-aligned items before metric computation. Useful as a recall control; precision becomes 100% by design.

Qualitative inspection is supported via `hallucinated_classes.json`, LLM reasoning patch files, axiom violations, and the detailed `summary.txt`. **Ontology graph visualization** is available in the UI (interactive graph of classes, relations, and hierarchy).

## Tools & Technology Stack

| Component | Technology |
|-----------|------------|
| Programming language | Python 3.10+ |
| LLM access (extraction) | OpenAI GPT-4o-mini (default), Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek (chat) |
| LLM access (reasoning) | OpenAI GPT-4o-mini (default), DeepSeek Reasoner R1 (alternative) |
| Retrieval & embeddings | TF-IDF (scikit-learn) with cosine similarity and MMR-based retrieval |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` (Guided mode and above) |
| Ontology representation | Internal JSON model; gold standard in OWL/RDF (Turtle) |
| Ontology processing | rdflib (load TTL gold standard; multi-parent hierarchy supported) |
| Pipeline orchestration | Python scripts + Flask-based web interface |
| Web UI | Flask, Jinja2 templates, vanilla JavaScript; Cytoscape.js for ontology graph |
| Plotting & figures | matplotlib, seaborn |
| Testing | pytest |
| Evaluation | Custom alignment (`src/evaluation/`), error analysis, per-stage ablation, semantic matching (TF-IDF) for class alignment |

## Delivered Outputs

- **Reproducible LLM-driven ontology engineering pipeline** with a four-mode progressive abstraction (Strict → Guided → Schema-Completed → Fully Reasoned).
- **Generated Neuro-ICU ontology artefacts** aligned with the BrainIT gold standard: `ontology.json`, raw/restricted variants as configured.
- **Comprehensive evaluation reports** per run: `metrics.json` (coverage/precision/recall, extraction-only, clinical-only, by-stratum, **per-stage ablation**, relation and hierarchy metrics), `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`, `axiom_violations.json`.
- **Enhanced human-readable run summary** (`summary.txt`): timestamp, **input paper names**, pipeline mode, evaluation settings, extraction-only baseline comparison, per-stage ablation table, relation and hierarchy metrics.
- **Traceable metadata** (`metadata.json`): run configuration, input paper list (filenames/paths), environment, code version.
- **Web UI** for running, comparing, and browsing experiments — standardised run label (`Strategy - PipelineMode - LLM - EvalSettings - Advanced`), **interactive ontology graph** (classes, relations, hierarchy), and comparison of multiple runs.
- **Comparative analysis** of four prompting strategies (Zero-Shot, One-Shot, Few-Shot, Few-Shot III) across multiple LLM providers and four pipeline modes.
- **Critical discussion** of LLM strengths and limitations: hallucination patterns, cognitive priming (Medical NER), cascading pipeline errors, and the trade-off between schema guidance and corpus grounding.
