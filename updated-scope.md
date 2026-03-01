# Project Scope (Delivered)

## Project Aim & Context

This project designed, implemented, and evaluated an LLM-driven ontology engineering framework for the neurointensive care (Neuro-ICU) domain. Using BrainIT consortium publications and related Neuro-ICU literature, the framework investigates the extent to which large language models (LLMs) can reconstruct and align with an expert-designed domain ontology under controlled prompting and evaluation conditions.

Generated ontologies are evaluated against an unpublished BrainIT gold standard (ontology and schema), enabling rigorous analysis of alignment quality, failure modes, and mitigation strategies in a low-resource, safety-critical clinical domain.

## Literature-Driven Motivation

Prior work demonstrates that while LLMs can extract structured knowledge from text, performance in ontology engineering tasks depends heavily on prompt constraints, ontology guidance, and evaluation methodology. Retrieval-augmented few-shot prompting improves relevance but does not sufficiently reduce hallucinations in specialised domains. Studies show that ontology-augmented prompting, phased extraction, and human-in-the-loop validation are essential to improve reliability and alignment.

Neuro-ICU literature further highlights that clinical knowledge is inherently context-dependent, relying on provenance, measurement configuration, validation status, and temporal structure — factors often omitted by unconstrained LLM outputs. These findings motivated a **hybrid symbolic–LLM framework** rather than a fully automated approach.

## Novelty

This project evaluates LLM-driven ontology engineering in a **low-resource clinical domain** using a clinician-designed but unpublished gold standard ontology, which differs from prior work relying on well-established biomedical ontologies. It contributes empirical insight into how ontology constraints, retrieval strategies, phased prompting, and a four-stage progressive pipeline affect hallucination rates, structural alignment, and contextual completeness in specialised medical domains.

## Research Objectives

- Assess the ability of LLMs to reconstruct BrainIT ontology concepts and relations from domain literature.
- Compare baseline, retrieval-augmented, and ontology-guided prompting strategies across multiple LLM providers.
- Analyse common failure modes, including hallucinations, scope drift, and structural inconsistencies.
- Evaluate how ontology constraints, phased extraction, and progressive post-processing improvements affect alignment in a low-resource domain.

## Delivered Methodology Pipeline

The framework implements a modular, reproducible pipeline with the following stages:

### 1. Corpus Ingestion & Preprocessing

- **Input:** BrainIT publications and Neuro-ICU literature (PDF/text), loaded from paste, upload, or default corpus.
- **Load pipeline:** strip control characters → normalize (hyphenation fix, symbol normalization, repeated header/footer removal) → optional clinical scope filter.
- **Clinical scope filter (optional):** Document-level paragraph/line filtering using compound-phrase blacklists, then chunk-level dual-score routing (admin_score / clinical_score). Clinical dataset sections (Monitoring, ICU Management, Secondary Insult Treatment) are always retained; governance sections are filtered out.
- **Provenance:** Each document and chunk carries source, section, and paragraph metadata. PDF documents additionally carry page-level segments.

### 2. Candidate Knowledge Identification

- **Scope filter** (`src/corpus/scope_filter.py`): Lightweight clinical-vs-governance scoring using weighted clinical term dictionaries (strong: 1.0; broad: 0.5). "Monitoring" and BrainIT-specific terms (ventilation, ICP, CPP, MAP, GCS, SaO2, etc.) are weighted as strong terms.
- **Vocabulary guardrails (optional):** Gold class labels and paper-wording relation labels are injected into prompts to constrain downstream generation. CamelCase gold relation labels are deliberately excluded to avoid ambiguity.
- **Medical NER anchor (optional/experimental):** ScispaCy (`en_ner_bc5cdr_md`) extracts biomedical entities and injects them as suggested concepts. Empirically shown to sometimes reduce recall via cognitive priming; exposed as an Advanced/Experimental feature with a warning.

### 3. Prompting & Knowledge Extraction

Three active prompting strategies are evaluated, using Maximal Marginal Relevance (MMR) retrieval for few-shot example selection:

| Strategy | ID | Description |
|----------|----|-------------|
| Zero-Shot | `baseline` | No examples; LLM extracts from text only. |
| One-Shot | `one_shot` | One MMR-selected concept example per chunk. |
| Few-Shot | `phased_2step` | Phase 1: 3 concept examples → classes. Phase 2: 3 relation examples → relations/hierarchy. Two LLM calls per chunk. |

Two additional strategies (`simple_fewshot`, `phased_3step`) are retained for backward compatibility with old run data but are hidden from the UI after empirical analysis showed no consistent benefit over the primary Few-Shot strategy.

### 4. Progressive Pipeline Modes (Phased Ontology Reconstruction)

The framework exposes **four pipeline modes** that progressively add post-processing capabilities. Each mode is a superset of the previous:

| Mode | Enabled features |
|------|-----------------|
| **Strict** | Extraction only; no guardrails or improvements |
| **Guided** | + Prompt vocabulary guardrails (gold labels in prompt) |
| **Schema-Completed** | + Schema-guided completion (LLM gap-filling from gold schema) |
| **Fully Reasoned** | + LLM Reasoning Layer + Rule-based Reasoning Layer |

This abstraction directly supports ablation analysis: running the same corpus and strategy across all four modes isolates the contribution of each layer.

**Schema-guided completion:** Identifies missing gold classes and relations; asks the LLM to confirm which are supported by the corpus; adds only corpus-evidenced items.

**LLM Reasoning Layer:** Infers missing superclass relations via a propose/verify LLM flow using the current ontology and gold schema (no corpus). Uses alias-aware canonical normalization to correctly match abbreviations (e.g. "ICP" → "Intracranial Pressure (ICP)").

**Rule-based Reasoning Layer:** Deterministically adds gold hierarchy edges where both endpoint classes exist in the ontology; optionally removes orphan classes.

### 5. Ontology Construction & Validation

- **Merge and build** (`src/ontology/build.py`): Chunk-level extractions merged by canonical key (classes) and (label, domain, range) (relations). Evidence is required for all elements. Canonical alias map (`CANONICAL_ALIAS_MAP`) resolves abbreviations to canonical labels before deduplication. Stratum (core/governance/provenance) stored per entity.
- **Built-in cleanup** (always on when gold is available): Nine-step deterministic pipeline — dedupe, out-of-scope pruning (compound patterns only), abstract label pruning, broad contextual pruning, evidence pruning, relation domain/range pruning, relation evidence pruning, hierarchy fragment pruning, axiom constraint enforcement.
- **Neuro-ICU axioms** (`src/ontology/neuro_axioms.py`): Forbidden hierarchy type pairs and canonical allowed relation labels per (domain_type, range_type). Violations written to `evaluation/axiom_violations.json`.
- **Validation:** `src/ontology/validate.py` checks structural integrity of the exported ontology.
- **Export:** Ontology exported to JSON with `evidence`, `aliases`, `provenance`, and `stratum` fields. Raw (pre-restriction) and restricted (gold-vocab only) versions saved separately.

### 6. Evaluation

Evaluation compares the generated ontology against the BrainIT gold standard:

**Quantitative metrics (classes):**
- Coverage, Precision, Recall (overall, clinical-only, by stratum: core/governance/provenance).
- **Extraction-only:** metrics captured before any improvement step — isolates raw LLM contribution.
- **Per-stage ablation:** compact metrics (n_classes, n_relations, n_hierarchy, coverage, precision, recall) after each pipeline stage (extraction → +SGC → +LLM Reasoning → +Cleanup → +Rule-based). Written to `metrics["by_stage"]`.

**Quantitative metrics (relations):**
- Label-level precision and recall via `RELATION_ALIASES_CORE` mapping.
- `per_gold_relation`: per-relation boolean showing which of the 4 BrainIT gold relations were recovered.
- Extraction-only relation metrics also captured for ablation.

**Error taxonomy:**
- Hallucinations (generated, unmatched to gold).
- Schema violations.
- Omissions (gold classes not recovered).
- Plausible-out-of-ontology (generated but clinically reasonable).

**Structural metrics:** hierarchy_edges, hierarchy_coverage, relation_domain_range_rate.

**Gold-vocabulary-only evaluation (optional):** Restricts generated ontology to gold-aligned items before metric computation. Useful as a recall control; precision becomes 100% by design.

Qualitative inspection is supported via `hallucinated_classes.json`, `llm_reasoning_patch_proposed.json`, and the detailed `summary.txt`.

## Tools & Technology Stack

| Component | Technology |
|-----------|------------|
| Programming language | Python 3.11 |
| LLM access (extraction) | OpenAI GPT-4o-mini (default), Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek (chat) |
| LLM access (reasoning) | OpenAI GPT-4o-mini (default), DeepSeek Reasoner R1 (alternative) |
| Retrieval & embeddings | TF-IDF (scikit-learn) with cosine similarity and MMR-based retrieval |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` (optional, experimental) |
| Ontology representation | Internal JSON model; gold standard in OWL/RDF (Turtle) |
| Ontology processing | rdflib (load TTL gold standard) |
| Pipeline orchestration | Python scripts + Flask-based web interface |
| Web UI | Flask, Jinja2 templates, vanilla JavaScript |
| Plotting & figures | matplotlib, seaborn |
| Testing | pytest |
| Evaluation | Custom alignment (`src/evaluation/`), error analysis, per-stage ablation |

## Delivered Outputs

- **Reproducible LLM-driven ontology engineering pipeline** with a four-mode progressive abstraction (Strict → Guided → Schema-Completed → Fully Reasoned).
- **Generated Neuro-ICU ontology artefacts** aligned with the BrainIT gold standard: `ontology.json`, `ontology_raw.json`, `ontology_restricted.json`.
- **Comprehensive evaluation reports** per run: `metrics.json` (coverage/precision/recall including extraction-only baseline, clinical-only, by-stratum, per-stage ablation, relation precision/recall), `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`.
- **Enhanced human-readable run summary** (`summary.txt`): timestamp, input paper names, pipeline mode, evaluation settings, extraction-only baseline comparison, per-stage ablation table, relation metrics.
- **Traceable metadata** (`metadata.json`): run configuration, input paper list (filenames + paths), environment, code version.
- **Web UI** for running, comparing, and browsing experiments — with a standardised 5-part run label (`Strategy - PipelineMode - LLM - EvalSettings - Advanced`) consistent across Python and JavaScript.
- **Comparative analysis** of three prompting strategies (Zero-Shot, One-Shot, Few-Shot) across six LLM providers and four pipeline modes.
- **Critical discussion** of LLM strengths and limitations: hallucination patterns, cognitive priming effects (Medical NER), cascading pipeline errors, and the trade-off between schema guidance and corpus grounding.
