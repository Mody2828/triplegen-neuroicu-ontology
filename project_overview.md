# TripleGen — Project Overview

## Overview

**TripleGen** is a research framework that uses Large Language Models (LLMs) to build ontologies from Neuro-ICU literature. It takes BrainIT consortium publications and related neurointensive care literature as input, extracts ontology elements such as classes, relations, and hierarchy, and evaluates the result against a reference ontology.

The central research question is whether LLMs can reconstruct a high-quality domain ontology in a low-resource clinical setting. TripleGen investigates this through controlled prompting strategies, three progressive pipeline modes, and rigorous evaluation — measuring not only what the LLM extracts correctly, but also where and why it fails.

The Neuro-ICU domain presents particular challenges. Clinical knowledge is inherently context-dependent, relying on provenance, measurement configuration, validation status, and temporal structure. These dimensions are frequently missed by unconstrained LLM outputs, which is why TripleGen adopts a hybrid symbolic-LLM framework rather than a fully automated approach.

## Research Context and Novelty

Prior work has shown that while LLMs can extract structured knowledge from text, their performance on ontology engineering tasks depends heavily on prompt constraints, ontology guidance, and evaluation design. Retrieval-augmented few-shot prompting improves relevance but does not sufficiently reduce hallucinations in specialised domains on its own. Research also shows that ontology-augmented prompting, phased extraction, and human-in-the-loop validation are important for reliability and alignment.

Most prior work in biomedical ontology engineering has focused on well-established ontologies such as **SNOMED CT** and **Gene Ontology**. TripleGen instead uses a **clinician-designed reference ontology** from the BrainIT consortium — a smaller, domain-specific ontology closer to what clinicians construct in practice. This makes the framework particularly useful for studying LLM ontology engineering in genuinely low-resource specialist settings, where schema constraints, retrieval strategies, phased prompting, and progressive post-processing all have measurable effects on hallucination rates, structural alignment, and contextual completeness.

## Research Objectives

1. **Assess LLM capability** — determine whether LLMs can reconstruct BrainIT ontology concepts, relations, and hierarchy from Neuro-ICU literature.
2. **Compare prompting strategies** — evaluate how Zero-Shot, One-Shot, and Few-Shot approaches, including phased decomposition and anti-anchoring design, perform across multiple LLM providers.
3. **Compare source-document effects** — investigate how differences between input papers influence ontology extraction quality, including concept coverage, relation recovery, hierarchy reconstruction, and error patterns.
4. **Analyse failure modes** — identify common errors such as hallucinations, scope drift, anchoring bias, and structural inconsistencies.
5. **Evaluate pipeline components** — assess how vocabulary guardrails, Medical NER, schema-guided completion, and rule-based reasoning influence alignment with the reference ontology.

## Key Features

- **Three prompting strategies**: Zero-Shot (baseline), One-Shot (single MMR example), and Few-Shot (three-phase extraction with dedicated example pools per phase and anti-anchoring instructions).
- **Three progressive pipeline modes**: Strict → Guided → Schema-Completed, with each mode adding further post-processing so the contribution of each component can be observed.
- **Controlled experiment page**: 19 individual pipeline toggles for fine-grained ablation, grouped into Preprocessing, Prompt Injection, and Post-Processing.
- **Cross-paper Ontology Engineering page**: takes the output of multiple prior runs, merges them into a single source ontology, semantically clusters the union (default 25 clusters), and asks the LLM to enrich each cluster as a separate call. The reconstruction enforces a closed relation vocabulary self-seeded from the merged source (no gold leakage), forbids the LLM from inventing new classes, and passes hierarchy through directly from the source seed.
- **Built-in BrainIT paper selector**: dropdown with 11 pre-loaded papers for quick experiment setup without manual file handling.
- **Medical NER anchor**: ScispaCy biomedical named entity recognition integrated into Guided mode and above, giving the LLM a structured starting point for clinical terminology.
- **Schema-guided completion**: LLM-based gap-filling against the gold schema, restricted to items supported by corpus evidence.
- **Per-stage ablation**: metrics captured after each major pipeline stage to show exactly where improvements or degradations occur.
- **Multi-LLM support**: OpenAI GPT-4o-mini, OpenAI GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, and DeepSeek.
- **Configurable reasoning LLM**: OpenAI GPT-4o-mini (default) or DeepSeek Reasoner R1 for schema-guided completion.
- **Web interface**: a Flask-based application for running experiments, comparing runs, and visualising ontologies as interactive graphs using Cytoscape.js.

## Methodology Pipeline

### Corpus Ingestion and Preprocessing

Input is BrainIT publications and Neuro-ICU literature supplied as pasted text, uploaded files (`.txt` / `.pdf`), or selected from the built-in BrainIT paper dropdown (11 papers). Multi-file uploads are supported.

The load sequence for each document is: strip control characters → normalise (citation stripping, abbreviation expansion) → optionally apply the clinical scope filter. Normalisation handles hyphenation repair, Unicode symbol normalisation, and removal of repeated headers and footers. PDF documents also retain page-level segments for header/footer detection.

The **clinical scope filter** is an important preprocessing step. BrainIT papers mix clinical content with governance, organisational, and administrative material. The filter operates at two levels: document-level paragraph and line filtering using compound-phrase blacklists, and chunk-level dual-score routing based on administrative and clinical scores. Clinical sections (Monitoring, ICU Management, Secondary Insult Treatment) are always kept; governance sections are filtered out.

**Section-aware chunking** classifies chunks as high/medium/low/skip priority based on their content type, and injects section context into LLM prompts. Reference chunks are skipped entirely. Chunk size is optimised at target 2000 tokens, hard max 3200, min 400, overlap 200.

Provenance is preserved throughout — each document and chunk carries source, section, and paragraph metadata.

### Candidate Knowledge Identification

The scope filter performs lightweight clinical-versus-governance scoring using weighted term dictionaries. Strong terms (weight 1.0) include `ICP`, `CPP`, `MAP`, `GCS`, `monitoring`, `ventilation`, `sedation`, `fluids`, and `nutrition`. Broader terms such as `core dataset` and `outcome` receive 0.5.

Vocabulary guardrails are active from Strict mode onwards: gold class labels and paper-wording relation labels are injected into prompts to constrain LLM output. A "leaf-first rule" instructs the LLM to prefer specific clinical labels (e.g. ICP, CPP) over broad containers (e.g. Monitoring Data) when both appear in the text.

Guided mode and above include a **Medical NER anchor**: ScispaCy (`en_ner_bc5cdr_md`) pre-identifies biomedical entities in the text and injects them into the prompt as "Suggested concepts." Broader noun phrases are also injected as candidate terms. If ScispaCy is unavailable, the system falls back through alternative models or proceeds without NER.

### Prompting and Knowledge Extraction

Example selection for few-shot strategies uses **Maximal Marginal Relevance (MMR)** with λ=0.7, balancing relevance to the current chunk against diversity among selected examples.

**Zero-Shot** — no examples; the LLM extracts classes, relations, and hierarchy directly from text using only its instructions. Serves as the control condition.

**One-Shot** — one MMR-selected concept example per chunk. Optionally triggers a hierarchy sub-call when the chunk has hierarchy cues and at least one allowed class label.

**Few-Shot (three-phase, primary strategy)** — breaks extraction into three separate LLM calls:

- **Phase 1 — Class extraction.** Three concept examples (MMR-selected) prompt the LLM to extract classes across eight concept types: clinical conditions, physiological parameters, treatments and interventions, clinical assessments, laboratory values, nursing activities, sensor and data quality concepts, and abstract category classes. A "breadth over caution" directive instructs the model to extract all concepts the text supports, not just those represented in the examples — added because testing showed few-shot examples were anchoring the model too narrowly.
- **Phase 2 — Relation extraction.** Three relation examples prompt for relations. The LLM is also permitted to extract supplementary classes that Phase 1 may have missed (e.g. concepts needed as relation domain or range). Hierarchy is deferred to Phase 3. Output is vocabulary-filtered before merge.
- **Phase 3 — Hierarchy extraction.** Three hierarchy examples prompt for subclass/superclass edges, but only when the chunk contains explicit cues (`"such as"`, `"is a"`, `"type of"`, `"include:"`, etc.) and at least one class has been extracted. When fewer than five classes are available, gold vocabulary labels are injected as valid endpoints. Output is filtered through `filter_hierarchy_to_lexical_cues` before merge.

The few-shot template also includes an explicit **anti-anchoring instruction**: the LLM is told that examples demonstrate output format and evidence style only, and must not limit extraction to the concept types shown in the examples.

### Progressive Pipeline Modes

Three modes progressively layer on functionality. Each mode is a superset of the previous one:

| Mode | What it adds |
|------|-------------|
| **Strict** | Vocabulary guardrails + gold-restricted evaluation. No NER, no post-processing. |
| **Guided** | + Medical NER anchor + candidate term injection. |
| **Schema-Completed** | + Schema-guided completion (LLM gap-filling) + rule-based reasoning. |

Schema-Completed is the default, balancing capability and speed.

The **Controlled Experiment** page bypasses predefined modes entirely, exposing 19 individual toggles so each feature can be tested independently, grouped into Preprocessing, Prompt Injection, and Post-Processing.

### Post-Processing

Post-processing runs after merge in a fixed order: TGC → SGC → built-in cleanup → orphan rescue → LLM CoT refinement → Rule-based Reasoning → gold-vocabulary filter.

**Text-Grounded Completion (TGC)** is a second-pass extraction where the LLM reviews the full document alongside the current ontology to find missed items. It uses a Chain-of-Layer taxonomy approach for structured multi-pass review.

**Schema-guided completion (SGC)** identifies missing gold classes, relations, and hierarchy edges and asks the LLM which are supported by the corpus. It adds only evidence-backed items, using canonical-aware matching, a lenient evidence threshold (to avoid rejecting short terms like `EtCO2` and `TCD`), a large token budget (8192 tokens), and truncated-JSON recovery for partial responses. A `sgc_diagnostic.json` file records counts at each filter stage.

**Built-in cleanup** always runs when gold is available and handles: deduplication, out-of-scope pruning, abstract/broad label pruning, class evidence pruning (with 30+ protected gold anchors), dangling hierarchy and relation pruning, and axiom constraint enforcement (17 forbidden hierarchy pairs, 8 allowed relation types).

**Rule-based Reasoning Layer** acts as a deterministic fallback. It adds gold hierarchy edges wherever both endpoint classes exist, with synthetic evidence strings, and prunes orphan classes while preserving gold-aligned ones.

### Ontology Construction and Validation

Chunk-level extractions are merged by canonical key for classes and by `(label, domain, range)` for relations. A canonical alias map resolves abbreviations to canonical labels before deduplication. Label normalisation includes singularisation, hyphen collapse, title case, and canonical alias mapping. Neuro-ICU axioms define forbidden hierarchy type pairs and permitted relation types; violations are written to `axiom_violations.json`. The ontology is exported to JSON with evidence, aliases, provenance, and stratum, with raw and gold-restricted variants saved as needed.

### Evaluation

The generated ontology is compared against a **BrainIT proxy reference ontology** (93 classes, 18 relations, 76 hierarchy edges). This proxy was constructed from BrainIT publications as a placeholder for pipeline validation while the full BrainIT consortium ontology is being sourced; final evaluation will be re-run against the canonical reference once available.

- **Class evaluation** — coverage, precision, and recall across overall, clinical-only, and by-stratum (core/governance/provenance) views. Extraction-only metrics isolate the raw LLM contribution before any improvement stage.
- **Relation evaluation** — label-level precision and recall via alias-aware matching, with per-gold-relation breakdown.
- **Hierarchy evaluation** — edge-level precision and recall using normalised key matching.
- **Per-stage ablation** — snapshots after each pipeline stage (extraction, after SGC, after cleanup, after rule-based, after gold filter), written to `metrics["by_stage"]` and rendered as a table in `summary.txt`.
- **Error taxonomy** — hallucinations, schema violations, omissions, and plausible-but-unmatched outputs.

## Delivered Outputs

Each run produces a timestamped directory containing:

- **Generated ontology artefacts** — `ontology.json`, `ontology_raw.json`, and a gold-restricted variant.
- **Evaluation reports** — `metrics.json` (coverage, precision, recall, extraction-only, clinical-only, by-stratum, per-stage ablation, relation and hierarchy metrics), `table.csv`, `hallucinated_classes.json`, `improvement_counts.json`, `axiom_violations.json`, and `sgc_diagnostic.json`.
- **Human-readable summary** — `summary.txt` with timestamp, input papers, pipeline mode, evaluation settings, extraction-only baseline, per-stage ablation table, and full class/relation/hierarchy listings.
- **Full traceability** — `metadata.json` recording run configuration, input paper list, environment, and code version.
- **Saved prompts** — per-chunk and per-phase prompts for reproducibility and inspection.

## Web Application

The web interface provides five main pages:

- **Run Experiment** — strategy selection (Zero-Shot, One-Shot, Few-Shot), pipeline mode cards, LLM provider selector with 7 providers, evaluation settings, and an Advanced section for reasoning LLM override.
- **Controlled Experiment** — 19 individual pipeline toggles for fine-grained ablation, with a built-in BrainIT paper selector dropdown (11 papers). Each toggle can be independently enabled or disabled to measure its contribution.
- **Run Comparison** — batch mode with per-run paper selection. Build a list of configurations, run them sequentially with pause/resume/skip controls, and compare metrics side by side with F1, precision, recall charts.
- **Ontology Engineering** — cross-paper reconstruction page. Pick any prior runs, merge them into a single source ontology, semantically cluster the merged classes (default 25 clusters), and ask the LLM to enrich each cluster as a separate call. The reconstruction enforces a closed relation vocabulary self-seeded from the merged source (no gold leakage), forbids new class invention, and passes hierarchy through directly from the source seed. A Compare Metrics modal ranks reconstruction runs by Overall F1 (mean of class/hierarchy/relation F1).
- **Results** — evaluation metrics card, per-stage ablation table, downloadable artifacts, and an interactive ontology graph (Cytoscape.js) with four layout modes, node detail panels, and PNG export.

Run labels follow the format `Strategy - PipelineMode - LLM - EvalSettings - Advanced` and appear consistently across all pages.

For a visual walkthrough with screenshots, see [TripleGen_WepApp.md](TripleGen_WepApp.md).

For a pipeline diagram and stage breakdown, see [pipeline.md](pipeline.md).

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Extraction LLMs | OpenAI GPT-4o-mini, OpenAI GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek |
| Reasoning LLMs | OpenAI GPT-4o-mini, DeepSeek Reasoner R1 |
| Retrieval | TF-IDF + cosine similarity + MMR (λ=0.7) |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` |
| Ontology | JSON (internal), OWL/RDF Turtle (reference ontology), rdflib |
| Web UI | Flask + Jinja2 + vanilla JavaScript + Cytoscape.js |
| Evaluation | Custom alignment with TF-IDF semantic matching (threshold 0.55), per-stage ablation, and error taxonomy |
