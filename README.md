# TripleGen: LLM-Driven Ontology Engineering for Neuro-ICU

## Overview

**TripleGen** is a research framework that uses Large Language Models to build ontologies from Neuro-ICU literature. It takes BrainIT consortium papers as input, extracts ontology elements (classes, relations, hierarchy), and evaluates the result against a clinician-designed gold standard.

The core question is straightforward: can LLMs reconstruct a high-quality domain ontology in a low-resource clinical setting where expert-built ontologies are hard to come by? TripleGen explores this through controlled prompting strategies, progressive post-processing, and rigorous evaluation — measuring not just what the LLM gets right, but where and why it fails.

## Research Objectives

1. **Assess LLM capability** — can LLMs reconstruct BrainIT ontology concepts, relations, and hierarchy from domain literature?
2. **Compare prompting strategies** — how do Zero-Shot, One-Shot, and Few-Shot (three-phase decomposition with anti-anchoring) approaches compare across multiple LLM providers?
3. **Analyse failure modes** — hallucinations, scope drift, anchoring bias, structural inconsistencies.
4. **Evaluate pipeline components** — how do vocabulary guardrails, Medical NER, schema-guided completion, and LLM reasoning layers affect alignment?

## What Makes This Different

Most prior work evaluates LLM ontology engineering against well-established biomedical ontologies (SNOMED, Gene Ontology, etc.). TripleGen uses a **clinician-designed gold standard** from the BrainIT consortium — a smaller, domain-specific ontology that's closer to what clinicians actually build in practice. This gives practical insight into LLM performance in genuinely low-resource settings.

## Key Features

- **Three prompting strategies**: Zero-Shot (baseline), One-Shot (single MMR example), and Few-Shot (three-phase extraction with dedicated example pools per phase and anti-anchoring instructions).
- **Four progressive pipeline modes**: Strict → Guided → Schema-Completed → Fully Reasoned — each adding more post-processing to show what each component contributes.
- **Medical NER anchor**: ScispaCy biomedical entity recognition built into Guided mode and above, giving the LLM a head start on clinical terms.
- **Schema-guided completion**: LLM-based gap-filling from the gold schema, adding only corpus-evidenced items.
- **LLM Reasoning Layer**: PROPOSE/VERIFY hierarchy inference using two separate LLM calls.
- **Per-stage ablation**: metrics captured after every pipeline stage so you can see exactly where value (or damage) comes from.
- **Multi-LLM support**: OpenAI GPT-4o-mini, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek.
- **Web interface**: Flask app for running experiments, comparing runs, and visualising ontologies as interactive graphs (Cytoscape.js).

## Documents in This Folder

| Document | Description |
|----------|-------------|
| [**implementation_chapter.md**](implementation_chapter.md) | Full design and implementation: pipeline architecture, prompting strategies, pipeline modes, post-processing, evaluation, gold standard. Start here to understand how TripleGen works. |
| [**Scope.md**](Scope.md) | Project scope, research objectives, delivered methodology, technology stack, and outputs. |
| [**TripleGen_WepApp.md**](TripleGen_WepApp.md) | Visual walkthrough of the web application with step-by-step usage instructions and screenshot placeholders. |

## Getting Started

- **How TripleGen works** — read [implementation_chapter.md](implementation_chapter.md). It covers the full pipeline from corpus ingestion through evaluation, including prompting strategies, pipeline modes, and the reasoning layers.
- **Project scope and objectives** — read [Scope.md](Scope.md). It covers what's in scope, the research questions, and the delivered outputs.
- **Using the web app** — read [TripleGen_WepApp.md](TripleGen_WepApp.md). It walks through running experiments, comparing results, and interpreting the ontology graph.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Extraction LLMs | OpenAI GPT-4o-mini, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq, Hugging Face, DeepSeek |
| Reasoning LLMs | OpenAI GPT-4o-mini, DeepSeek Reasoner R1 |
| Retrieval | TF-IDF + cosine similarity + MMR (λ=0.7) |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` |
| Ontology | JSON (internal), OWL/RDF Turtle (gold standard), rdflib |
| Web UI | Flask + Jinja2 + vanilla JS + Cytoscape.js |
| Evaluation | Custom alignment with TF-IDF semantic matching, per-stage ablation, error taxonomy |

---
