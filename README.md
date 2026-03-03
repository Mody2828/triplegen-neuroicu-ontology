# TripleGen: LLM-Driven Ontology Engineering for Neuro-ICU

## Overview

**TripleGen** is a research framework that uses Large Language Models (LLMs) to build ontologies from Neuro-ICU literature. It takes BrainIT consortium papers as input, extracts ontology elements such as classes, relations, and hierarchy, and evaluates the resulting ontology against a clinician-designed gold standard.

The central research question is whether LLMs can reconstruct a high-quality domain ontology in a low-resource clinical setting. TripleGen investigates this through controlled prompting strategies, progressive post-processing, and rigorous evaluation, measuring not only what the LLM extracts correctly, but also where and why it fails.

## Research Objectives

1. **Assess LLM capability** — determine whether LLMs can reconstruct BrainIT ontology concepts, relations, and hierarchy from Neuro-ICU literature.
2. **Compare prompting strategies** — evaluate how Zero-Shot, One-Shot, and Few-Shot approaches, including phased decomposition and anti-anchoring design, perform across multiple LLM providers.
3. **Compare source-document effects** — investigate how differences between input papers influence ontology extraction quality, including concept coverage, relation recovery, hierarchy reconstruction, and error patterns.
4. **Analyse failure modes** — identify common errors such as hallucinations, scope drift, anchoring bias, and structural inconsistencies.
5. **Evaluate pipeline components** — assess how vocabulary guardrails, Medical NER, schema-guided completion, and LLM reasoning layers influence alignment with the gold standard.

## What Makes This Different

Much prior work in biomedical ontology engineering has focused on well-established ontologies such as **SNOMED CT** and **Gene Ontology**. TripleGen instead uses a **clinician-designed gold standard** from the BrainIT consortium: a smaller, domain-specific ontology that is closer to what clinicians may construct in practice. This makes the framework particularly useful for studying LLM ontology engineering in genuinely low-resource specialist settings.

## Key Features

- **Three prompting strategies**: Zero-Shot (baseline), One-Shot (single MMR example), and Few-Shot (three-phase extraction with dedicated example pools per phase and anti-anchoring instructions).
- **Four progressive pipeline modes**: Strict → Guided → Schema-Completed → Fully Reasoned, with each mode adding further post-processing so that the contribution of each component can be observed.
- **Medical NER anchor**: ScispaCy biomedical named entity recognition integrated into Guided mode and above, giving the LLM a structured starting point for clinical terminology.
- **Schema-guided completion**: LLM-based gap filling against the gold schema, restricted to items supported by corpus evidence.
- **LLM Reasoning Layer**: a two-step PROPOSE/VERIFY mechanism for hierarchy inference.
- **Per-stage ablation**: metrics captured after each major pipeline stage to show exactly where improvements or degradations occur.
- **Multi-LLM support**: OpenAI GPT-4o-mini, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, and DeepSeek.
- **Web interface**: a Flask-based application for running experiments, comparing runs, and visualising ontologies as interactive graphs using Cytoscape.js.

## Documents in This Folder

| Document | Description |
|----------|-------------|
| [**implementation_chapter.md**](implementation_chapter.md) | Full design and implementation details, including pipeline architecture, prompting strategies, pipeline modes, post-processing, evaluation, and gold standard handling. |
| [**Scope.md**](Scope.md) | Project scope, research objectives, delivered methodology, technology stack, and outputs. |
| [**TripleGen_WepApp.md**](TripleGen_WepApp.md) | Visual walkthrough of the web application with step-by-step usage instructions and screenshot placeholders. |

## Getting Started

- **How TripleGen works** — read [implementation_chapter.md](implementation_chapter.md). This document explains the full pipeline from corpus ingestion through to evaluation, including prompting strategies, pipeline modes, and reasoning layers.
- **Project scope and objectives** — read [Scope.md](Scope.md). This document outlines what is in scope, the research questions, and the delivered outputs.
- **Using the web app** — read [TripleGen_WepApp.md](TripleGen_WepApp.md). This document explains how to run experiments, compare results, and interpret the ontology graph.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Extraction LLMs | OpenAI GPT-4o-mini, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq, Hugging Face, DeepSeek |
| Reasoning LLMs | OpenAI GPT-4o-mini, DeepSeek Reasoner R1 |
| Retrieval | TF-IDF + cosine similarity + MMR (λ=0.7) |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` |
| Ontology | JSON (internal), OWL/RDF Turtle (gold standard), rdflib |
| Web UI | Flask + Jinja2 + vanilla JavaScript + Cytoscape.js |
| Evaluation | Custom alignment with TF-IDF semantic matching, per-stage ablation, and error taxonomy |

---