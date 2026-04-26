# TripleGen — Project Overview

## Project

**TripleGen** is an LLM-based framework for extracting and assembling clinical ontologies from neurointensive care literature, developed as part of the CMM303 MSci Capstone Project at Robert Gordon University. It targets the **BrainIT** consortium domain — a low-resource clinical setting in which no public open-access ontology currently exists, despite the consortium defining many key concepts across its publications. TripleGen produces a structured ontology output **intended as a draft for expert review**, reducing the manual effort required to bootstrap a domain ontology while keeping final validation with domain specialists.

## Research question

> To what extent can large language models automatically recreate or align with high-quality domain ontologies in a low-resource clinical (Neuro-ICU) setting when guided by evidence-based prompting and qualitative evaluation?

## Motivation and context

Manual ontology engineering is slow because it requires both clinical expertise and specialist knowledge-modelling skills. Recent research has explored LLM-assisted ontology extraction, but most studies evaluate in domains where mature public ontologies already exist, leaving limited evidence for performance in genuinely low-resource clinical settings. The BrainIT consortium curates multi-site traumatic-brain-injury monitoring data across European centres, but its data model is closed and defined only as a database schema. TripleGen investigates whether LLMs can help reduce the knowledge-engineering bottleneck in this kind of setting through controlled prompting strategies, progressive post-processing layers, and rigorous qualitative evaluation.

## Objectives

1. **Design and implement an end-to-end pipeline** that ingests clinical literature and produces a structured ontology through LLM-driven extraction, with saved artefacts at every stage for auditability.
2. **Compare prompting strategies** — Zero-Shot, One-Shot, and phased Few-Shot with MMR-based example selection — across multiple LLM providers and paper groups of increasing scope.
3. **Evaluate pipeline components** through progressive modes (Strict, Guided, Schema-Completed) and per-stage ablation snapshots, isolating the contribution of evidence gating, vocabulary guardrails, Chain-of-Layer construction, schema-guided completion, and rule-based reasoning.
4. **Synthesise across papers**, reconstructing a unified ontology from multiple per-paper runs through a four-stage Ontology Engineering workflow under closed source-seeded constraints.
5. **Analyse failure modes** — including hallucinations, scope drift, anchoring bias, and structural inconsistencies — to understand the practical limits of LLM-based ontology engineering in this setting.

## Design and methodology

### Pipeline architecture

TripleGen runs as a 10-step pipeline (full detail in [pipeline.md](pipeline.md)) organised into four broad phases:

- **Input and preprocessing** — corpus ingestion, document-level normalisation (citation stripping, abbreviation expansion, hyphenation repair), section-aware semantic chunking, and a chunk-level scope filter that drops administrative content.
- **LLM extraction loop** — for every chunk that passes scoping, the framework assembles a Chain-of-Thought prompt with a strict output schema, calls the extraction LLM, and parses the response under **strict evidence gating**: every extracted class, relation, and hierarchy edge must carry a verbatim quote from the source, otherwise it is rejected.
- **Post-processing** — Text-Grounded Completion with Chain-of-Layer taxonomy reconstruction (always on when corpus is present), Schema-Guided Completion, deterministic cleanup and orphan rescue, an LLM Chain-of-Thought refinement pass, and a rule-based reasoner.
- **Evaluation and artefacts** — qualitative LLM-as-judge scoring (described below) and a complete artefact set written per run for full reproducibility.

### Three progressive pipeline modes

| Mode | What it adds |
|------|--------------|
| **Strict** | Vocabulary-free, Zero-Shot-only, evidence-only baseline. Lets the framework generalise to unseen literature without a hard-coded vocabulary. |
| **Guided** | Vocabulary guardrails, biomedical NER anchoring (ScispaCy), and candidate-term injection. |
| **Schema-Completed** | Schema-Guided Completion + Chain-of-Layer + deterministic cleanup + LLM CoT refinement + rule-based reasoning. |

Each mode is a strict superset of the previous one, so the contribution of each post-processing layer can be measured directly through per-stage ablation snapshots.

### Three prompting strategies

- **Zero-Shot** — task instructions only, no worked examples.
- **One-Shot** — one Maximal Marginal Relevance (MMR)-selected example per chunk.
- **Few-Shot (three-phase)** — three separate LLM calls per chunk: classes, then relations, then hierarchy (the last gated by lexical cues such as *"such as"*, *"is a"*, *"type of"*). Each phase has its own example pool. An explicit anti-anchoring instruction warns the model that examples demonstrate format only, not content.

### Cross-paper Ontology Engineering

Independent of the per-paper pipeline, the **Ontology Engineering** subsystem reconstructs a single domain ontology from multiple prior runs through a **four-stage workflow: load → cluster → reconstruct → evaluate**. Stage 3 produces per-class OWL existential restrictions under three deliberate constraints — no new classes, closed source-seeded relation vocabulary, and hierarchy passthrough from source — and Stage 4 (qualitative evaluation) is decoupled from Stage 3 so the same evaluation procedure can be applied to a fresh reconstruction, a previously saved one, or any individual per-paper run loaded directly.

### Evaluation approach

Output quality is assessed using a **two-stage qualitative LLM-as-judge protocol (LLMQV)** rather than token-level alignment with a reference. In low-resource clinical domains, valid triples can be expressed in many different ways, and qualitative evaluation produces actionable feedback alongside the score: each verdict identifies what a knowledge engineer would need to change, not only whether the triple is right or wrong.

- **Stage A** classifies each extracted relation and hierarchy edge as `accept_direct`, `accept_indirect`, `reject`, or `revise`.
- **Stage B** assigns a sub-type (4.1–4.10) to every revise verdict, capturing the concrete edit a knowledge engineer would apply (target change, name fix, swap subject and object, relation-to-subclass conversion, etc.).
- Verdicts are aggregated into the **Action-Weighted Quality (AWQ)** score in [0, 1], weighted by knowledge-engineer effort.

The same LLMQV procedure runs at every per-stage snapshot, supporting structured ablation analysis.

## Findings (summary)

Across **38 valid pipeline runs** and **two cross-paper reconstructions**, four findings stand out:

1. **Pipeline configuration is the strongest driver of judged quality.** The ordering Strict (0.705) < Guided (0.800) < Schema-Completed (0.832) holds across every control tested.
2. **Prompting strategy matters in Guided mode but washes out in Schema-Completed.** Within GPT-4o-mini, Guided × One-Shot (0.841) and Guided × Few-Shot (0.821) clearly beat Guided × Zero-Shot (0.737), but Schema-Completed converges all three at AWQ ≈ 0.89 — the post-processing layers recover the same final ontology regardless of how richly the per-chunk prompt was written.
3. **GPT-4o-mini leads decisively** (mean AWQ 0.834, against 0.674–0.750 for the alternatives).
4. **The strongest single configuration came from zero-shot prompting**: Schema-Completed + Zero-Shot + GPT-4o-mini reached AWQ 0.948, suggesting that on structurally clean clinical text the post-processing pipeline alone is sufficient, even without worked examples.

Failure-mode analysis identified the four named modes (anchoring bias, scope drift, hallucinations, structural inconsistencies) and the targeted fixes that addressed each one through iterative refinement.

## Key contributions

1. **End-to-end framework + web application.** A complete, reproducible LLM-driven ontology engineering pipeline supported by a Flask web interface for experiment management, ontology visualisation, cross-run comparison, and saved artefacts at every stage.
2. **Progressive pipeline-mode design.** A vocabulary-free, evidence-only Strict baseline that lets TripleGen generalise to unseen literature, in contrast to most published LLM-driven ontology systems whose generalisation is limited by hard-coded domain vocabularies. Schema-Completed mode further integrates Chain-of-Layer taxonomy construction, producing structured hierarchy induction that flat per-chunk extraction cannot achieve.
3. **Cross-paper Ontology Engineering subsystem.** A four-stage workflow that merges multiple per-paper outputs into a cross-paper ontology under closed-vocabulary constraints (no new classes, source-seeded relations, hierarchy passthrough), with the qualitative evaluation step decoupled and reusable across any pipeline output.
4. **Two-stage qualitative LLM-as-judge evaluation.** An automated method that produces both a single AWQ score and actionable per-triple feedback, providing a practical evaluation approach in a low-resource clinical domain.

Together, these contributions position TripleGen as a draft-generation framework: it substantially reduces the manual effort of bootstrapping a domain ontology while keeping final validation and approval with the domain expert.

## Related documents

- [pipeline.md](pipeline.md) — full pipeline diagram and 10-step breakdown
- [TripleGen_WepApp.md](TripleGen_WepApp.md) — web application walkthrough with screenshots

## Technology stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Extraction LLMs | OpenAI GPT-4o-mini / GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek |
| Reasoning LLMs | OpenAI GPT-4o-mini, DeepSeek Reasoner R1 |
| Retrieval | TF-IDF + cosine similarity + MMR (λ = 0.7) |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` |
| Ontology serialisation | JSON (internal), OWL/RDF Turtle (output), rdflib |
| Web UI | Flask + Jinja2 + Cytoscape.js |
| Evaluation | Two-stage LLM-as-judge protocol producing Action-Weighted Quality (AWQ) scores |
