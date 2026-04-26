# TripleGen: LLM-Driven Ontology Engineering for Neurointensive Care

**TripleGen** is an LLM-based framework that extracts ontologies from Neuro-ICU literature. It takes BrainIT consortium publications as input, applies strict evidence grounding to control hallucinations, and produces a structured ontology output intended as a **draft for expert review**.

## Quick Start

```bash
cd WepApp

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# .venv\Scripts\activate        # Windows CMD

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (OPENAI_API_KEY at minimum)

# Start the web app
python web/app.py
# Opens at http://127.0.0.1:5000
```

## Browsing Existing Results

The `WepApp/runs/` directory contains pre-computed experiment results. All previous runs appear in the UI automatically — no API keys needed just to browse them.

## Web Application

The app provides five main pages:

| Page | Purpose |
|------|---------|
| **Run Experiment** | Run a single extraction with configurable prompting strategy, pipeline mode, and LLM provider. |
| **Controlled Experiment** | Fine-grained ablation with individual pipeline toggles and a built-in BrainIT paper selector. |
| **Run Comparison** | Batch mode — build a list of configurations, run them sequentially, and compare results side by side. |
| **Ontology Engineering** | Cross-paper synthesis — pick prior runs, semantically cluster the merged classes, and reconstruct an enriched ontology under closed relation vocabulary and no-new-classes constraints. |
| **Results** | View qualitative evaluation verdicts, AWQ scores, per-stage ablation snapshots, downloadable artefacts, and an interactive ontology graph. |

For a full visual walkthrough with screenshots, see [TripleGen_WepApp.md](TripleGen_WepApp.md).

## Key Capabilities

- **Strict evidence gating** — every extracted class, relation, and hierarchy edge must be supported by a verbatim quote from the source paper, otherwise it is rejected.
- **Three prompting strategies** — Zero-Shot, One-Shot, and Few-Shot (three-phase extraction with anti-anchoring safeguards).
- **Three pipeline modes**:
  - **Strict** — vocabulary-free, evidence-only baseline (Zero-Shot only).
  - **Guided** — adds vocabulary standardisation and Medical NER anchoring.
  - **Schema-Completed** — adds schema-guided completion, Chain-of-Layer taxonomy construction, and rule-based reasoning.
- **Seven LLM providers** — OpenAI GPT-4o-mini, OpenAI GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek.
- **Configurable reasoning LLM** — OpenAI GPT-4o-mini or DeepSeek Reasoner R1 for the post-processing stages.
- **Two-stage qualitative LLM-as-judge evaluation** — every extracted relation and hierarchy edge is classified (`accept_direct` / `accept_indirect` / `reject` / `revise`) and aggregated into an Action-Weighted Quality (AWQ) score in [0, 1].
- **Cross-paper Ontology Engineering** — merge multiple per-paper runs into a single domain ontology through a four-stage workflow (load, cluster, reconstruct, evaluate). The LLM reconstruction step operates under three constraints: no new classes, closed source-seeded relation vocabulary, and hierarchy passthrough from source.
- **Per-stage ablation snapshots** showing exactly where improvements or degradations occur in the pipeline.
- **Built-in BrainIT paper selector** with 11 pre-loaded papers.
- **Interactive ontology graph** powered by Cytoscape.js.

## Documentation

| Document | Content |
|----------|---------|
| [TripleGen_WepApp.md](TripleGen_WepApp.md) | Web app walkthrough with screenshots |
| [project_overview.md](project_overview.md) | Methodology, pipeline details, evaluation design, and technology stack |
| [pipeline.md](pipeline.md) | Visual pipeline diagram and stage breakdown |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Extraction LLMs | OpenAI GPT-4o-mini / GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek |
| Reasoning LLMs | OpenAI GPT-4o-mini, DeepSeek Reasoner R1 |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` |
| Web UI | Flask + Jinja2 + Cytoscape.js |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` |
| Evaluation | Two-stage LLM-as-judge protocol producing Action-Weighted Quality (AWQ) scores |
