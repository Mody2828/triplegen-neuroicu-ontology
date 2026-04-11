# TripleGen: LLM-Driven Ontology Engineering for Neurointensive Care

**TripleGen** is a research framework that uses Large Language Models to build ontologies from Neuro-ICU literature. It takes BrainIT consortium publications as input, extracts ontology elements (classes, relations, hierarchy), and evaluates the result against a reference ontology.


## Quick Start

```bash
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

The `runs/` directory contains pre-computed experiment results. All previous runs appear in the UI automatically — no API keys needed just to browse them.

## Web Application

The app has five main pages:

| Page | Purpose |
|------|---------|
| **Run Experiment** | Run a single extraction with configurable prompting strategy, pipeline mode, and LLM provider. |
| **Controlled Experiment** | Fine-grained ablation with 19 individual pipeline toggles and a built-in BrainIT paper selector. |
| **Run Comparison** | Batch mode — build a list of configurations, run them sequentially, and compare metrics side by side. |
| **Ontology Engineering** | Cross-paper ontology reconstruction — pick prior runs, merge them, semantically cluster the merged classes, and ask the LLM to enrich each cluster (closed relation vocabulary, hierarchy passthrough from source). |
| **Results** | View evaluation metrics, per-stage ablation tables, download artifacts, and explore an interactive ontology graph. |

For a full visual walkthrough with screenshots, see [TripleGen_WepApp.md](TripleGen_WepApp.md).

## Key Capabilities

- **Three prompting strategies**: Zero-Shot, One-Shot, Few-Shot (three-phase extraction with anti-anchoring)
- **Three pipeline modes**: Strict, Guided (+ Medical NER), Schema-Completed (+ schema-guided completion + rule-based reasoning)
- **Seven LLM providers**: OpenAI GPT-4o-mini, OpenAI GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek
- **Configurable reasoning LLM**: OpenAI GPT-4o-mini or DeepSeek Reasoner R1
- **19 pipeline toggles** for controlled ablation studies
- **Built-in BrainIT paper selector** with 11 pre-loaded papers
- **Per-stage ablation metrics** showing exactly where improvements or degradations occur
- **Interactive ontology graph** powered by Cytoscape.js
- **Cross-paper ontology engineering** — merge multiple prior runs, semantically cluster the union, and reconstruct an enriched ontology one cluster at a time using an LLM with a closed relation vocabulary self-seeded from the merged source

## Documentation

| Document | Content |
|----------|---------|
| [TripleGen_WepApp.md](TripleGen_WepApp.md) | Web app walkthrough with screenshots |
| [project_overview.md](project_overview.md) | Full methodology, pipeline details, evaluation design, and technology stack |
| [pipeline.md](pipeline.md) | Visual pipeline diagram and stage breakdown |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Extraction LLMs | OpenAI GPT-4o-mini/GPT-4o, Anthropic Claude Haiku 4.5, Google Gemini 2.5 Flash, Groq Llama 3.1 8B, Hugging Face Mistral 7B, DeepSeek |
| Reasoning LLMs | OpenAI GPT-4o-mini, DeepSeek Reasoner R1 |
| Medical NER | ScispaCy `en_ner_bc5cdr_md` |
| Web UI | Flask + Jinja2 + Cytoscape.js |
| Reference Ontology | BrainIT Golden Standard ontology |
