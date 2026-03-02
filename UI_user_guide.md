# TripleGen UI User Guide

This guide explains how to use the web interface for the LLM-driven Neuro-ICU Ontology Framework (TripleGen). The UI lets you run single experiments, run batches of comparison configurations, and inspect results.

---

## 1. Getting started

### Starting the application

From the project root:

```bash
python web/app.py
```

The app runs at `http://127.0.0.1:5000` (or the port shown in the terminal).

### Prerequisites

- **API keys:** For cloud LLMs set the corresponding key in a `.env` file at the project root:

| Provider | Key name |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Hugging Face | `HUGGINGFACE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

- **Gold standard:** The default gold ontology is `resources/brainit_core_2003.ttl`. For Restricted evaluation mode, this path is used automatically.

---

## 2. Navigation

The top navigation bar has three links:

| Link | Purpose |
|------|---------|
| **Run experiment** | Home page — run a single experiment |
| **Run comparison** | Batch page — run multiple configurations on the same corpus |
| **High contrast** | Toggle high-contrast accessibility theme |

---

## 3. Run experiment (home page)

The home page is split into two panels: **Corpus input** (left) and **Run configuration** (right).

### 3.1 Corpus input

**Input method** — Toggle between two modes:

| Mode | Description |
|------|-------------|
| **Paste Text** | Paste or type Neuro-ICU / BrainIT literature directly into the text area. |
| **Upload File** | Upload one or more `.txt` or `.pdf` files (max 10 MB each). Selecting multiple files builds a multi-document corpus. |

**Use default paper** — When checked, the run uses the built-in BrainIT paper excerpt and ignores any pasted or uploaded content. Automatically unchecked when you paste or upload.

**Paper title** *(Paste mode only, optional)* — A short label for the pasted text. Appears in the run summary under **Input paper(s)** instead of the generic `corpus.txt`. Example: `BrainIT 2003 core dataset`.

---

### 3.2 Run configuration

Click **"What do these options do?"** at the top of the configuration panel to expand a glossary of every option.

Each option also has an **ⓘ** info icon — click or focus it to see a short description in a popover.

#### Prompting method

Choose **exactly one** strategy using the toggle switches (selecting one deselects the others):

| Strategy | Default | Description |
|----------|---------|-------------|
| **Zero-Shot** | — | No examples. The model extracts classes, relations, and hierarchy from text only. Scientific baseline. |
| **One-Shot** | — | One MMR-retrieved concept example per chunk guides extraction style and format. |
| **Few-Shot** | ✓ | Two-phase extraction. Phase 1: three concept examples → extract classes. Phase 2: three relation examples → extract relations and hierarchy. Best single-extractor performance. |

#### Pipeline mode

Choose **exactly one** of four progressive mode cards. Each mode is a strict superset of the one above it:

| Mode | Sub-label | What it adds | Default |
|------|-----------|--------------|---------|
| **1 · Strict** | Strict Extraction | Scope filter removes admin/governance text before chunking. Vocab guardrails inject gold class and relation labels into every prompt. | — |
| **2 · Guided** | Guided Extraction | Everything in Mode 1 + candidate noun phrases injected as optional "Candidate terms" hints to broaden concept discovery. | — |
| **3 · Schema-Completed** | Schema-Completed | Everything in Mode 2 + Schema-guided completion (LLM fills corpus-evidenced gaps) + Rule-based Reasoning Layer (deterministic hierarchy completion and orphan pruning). | ✓ |
| **4 · Fully Reasoned** | Fully Reasoned | Everything in Mode 3 + LLM Reasoning Layer (PROPOSE → VERIFY hierarchy edges from the gold schema; no corpus evidence required). | — |

> **How to read the run label:** Each run is labelled `Strategy - Mode - LLM - EvalSettings - Advanced` (e.g. `Few-Shot - Schema-Completed - GPT‑4o‑mini - Gold-vocab - None`). This label appears in the comparison table, run list, and summary file.

#### LLM provider

Choose **exactly one** extraction LLM:

| Provider | Model |
|----------|-------|
| **OpenAI** (default) | GPT‑4o‑mini |
| **Anthropic** | Claude Haiku 4.5 |
| **Google** | Gemini 2.5 Flash |
| **Groq** | Llama 3.1 8B (free tier) |
| **Hugging Face** | Mistral 7B via Router API (free) |
| **DeepSeek** | deepseek-chat |

#### Evaluation settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Gold-vocabulary-only** | ✓ (on) | Evaluation control only — does not change what is extracted. Before computing precision/recall, the generated ontology is filtered to the gold vocabulary so non-gold items do not inflate recall or deflate precision. The full unfiltered output is always saved to `ontology.json`. |

---

### 3.3 Advanced / Experimental

Click **Advanced / Experimental** to expand the collapsible section. Two options are available:

**Medical NER ⚠ — use with caution**

Runs ScispaCy's BC5CDR biomedical NER model on the corpus. Identified disease and chemical entities are injected as *"Suggested concepts"* hints into every prompt. Empirically, this reduces class coverage by ~14 percentage points on GPT‑4o‑mini because BC5CDR identifies disease/chemical *instances* (wrong abstraction level) rather than schema-level ontology classes. Only enable if you specifically want to study NER-primed extraction. Reported in the run label as `M.NER`.

**Reasoning LLM** *(Modes 3 and 4 only)*

The LLM used exclusively for Schema-guided completion (Mode 3+) and the LLM Reasoning Layer (Mode 4). Independent of the extraction LLM.

| Option | Description |
|--------|-------------|
| **OpenAI** (default) | GPT‑4o‑mini — same model as extraction, fast. |
| **DeepSeek Reasoner** | deepseek-reasoner (R1) — stronger multi-step reasoning, slower and more expensive. |

Reported in the run label as `DSR` when DeepSeek Reasoner is selected.

---

### 3.4 Starting a run

Click **Run experiment** at the bottom of the Corpus input panel. You are redirected to the **Progress** page.

### 3.5 Additional controls

| Control | Location | Function |
|---------|----------|----------|
| **View pipeline** | Top-right of hero | Opens a modal with the full pipeline diagram (Corpus → Extract → Reason → Evaluate) with zoom controls. |
| **What do these options do?** | Top of Run configuration | Expands a scrollable glossary of all options. |
| **ⓘ info icons** | Next to each option | Click or focus to see a one-line popover description. |

---

## 4. Progress page

After starting a run you see:

| Element | Description |
|---------|-------------|
| **Run ID** | Unique identifier (`YYYYMMDD-HHMMSS-<hash>`). |
| **Status badge** | Running · Completed · Failed · Cancelled. |
| **Progress bar** | Percentage of chunks processed. |
| **Progress message** | Current step, e.g. *"Processing chunk 3 of 12"* or *"Applying Schema-guided completion…"*. |
| **Knowledge graph** | Live visualisation of extracted classes and relations as they are produced. |
| **Triple stream** | Subject · Predicate · Object pills for the most recent extraction. |

**Cancel** — Click **Cancel** to stop the run immediately. A confirmation modal appears. Cancelled runs have their generated files deleted.

On completion you are automatically redirected to the **Results** page.

---

## 5. Results page

### 5.1 Evaluation metrics

A JSON block showing the final evaluation:

| Metric group | Key fields |
|---|---|
| **Class** | coverage, precision, recall, hallucinations, schema violations, omissions |
| **Structural** | hierarchy_edges (in evaluated ontology), hierarchy_coverage, relation_domain_range_rate |
| **Relations** | precision, recall, n_generated, n_gold, per_gold_relation breakdown |
| **Clinical-only** | same as class metrics but restricted to clinical-vocabulary classes |

### 5.2 Per-stage ablation table

Below the metrics, the **Per-stage ablation** table shows how the ontology evolved through each pipeline stage:

```
Stage              Classes  Rels  Hier  Coverage  Precision  Recall
──────────────────────────────────────────────────────────────────
Extraction              25    18     7    89.29%    100.00%  89.29%
+ Schema-Guided         27    22     7    92.86%    100.00%  92.86%
+ Cleanup               24     5     1    85.71%    100.00%  85.71%
+ Rule-based            24     5     7    85.71%    100.00%  85.71%
```

This lets you see exactly which pipeline stage contributed or removed ontology content.

### 5.3 Artifacts

| Artifact | File | Description |
|----------|------|-------------|
| **Ontology** | `generated/ontology.json` | Full generated ontology (classes, relations, hierarchy). |
| **Restricted ontology** | `generated/ontology_restricted.json` | Gold-vocab-filtered copy used for evaluation metrics (only present when Gold-vocabulary-only is on). |
| **Summary** | `generated/summary.txt` | Human-readable run summary — metadata, pipeline mode, input papers, improvement counts, metrics, per-stage ablation table, and full class/relation/hierarchy listings. |
| **Metrics** | `evaluation/metrics.json` | Full metrics JSON including `by_stage`, `relations`, `extraction_only`, `clinical_only`. |
| **Improvement counts** | `evaluation/improvement_counts.json` | Per-feature counts of added/removed/inferred items. |
| **Prompts** | `prompts/prompt_chunk_NNNN.txt` | Exact prompt text sent to the LLM for each chunk. |
| **Metadata** | `metadata.json` | Run config, timestamp, `input_papers` list (filename, stem, path), environment info. |

Click any artifact link to view or download it.

### 5.4 Run label

The run label shown on this page and in the summary follows the 5-part format:

```
Strategy - PipelineMode - LLM - EvalSettings - Advanced
```

Examples:
- `Few-Shot - Schema-Completed - GPT‑4o‑mini - Gold-vocab - None`
- `Zero-Shot - Fully Reasoned - Anthropic - Gold-vocab - M.NER`
- `One-Shot - Strict - DeepSeek - None - DSR`

---

## 6. Run comparison (batch)

Use **Run comparison** in the nav bar to run multiple configurations on the same corpus and compare them side-by-side.

### 6.1 Corpus input

Identical to the home page: paste text (with optional Paper title), upload one or more files, or use the default paper. All runs in a batch share the same corpus.

### 6.2 Run configuration panel

The configuration panel mirrors the **Run experiment** form exactly — same three strategy toggles, four pipeline mode cards, six LLM provider switches, Gold-vocabulary-only checkbox, and Advanced section.

### 6.3 Adding runs to the batch

1. Select a **Prompting method**, **Pipeline mode**, **LLM provider**, and (optionally) Advanced settings.
2. Click **Add run** to add this configuration to the **Run configurations** table below.
3. Change any options and click **Add run** again for each additional configuration.
4. Use the trash icon (🗑) in the table to remove an unwanted row.
5. Use **Reset** to clear all configuration fields before building a new entry.

Each row in the table shows the **5-part run label** (`Strategy - Mode - LLM - EvalSettings - Advanced`) and the full config breakdown in five columns.

### 6.4 Pre-configured comparison groups

Three buttons let you instantly load a standard comparison batch:

| Group | Configurations loaded |
|-------|-----------------------|
| **Cross-LLM** | Few-Shot Schema-Completed across all 6 LLM providers. |
| **Pipeline Modes** | Few-Shot with all 4 pipeline modes using OpenAI. |
| **Reasoning LLM** | Few-Shot Fully Reasoned comparing OpenAI vs DeepSeek Reasoner. |

### 6.5 Running the batch

- Tick the checkboxes next to runs (or **Select all**) to choose which to run.
- Click **Run selected** to start. You are redirected to the **Batch progress** page.
- **Analyze selected** compares only the most recent completed run for each selected configuration without launching new runs.

### 6.6 Batch progress page

- One status row per run with run ID, label, and status badge.
- **Cancel** stops all running batch jobs.
- When all runs complete, an **Analyze** button appears.

### 6.7 Batch analysis

The comparison table shows one row per run:

| Column | Description |
|--------|-------------|
| Strategy | Zero-Shot / One-Shot / Few-Shot |
| Pipeline mode | Strict / Guided / Schema-Completed / Fully Reasoned |
| LLM | Provider and model |
| Eval settings | Gold-vocab / None |
| Advanced | M.NER / DSR / None |
| Precision | % |
| Recall | % |
| Coverage | % |
| Hallucinations | Count |
| Hierarchy edges | Count (in evaluated ontology) |

The best run by recall is highlighted. **View results** for any row opens its full Results page.

---

## 7. Understanding the summary file

The `generated/summary.txt` for each run is the primary human-readable report. It contains:

1. **Metadata** — Run ID, timestamp (UTC), prompting method, pipeline mode, LLM provider, reasoning LLM (if different), improvements applied, evaluation settings.
2. **Input paper(s)** — Filenames of all ingested source documents. For pasted text, shows the Paper title you entered (or `corpus.txt` if left blank).
3. **Improvement counts** — Per-feature tallies of classes/relations/hierarchy added, removed, and inferred, plus any NOTE about schema-only inferences.
4. **Metrics summary (final ontology)** — Coverage, precision, recall, errors, structural metrics, clinical-only sub-metrics, relation precision/recall.
5. **Metrics at extraction only** — The same metrics *before* any improvement step — your raw LLM baseline for direct comparison.
6. **Per-stage ablation table** — Compact table showing n_classes, n_relations, n_hierarchy, coverage, precision, recall at every pipeline stage.
7. **Concept counts** — Total classes, relations, and hierarchy edges in the final evaluated ontology.
8. **Classes, Relations, Hierarchy** — Full listings.

---

## 8. Tips and troubleshooting

### Which mode should I use?

| Goal | Recommended |
|------|-------------|
| Raw LLM output baseline, no post-processing | **Zero-Shot / One-Shot, Mode 1 (Strict)** |
| Best balance of quality and speed | **Few-Shot, Mode 3 (Schema-Completed)** ← default |
| Maximum ontology completeness | **Few-Shot, Mode 4 (Fully Reasoned)** |
| Study impact of candidate-term hints | **Any strategy, Mode 2 (Guided)** |
| Ablation study | Run all 4 modes via *Pipeline Modes* comparison group |

### API errors

- Verify API keys in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `HUGGINGFACE_API_KEY`).
- Check provider rate limits and quotas.
- Groq and Hugging Face free tiers have rate limits — add delays or use paid tiers for large corpora.

### No metrics / zero recall

- Metrics require a gold standard. The default is `resources/brainit_core_2003.ttl` and is always used in Restricted mode.
- If the run failed or was cancelled, metrics may be missing or partial.

### Run not appearing in the comparison dropdown

- Runs are stored under `runs/`. Only fully completed runs appear.
- Refresh the page. If still missing, check `runs/<run_id>/run.log` for errors.

### Slow extraction

- Few-Shot makes two LLM calls per chunk.
- Mode 3 (Schema-Completed) adds one SGC call after merge.
- Mode 4 (Fully Reasoned) adds the LLM Reasoning Layer on top.
- The Scope filter (active in all modes) reduces chunk count by removing admin text — this speeds things up significantly on the full BrainIT paper.

### Medical NER not finding anything

- Requires `scispacy` and the BC5CDR model installed: `pip install scispacy && python -m spacy download en_ner_bc5cdr_md`.
- Note: empirically shown to *reduce* class coverage. Use only for ablation purposes.

### Hierarchy coverage showing 0

- Check that Schema-Completed or Fully Reasoned mode is used (Modes 3–4).
- Gold-vocabulary-only must be enabled so hierarchy edges are evaluated against gold.
- Hierarchy coverage is the fraction of gold hierarchy edges (subClassOf) found in the generated ontology.

---

## 9. File locations

| Path | Contents |
|------|----------|
| `runs/<run_id>/` | Root of a single run |
| `runs/<run_id>/metadata.json` | Config, timestamp, `input_papers` list, environment info |
| `runs/<run_id>/generated/ontology.json` | Full generated ontology |
| `runs/<run_id>/generated/ontology_restricted.json` | Gold-vocab-filtered ontology (when Gold-vocabulary-only is on) |
| `runs/<run_id>/generated/summary.txt` | Human-readable run summary |
| `runs/<run_id>/evaluation/metrics.json` | Full metrics (incl. `by_stage`, `relations`, `extraction_only`) |
| `runs/<run_id>/evaluation/improvement_counts.json` | Per-feature improvement tallies |
| `runs/<run_id>/evaluation/axiom_violations.json` | Axiom constraint violations (if any) |
| `runs/<run_id>/prompts/` | Exact prompt text per chunk |
| `runs/<run_id>/generated/llm_reasoning_patch.json` | LLM Reasoning Layer final patch (Mode 4 only) |
| `data/corpus_ui/` | Uploaded corpus files |
| `resources/brainit_core_2003.ttl` | Gold standard ontology |
| `resources/pool_strict_concepts.json` | Few-shot concept pool |
| `resources/pool_strict_relations.json` | Few-shot relation pool |
| `resources/pool_strict_hierarchy.json` | Few-shot hierarchy pool |

---

## 10. Quick-reference flow summary

| Flow | Steps |
|------|-------|
| **Single run** | Home → Corpus input → Run configuration → **Run experiment** → Progress → Results |
| **Batch comparison** | Run comparison → Corpus input → Add runs (or load preset group) → **Run selected** → Batch progress → Analyze |
| **Quick preset batch** | Run comparison → load *Cross-LLM*, *Pipeline Modes*, or *Reasoning LLM* → **Run selected** |

For implementation and technical details, see `docs/implementation_summary.md`.
