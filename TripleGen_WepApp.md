# TripleGen — Application Walkthrough

## How to run the app

### Prerequisites

- **Python 3.11** (or later)
- API keys for at least one LLM provider (OpenAI at minimum)

### Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# .venv\Scripts\activate        # Windows CMD

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Open .env and fill in your API keys (OPENAI_API_KEY at minimum)
```

### Start the app

```bash
python web/app.py
```

The app opens at **http://127.0.0.1:5000**.

### Browsing existing results

The `runs/` directory contains pre-computed experiment results. When you open the app, all previous runs appear in the results dropdown and on the comparison pages — no API keys are needed just to browse them.

---

## 1. Navigation

The navigation bar has five items:

| Link | What it does |
|------|-------------|
| **Run experiment** | Run a single configuration (the home page). |
| **Controlled experiment** | Fine-grained ablation with 18 individual pipeline toggles. |
| **Run comparison** | Batch mode to compare multiple configurations at once. |
| **Ontology engineering** | Cross-paper reconstruction — merge multiple prior runs, cluster the merged classes, and ask the LLM to enrich each cluster. |
| **Contrast** | Accessibility toggle for a higher-contrast theme. |

---

## 2. Home page — Run experiment

When you open the app you see two panels side by side.

The **left panel** is where you provide the text you want to analyse — you can paste it, upload files, or use the built-in default paper.

The **right panel** is where you configure the run — pick a prompting method, a pipeline mode, an LLM provider, and optional settings.

![Screenshot: Home page overview with left and right panels labelled](screenshots/02_home_overview.png)

---

### 2.1 Providing input text

You have three ways to give the system something to work with.

**Option A — Paste text**

Click the **Paste Text** tab, paste or type your clinical text into the box, and optionally give it a title (e.g. "BrainIT 2003 core dataset"). The title makes the run summary easier to read — otherwise it shows up as `corpus.txt`.

![Screenshot: Paste text input with paper title field](screenshots/03a_paste_text.png)

**Option B — Upload files**

Click the **Upload File** tab and drag-and-drop one or more `.txt` or `.pdf` files (up to 10 MB each). If you upload multiple files they are all combined into one corpus and processed together.

![Screenshot: File upload with two files selected](screenshots/03b_upload_files.png)

**Option C — Use the default paper**

Turn on the **Use default paper** toggle. The run uses the built-in BrainIT paper, and any text or files you have added are ignored.

---

### 2.2 Configuring the run

> Click **"What do these options do?"** at the top of the right panel to expand a short glossary. Each option also has a **ⓘ** tooltip on hover.

**Prompting method**

This controls how the LLM is guided during extraction:

| Strategy | What it does |
|----------|-------------|
| **Zero-Shot** | No examples — the LLM works purely from the text. This is the baseline. |
| **One-Shot** | One retrieved example is included per chunk to give the LLM a format to follow. A separate hierarchy step runs when the text contains obvious hierarchy cues. |
| **Few-Shot** | Three dedicated extraction phases, each with its own examples. Phase 1 extracts concepts broadly, Phase 2 picks up relations (and any concepts Phase 1 missed), and Phase 3 extracts hierarchy edges — but only when the text contains cues like "such as", "is a", or "type of". Instructions actively discourage the LLM from copying the example content. |

**Pipeline mode**

Each mode adds a layer on top of the previous one:

| Mode | What it adds |
|------|-------------|
| **1 · Strict** | Raw extraction with no vocabulary guardrails, no gold filtering, no NER. Evidence is the only quality gate. Restricted to Zero-Shot prompting only. |
| **2 · Guided** | Everything in Mode 1, plus vocabulary guardrails, gold-vocabulary evaluation filtering, medical NER anchoring, and candidate term suggestions. |
| **3 · Schema-Completed** | Everything in Mode 2, plus schema-guided completion (fills in gold-schema items supported by corpus evidence) and rule-based reasoning to tidy the hierarchy. |

**LLM provider**

Pick which model does the extraction:

| Provider | Model |
|----------|-------|
| **OpenAI** | GPT-4o-mini (default) |
| **OpenAI** | GPT-4o |
| **Anthropic** | Claude Haiku 4.5 |
| **Google** | Gemini 2.5 Flash |
| **Groq** | Llama 3.1 8B (free) |
| **Hugging Face** | Mistral 7B (free) |
| **DeepSeek** | deepseek-chat |

**Evaluation settings**

The **Gold-vocabulary-only** option filters the generated ontology down to the gold vocabulary before computing precision and recall. It is an evaluation control — it does not change what gets extracted, and the full unfiltered output is always saved alongside.

**Advanced / Experimental**

Expand this section to change the reasoning LLM — the model used for schema-guided completion. Choose between **OpenAI GPT-4o-mini** (default) and **DeepSeek Reasoner R1** (stronger reasoning, slower).

![Screenshot: Run configuration panel](screenshots/4_run_configuration.png)

---

### 2.3 Starting the run

Click **Run experiment** at the bottom of the left panel. The app takes you to the progress page.

---

## 3. Controlled experiment page

This page gives full granular control over every pipeline stage. Instead of the three predefined pipeline modes, you get **18 individual pipeline toggles** grouped into three sections:

![Screenshot: Controlled experiment page with pipeline toggles](screenshots/05_controlled_experiment.png)

### 3.1 Input

At the top of the left panel, a **BrainIT Paper** dropdown lets you select any of the 11 built-in BrainIT papers directly. Select a paper and its full text is loaded automatically — no pasting or uploading needed. You can also leave the dropdown blank and paste or upload your own corpus below it.

![Screenshot: BrainIT paper selector dropdown expanded](screenshots/06_paper_selector.png)

### 3.2 Pipeline controls

The right panel exposes every pipeline feature as an independent on/off switch:

**1. Preprocessing (5 toggles)**

| Toggle | What it does |
|--------|-------------|
| **Scope Filter** | Remove admin/non-clinical sections at document load time. |
| **Chunk-level Clinical Filter** | Drop low clinical-density chunks after chunking. |
| **Embedding Scope Fallback** | Use sentence-transformer embeddings to classify borderline chunks when keyword scores are ambiguous. |
| **Clinical-only Routing** | Restrict prompt routing to clinical vocabulary paths. |
| **Require Label in Evidence** | Only keep extracted classes whose label appears in the evidence text. |

**2. Prompt Injection (5 toggles)**

| Toggle | What it does |
|--------|-------------|
| **Vocabulary Guardrails** | Inject gold class/relation labels into every prompt as allowed vocabulary. |
| **Candidate Terms** | Add linguistics-based candidate noun phrases as optional hints. |
| **Medical NER Anchoring** | Run ScispaCy BC5CDR NER and inject detected entities as suggested concepts. |
| **Filter to Gold Vocabulary** | Post-parse filter: only keep classes/relations matching gold schema labels. |
| **Strict Relations** | Use strict clinical reference relation set instead of core allowed relations. |

**3. Post-Processing (8 toggles)**

| Toggle | What it does |
|--------|-------------|
| **Text-Grounded Completion** | Second-pass extraction: LLM reviews full document + current ontology to find missed items. |
| **Schema-Guided Completion** | LLM fills in gold-schema items that are missing but supported by corpus evidence. |
| **Symbolic Reasoner** | Deterministic hierarchy completion from gold schema + orphan class pruning. |
| **Deduplication** | Deduplicate classes, relations, and hierarchy edges. |
| **Scope / Abstract Pruning** | Remove out-of-scope classes, abstract data labels, and overly broad contextual classes. |
| **Evidence Pruning** | Drop classes/relations with weak or missing evidence text. |
| **Structural Validation** | Auto-add missing endpoints, prune dangling hierarchy, validate domain/range. |
| **Axiom Constraints** | Enforce ontology axiom constraints (disjointness, cardinality, range restrictions). |

Below the toggles are the same **Prompting method**, **LLM provider**, **Evaluation**, and **Advanced** sections as the home page.

This page is designed for ablation studies — toggle individual features on/off to measure their contribution.

---

## 4. Progress page

This page updates in real time while the pipeline runs.

| Element | What it shows |
|---------|--------------|
| **Run ID** | A unique identifier for this run (`YYYYMMDD-HHMMSS-<hash>`). |
| **Status badge** | Running / Completed / Failed / Cancelled. |
| **Progress bar** | How far through the chunks the pipeline is. |
| **Progress message** | The current step, e.g. "Processing chunk 3 of 12". |
| **Live knowledge graph** | A growing visual network of the classes and relations extracted so far. |
| **Triple stream** | The latest Subject, Predicate, Object extractions shown as pills. |

Click **Cancel** to stop early — a confirmation dialog appears, and if you confirm, the run stops and its files are deleted.

When the run finishes, the page automatically redirects to the Results page.

![Screenshot: Progress page showing a run in progress with the live graph](screenshots/09_progress.png)

---

## 5. Results page

### 5.1 Evaluation metrics

The left card shows the full evaluation metrics:

| Metric group | What is included |
|---|---|
| **Class** | Coverage, precision, recall, hallucinations, schema violations, omissions. |
| **Structural** | Hierarchy edges, hierarchy coverage, relation domain/range rate. |
| **Relations** | Precision, recall, counts, and a per-relation breakdown. |
| **Clinical-only** | The same class metrics but limited to clinical vocabulary. |

Below the metrics is a **per-stage ablation table** showing how the ontology changed at each pipeline stage:

```
Stage              Classes  Rels  Hier  Coverage  Precision  Recall
──────────────────────────────────────────────────────────────────
Extraction              35     8    28    48.61%    100.00%  48.61%
+ Schema-Guided         59    22    59    79.17%    100.00%  79.17%
+ Cleanup               57    19    19    79.17%    100.00%  79.17%
+ Rule-based            57    19    41    79.17%    100.00%  79.17%
```

![Screenshot: Results page — evaluation metrics card and per-stage ablation table](screenshots/10_results_metrics.png)

---

### 5.2 Knowledge artifacts

The right card lists everything the run produced. Click any link to view or download.

| Artifact | File | What it contains |
|----------|------|-----------------|
| **Ontology** | `ontology.json` | The full generated ontology — classes, relations, and hierarchy with evidence. |
| **Restricted ontology** | `ontology_restricted.json` | A copy filtered to the gold vocabulary, used for evaluation. Only produced when Gold-vocab is on. |
| **Summary** | `summary.txt` | A human-readable report covering metadata, input papers, metrics, the ablation table, and full listings. |
| **Metrics** | `metrics.json` | All metrics in structured form, including per-stage and per-relation breakdowns. |
| **Improvement counts** | `improvement_counts.json` | How many classes, relations, and hierarchy edges were added or removed at each stage. |
| **SGC diagnostic** | `sgc_diagnostic.json` | Counts from the schema-guided completion step — raw response size, items parsed, items kept. |
| **Prompts** | `prompt_chunk_NNNN.txt` | The exact prompt sent to the LLM for each chunk and phase. |
| **Metadata** | `metadata.json` | Run configuration, input file names, environment info, and code version. |

![Screenshot: Results page — knowledge artifacts card with download links](screenshots/11_results_artifacts.png)

---

### 5.3 Ontology graph

If an ontology was produced, a **Show Ontology Graph** button appears near the artifacts.

1. Click **Show Ontology Graph**.
2. A full-screen interactive graph opens, powered by Cytoscape.js.

| Feature | How it works |
|---------|-------------|
| **Colour-coded nodes** | Classes are coloured by type: teal = core, amber = governance, purple = provenance, grey = inferred. |
| **Edge types** | Solid cyan edges show hierarchy (subClassOf); dashed orange edges show relations (domain to range). |
| **Click a node** | Highlights the node and its neighbours. A panel on the right shows its label, definition, parents, children, related relations, and evidence. |
| **Layout switcher** | Switch between Force (physics), Tree (hierarchical), Circle, and Grid layouts. |
| **Export** | Click **PNG** in the toolbar to save the current graph view as an image. |
| **Stats bar** | Shows the total class, relation, and hierarchy edge counts at the bottom. |

3. Click **Close** or press **Escape** to go back to the results page.

![Screenshot: Ontology graph — tree layout](screenshots/14_ontology_graph_tree.png)

---

## 6. Running a batch comparison

Click **Run comparison** in the navigation bar. This page has the same corpus input on the left, and a configuration panel on the right, but instead of running immediately you build a list of configurations to run one after another.

The comparison page also includes a **BrainIT Paper** dropdown, so each run in the batch can target a different paper, or all runs can share the same corpus.

![Screenshot: Run comparison page overview](screenshots/15_comparison_overview.png)

---

### 6.1 Adding runs to the batch

1. Set up a configuration using the right panel — same options as a single run.
2. Click **Add run**. It appears as a new row in the table below.
3. Change any options and click **Add run** again to add another configuration.
4. Use **Reset** at any point to clear all fields back to defaults.

---

### 6.2 The runs table

Each row in the table represents one configuration:

| Column | What it shows |
|--------|--------------|
| **Checkbox** | Select this row for running or analysis. |
| **#** | Row number. |
| **Run name** | An auto-generated label, e.g. `Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - None`. |
| **Strategy** | Zero-Shot / One-Shot / Few-Shot. |
| **Pipeline mode** | Strict / Guided / Schema-Completed / Controlled. |
| **LLM** | Provider and model. |
| **Eval settings** | Gold-vocab or None. |
| **Advanced** | DSR (DeepSeek Reasoner) or None. |
| **Actions** | View the last result, view the full analysis history, or remove the row. |

The **Select all** checkbox selects or deselects everything at once.

![Screenshot: Runs table with several configurations added](screenshots/17_runs_table.png)

---

### 6.3 Running the batch

1. Tick the rows you want to run (or use **Select all**).
2. Click **Run selected**. The app takes you to the batch progress page.

![Screenshot: Run selected button with selection count](screenshots/18_run_selected.png)

---

## 7. Batch progress page

Each configuration gets its own status row:

| Element | What it shows |
|---------|--------------|
| **Run label** | The configuration name. |
| **Run ID** | Assigned when the run starts. |
| **Status badge** | Pending / Running / Paused / Completed / Failed / Skipped / Cancelled. |

Runs execute one at a time in sequence. For each running run you can **Pause**, **Resume**, or **Skip** it. **Cancel** stops everything that has not finished yet; already-completed runs are kept.

When everything is done, an **Analyze** button appears to open the batch analysis.

![Screenshot: Batch progress page with some runs completed and one running](screenshots/19_batch_progress.png)

![Screenshot: Batch progress controls — Pause, Resume, and Skip buttons](screenshots/19b_batch_controls.png)

---

## 8. Comparing and analysing results

### 8.1 Analyze selected

From the **Run comparison** page (no need to re-run anything), tick the configurations you want to compare and click **Analyze selected**. A modal shows the last completed run for each configuration.

| Column | What it shows |
|--------|--------------|
| **Run** | Row label. |
| **Overall F1** | Mean of Class, Hierarchy, and Relation F1. |
| **Class F1** | F1 for class extraction (harmonic mean of precision and recall). |
| **Hier F1** | F1 for hierarchy edge extraction. |
| **Rel F1** | F1 for relation extraction. |
| **View** | Opens the full Results page for that run in a new tab. |

The best-performing run (by Overall F1) gets a "best" badge. A grouped bar chart shows all four F1 dimensions per run (Overall in green, Class in cyan, Hierarchy in orange, Relation in purple), and a Class F1 vs Hierarchy F1 scatter plot is shown alongside the table. You can search by run name and switch between Top 10, Top 20, or All views.

![Screenshot: Analyze selected modal with comparison table and chart](screenshots/20_analyze_selected.png)

---

### 8.2 Per-run actions

In the runs table, each row has two action buttons:

| Button | What it does |
|--------|-------------|
| **Last result** (clipboard icon) | Shows a summary of the most recent completed run for this configuration. |
| **Analysis** (chart icon) | Shows Overall F1, Class F1, Hier F1, and Rel F1 for every past run with this exact configuration — useful for seeing consistency across repeated runs. |

![Screenshot: Per-run action buttons — Last result and Analysis](screenshots/21_per_run_actions.png)

![Screenshot: Analysis modal for a single configuration showing multiple past runs](screenshots/22_per_config_analysis.png)

---

### 8.3 Batch analysis

After a batch finishes from the Batch progress page, click **Analyze** to see a full comparison table showing Overall F1, Class F1, Hier F1, and Rel F1 for every run. The best run (by Overall F1) is highlighted, and each row has a **View** link to open its Results page.

---

## 9. The summary file

Every run saves a `summary.txt` that you can open from the Results page. It is the easiest way to read everything in one place.

| Section | What it covers |
|---------|---------------|
| **Metadata** | Run ID, timestamp, prompting method, pipeline mode, LLM, evaluation settings. |
| **Input papers** | The filenames you provided. Pasted text shows your chosen title, or `corpus.txt` if you did not enter one. |
| **Improvement counts** | How many classes, relations, and hierarchy edges were added, removed, or inferred at each stage. |
| **Final metrics** | Coverage, precision, recall, errors, structural metrics, clinical-only metrics, relation precision/recall. |
| **Extraction-only metrics** | The same metrics before any improvement step — your raw LLM baseline. |
| **Per-stage ablation table** | A compact table showing how the ontology changed at every stage. |
| **Concept counts** | Total classes, relations, and hierarchy edges in the final ontology. |
| **Full listings** | Every class, relation, and hierarchy edge with labels and evidence. |

![Screenshot: Summary file content in the browser](screenshots/24_summary_file.png)

---

## 10. Run labels explained

Every run gets a label that describes its exact configuration:

```
Strategy - PipelineMode - LLM - EvalSettings - Advanced
```

**Examples:**

| Label | What it means |
|-------|--------------|
| `Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - None` | 3-phase Few-Shot extraction, Schema-Completed mode, OpenAI GPT-4o-mini, gold-vocab evaluation, no advanced options. |
| `Zero-Shot - Strict - Anthropic - None - None` | Zero-Shot baseline, strict mode, using Anthropic Claude Haiku 4.5. |
| `Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - DSR` | Few-Shot with DeepSeek Reasoner handling schema-guided completion. |
| `Few-Shot - Controlled - GPT-4o-mini - Gold-vocab - None` | Few-Shot with custom pipeline toggle overrides from the Controlled Experiment page. |

This label appears everywhere — the comparison table, run list, batch progress, summary file, and metadata.

---

## 11. Ontology engineering page

Click **Ontology engineering** in the navigation bar. This page is different from every other page in the app — instead of running an extraction over a corpus, it reconstructs a single, richer ontology by combining the output of *multiple prior runs* and asking the LLM to enrich each semantic cluster of classes.

The key design properties:

- **No new classes** — the LLM is constrained to rearrange and enrich the merged source. Any class it tries to invent is dropped post-parse.
- **Closed relation vocabulary** — the allowed relation labels are self-seeded from the merged source's existing relations. The LLM never sees the gold standard, and any out-of-vocabulary relation it produces is dropped.
- **Hierarchy passthrough** — the LLM is told not to emit hierarchy. All hierarchy in the final ontology comes from the source seed.
- **Source-seeded merge** — the merged source ontology is prepended to the cluster fragments, so anything the LLM forgets to re-emit is rescued from the source.

![Screenshot: Ontology engineering page overview](screenshots/25_ontology_engineering_overview.png)

---

### 11.1 Picking source runs

The left panel lists all prior runs whose `ontology.json` is on disk. Tick the runs you want to combine — these become the source set for reconstruction. You can pick a single run, a handful of runs covering different papers, or every run in the list.

![Screenshot: Source run picker with multiple runs selected](screenshots/26_oe_source_runs.png)

---

### 11.2 Configuring the reconstruction

The right panel has a small set of options:

| Option | What it does |
|--------|-------------|
| **Number of clusters** | How many semantic clusters to split the merged classes into. Defaults to **25**. Smaller numbers produce broader clusters with more cross-cluster relations; larger numbers produce tighter, more specific clusters. |
| **LLM provider** | Which model performs the per-cluster enrichment. Same provider list as the rest of the app. |

When you click **Run reconstruction**, the app:

1. Loads each selected run's `ontology.json`.
2. Merges them by canonical key into one source ontology.
3. Embeds the class labels and definitions and clusters them semantically.
4. For each cluster, calls the LLM with the closed relation vocabulary inline in the prompt and asks it to enrich the cluster.
5. Filters the LLM output (drops new classes, drops out-of-vocab relations, drops any hierarchy edges).
6. Merges all cluster fragments back together with the source seed.
7. Runs the result through the standard evaluator.

![Screenshot: Reconstruction in progress with cluster counter](screenshots/27_oe_progress.png)

---

### 11.3 Cluster results view

When the reconstruction finishes, the page lists each cluster as a card. Each card shows the cluster name, member classes, the relations the LLM added (or kept), and how many relations were dropped by the closed-vocabulary filter and how many "inferred" classes were dropped.

This view is the cleanest way to *see* what the LLM did per cluster — and to confirm that no new classes leaked through.

![Screenshot: Cluster results — one card per cluster with member classes and added relations](screenshots/28_oe_cluster_results.png)

---

### 11.4 Compare metrics modal

The page also has a **Compare metrics** button that opens a modal showing every reconstruction run side by side, ranked by **Overall F1** (the mean of class F1, hierarchy F1, and relation F1). Each row also shows the individual class / hierarchy / relation F1, the source run IDs, and a link to open the reconstructed ontology's full Results page.

A bar chart underneath the table plots the four F1 values per run so you can spot trade-offs at a glance — for example, a reconstruction that gains hierarchy F1 at the cost of a small drop in relation F1.

![Screenshot: Compare metrics modal — table and bar chart of reconstruction runs](screenshots/29_oe_compare_metrics.png)

---

### 11.5 Reconstruction artifacts

A reconstruction run produces the same artifact set as a normal run (`ontology.json`, `metrics.json`, `summary.txt`, etc.) plus an extra `cluster_completion_log.json` recording:

- the number of clusters
- the prompt and response length per cluster
- how many "inferred" classes were dropped
- how many relations were dropped by the closed-vocabulary filter
- how many hierarchy edges were dropped

The run's `metadata.json` is also tagged with `closed_relation_vocab: true`, `n_allowed_relations: <count>`, `hierarchy_from_source_only: true`, and `source_seeded: true` so reconstruction runs are clearly distinguishable from normal extraction runs.

---

## Where are my run files?

All outputs are saved under `runs/<run_id>/`:

```
runs/<run_id>/
├── metadata.json                  # Config, input papers, environment
├── generated/
│   ├── ontology.json              # Full generated ontology
│   ├── ontology_restricted.json   # Gold-filtered copy (when Gold-vocab is on)
│   └── summary.txt                # Human-readable report
├── evaluation/
│   ├── metrics.json               # All metrics (by_stage, relations, etc.)
│   ├── improvement_counts.json
│   ├── axiom_violations.json
│   └── hallucinated_classes.json
└── prompts/
    ├── prompt_chunk_0000_phase1.txt
    ├── prompt_chunk_0000_phase2.txt
    ├── sgc_prompt.txt             # Schema-guided completion prompt
    ├── sgc_response.txt           # SGC raw LLM response
    ├── sgc_diagnostic.json        # SGC parsing counts
    └── ...
```
