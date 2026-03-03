# TripleGen — Application Walkthrough


## 1. Home page overview

The home page is split into two main panels:

| Panel | Purpose |
|-------|---------|
| **Left — Corpus input** | Choose your input text: paste, upload files, or use the built-in default paper. |
| **Right — Run configuration** | Select prompting method, pipeline mode, LLM provider, evaluation settings, and advanced options. |

The **navigation bar** at the top provides:
- **Run experiment** — the home page (single run).
- **Run comparison** — batch mode for comparing multiple configurations.
- **High contrast** — accessibility toggle for a high-contrast theme.

![Screenshot: Home page overview with left and right panels labelled](screenshots/02_home_overview.png)

---

## 2. Run Experiment — single run

### 2.1 Corpus input

The left panel lets you provide input text in one of three ways:

#### Option A: Paste text

1. Select the **Paste Text** tab.
2. Paste or type clinical / BrainIT literature into the text area.
3. Optionally enter a **Paper title** (e.g. "BrainIT 2003 core dataset") — this appears in the run summary instead of the generic `corpus.txt`.

![Screenshot: Paste text input with paper title field](screenshots/03a_paste_text.png)


#### Option B: Upload files

1. Select the **Upload File** tab.
2. Click the upload area or drag-and-drop one or more `.txt` or `.pdf` files (max 10 MB each).
3. Multiple files build a multi-document corpus — the pipeline extracts from all of them.

![Screenshot: File upload with two files selected](screenshots/03b_upload_files.png)

#### Option C: Use default paper

1. Check the **Use default paper** toggle.
2. The run uses the built-in BrainIT paper. Any pasted text or uploaded files are ignored.

---

### 2.2 Run configuration

The right panel contains all experiment settings.

> Click **"What do these options do?"** at the top to expand a built-in glossary. Each option also has an **ⓘ** info icon with a tooltip.

#### Prompting method

Choose **one** extraction strategy. Selecting one deselects the others:

| Strategy | What it does |
|----------|-------------|
| **Zero-Shot** | No examples — the LLM extracts classes, relations, and hierarchy from text alone. Serves as the baseline for comparison. |
| **One-Shot** | One MMR-retrieved example per chunk guides the LLM's extraction format and style. May trigger a hierarchy sub-call when hierarchy cues are detected. |
| **Few-Shot** | Three-phase extraction with dedicated examples per phase. Phase 1 uses 3 concept examples and a comprehensive 8-type prompt to extract classes broadly. Phase 2 uses 3 relation examples to extract relations (and any supplementary classes Phase 1 missed). Phase 3 uses 3 hierarchy examples to extract subClassOf edges, gated by lexical cues ("such as", "is a", "type of", "include:"). Anti-anchoring instructions prevent the model from narrowing its focus to example content. |


#### Pipeline mode

Choose **one** of four progressive modes. Each mode builds on top of the previous one:

| Mode | What it adds |
|------|-------------|
| **1 · Strict** | Scope filter + vocab guardrails inject gold labels into prompts. No NER, no post-processing. |
| **2 · Guided** | Strict + Medical NER anchor (ScispaCy biomedical entities as prompt hints) + candidate term injection. |
| **3 · Schema-Completed** | Guided + Schema-guided completion (LLM fills corpus-evidenced gaps from the gold schema) + Rule-based reasoning (deterministic hierarchy completion + orphan pruning). |
| **4 · Fully Reasoned** | Schema-Completed + LLM Reasoning Layer (PROPOSE → VERIFY hierarchy inference from the gold schema). |

The mode hint below the selector shows what each number enables:
`1: Scope+Vocab · 2: +Candidates+M.NER · 3: +Schema-guided+Rule-based · 4: +LLM Reasoning`


#### LLM provider

Choose **one** extraction LLM:

| Provider | Model |
|----------|-------|
| **OpenAI** | GPT-4o-mini |
| **Anthropic** | Claude Haiku 4.5 |
| **Google** | Gemini 2.5 Flash |
| **Groq** | Llama 3.1 8B (free) |
| **Hugging Face** | Mistral 7B (free) |
| **DeepSeek** | deepseek-chat |

#### Evaluation settings

| Setting | Description |
|---------|-------------|
| **Gold-vocabulary-only** | Before computing precision/recall, the generated ontology is filtered to the gold vocabulary. This is an evaluation control — it doesn't change what gets extracted. The full unfiltered output is always saved alongside. |

#### Advanced / Experimental

Click **Advanced / Experimental** to expand the collapsible section:

| Option | Description |
|--------|-------------|
| **Reasoning LLM** | The LLM used for Schema-guided completion and the LLM Reasoning Layer (Modes 3 & 4). Choose between **OpenAI** (GPT-4o-mini) and **DeepSeek Reasoner** (R1, stronger reasoning but slower). |

If you don't change anything here, OpenAI GPT-4o-mini is used for both extraction and reasoning.

![Screenshot: Run configuration](screenshots/4_run_configuration.png)

---

### 2.3 Starting the run

1. After configuring all options, click the **Run experiment** button at the bottom of the corpus input panel.
2. You are redirected to the **Progress page**.

---

## 3. Progress page

The progress page shows real-time status while the pipeline runs:

| Element | Description |
|---------|-------------|
| **Run ID** | Unique identifier (format: `YYYYMMDD-HHMMSS-<hash>`). |
| **Status badge** | Running / Completed / Failed / Cancelled. |
| **Progress bar** | Percentage of chunks processed. |
| **Progress message** | Current step, e.g. "Processing chunk 3 of 12" or "Running Schema-guided completion…". |
| **Live knowledge graph** | Visual network of extracted classes and relations, updated as chunks are processed. |
| **Triple stream** | Subject → Predicate → Object pills showing the latest extraction. |

**Cancel** — click the Cancel button to stop the run immediately. A confirmation dialog appears; cancelled runs have their files deleted.

When the run completes, you are automatically redirected to the **Results page**.

![Screenshot: Progress page showing a run in progress with the live graph](screenshots/09_progress.png)

---

## 4. Results page

### 4.1 Evaluation metrics

The left card displays the full evaluation metrics as a formatted JSON block:

| Metric group | Key fields |
|---|---|
| **Class** | coverage, precision, recall, hallucinations, schema violations, omissions |
| **Structural** | hierarchy_edges, hierarchy_coverage, relation_domain_range_rate |
| **Relations** | precision, recall, n_generated, n_gold, per_gold_relation breakdown |
| **Clinical-only** | Same as class metrics but restricted to clinical-vocabulary classes |

Below the metrics, the **Per-stage ablation table** shows how the ontology evolved through each pipeline stage:

```
Stage              Classes  Rels  Hier  Coverage  Precision  Recall
──────────────────────────────────────────────────────────────────
Extraction              35     8    28    48.61%    100.00%  48.61%
+ Schema-Guided         59    22    59    79.17%    100.00%  79.17%
+ Cleanup               57    19    19    79.17%    100.00%  79.17%
+ Rule-based            57    19    41    79.17%    100.00%  79.17%
```

This lets you see exactly which pipeline stage added or removed content — useful for understanding where the value (or damage) is coming from.

![Screenshot: Results page — evaluation metrics card and per-stage ablation table](screenshots/10_results_metrics.png)

---

### 4.2 Knowledge artifacts

The right card lists all generated artifacts. Click any link to view or download:

| Artifact | File | Description |
|----------|------|-------------|
| **Ontology** | `ontology.json` | Full generated ontology (classes, relations, hierarchy with evidence). |
| **Restricted ontology** | `ontology_restricted.json` | Gold-vocab-filtered copy used for evaluation (only produced when Gold-vocab is on). |
| **Summary** | `summary.txt` | Human-readable run report — metadata, input papers, improvement counts, metrics, ablation table, full listings. |
| **Metrics** | `metrics.json` | All metrics including `by_stage`, `relations`, `extraction_only`, `hierarchy`. |
| **Improvement counts** | `improvement_counts.json` | Per-feature tallies of items added/removed at each stage. |
| **SGC diagnostic** | `sgc_diagnostic.json` | Schema-guided completion parsing counts — raw response length, parsed items, filtered items (Modes 3–4 only). |
| **Prompts** | `prompt_chunk_NNNN.txt` | Exact prompt text sent to the LLM for each chunk and phase. |
| **Metadata** | `metadata.json` | Run configuration, input paper names and paths, environment info, code version. |

![Screenshot: Results page — knowledge artifacts card with download links](screenshots/11_results_artifacts.png)

---

### 4.3 Ontology graph

If an ontology was generated, a **Show Ontology Graph** button appears below the ontology artifact link.

1. Click **Show Ontology Graph**.
2. A full-screen interactive graph overlay opens, powered by Cytoscape.js.

#### Graph features

| Feature | Description |
|---------|-------------|
| **Colour-coded nodes** | Classes are coloured by stratum: **core** (teal), **governance** (amber), **provenance** (purple), **inferred** (grey). |
| **Edge types** | Solid cyan edges = hierarchy (subClassOf); dashed orange edges = relations (domain → range). |
| **Click a node** | Highlights the node and its immediate neighbours. A detail panel appears on the right showing the class label, definition, stratum, parents, children, connected relations, and evidence. |
| **Layout switcher** | Toolbar buttons to switch between **Force** (physics-based), **Tree** (hierarchical), **Circle**, and **Grid** layouts. |
| **Export** | Click **PNG** in the toolbar to export the current graph view as an image. |
| **Stats bar** | Bottom bar shows total classes, relations, and hierarchy edges. |
| **Legend** | Top-right legend explains node and edge colours. |

3. Click **Close** (top-right) or press **Escape** to return to the results page.

![Screenshot: Ontology graph — tree layout](screenshots/14_ontology_graph_tree.png)

---

## 5. Run Comparison — batch runs

Click **Run comparison** in the navigation bar to open the batch comparison page. This lets you run multiple configurations on the **same corpus** and compare results side-by-side.

The page has the same two-panel layout: **Corpus input** (left, shared by all runs) and **Run configuration** (right, for building individual configurations to add to the batch).

![Screenshot: Run comparison page overview](screenshots/15_comparison_overview.png)


---

### 5.1 Adding runs to the batch

1. In the **Run configuration** panel, select your desired **Prompting method**, **Pipeline mode**, **LLM provider**, **Evaluation settings**, and (optionally) **Advanced** options — exactly like the single-run page.

2. Click the **Add run** button at the bottom of the configuration panel.

3. The configuration is added as a new row in the **Run configurations** table below.

4. **Repeat** for each configuration you want to compare. Change any option before clicking **Add run** again.

5. Use the **Reset** button to clear all configuration fields back to defaults before building a new entry.

---

### 5.2 The runs table

The table shows all configurations you have added:

| Column | Description |
|--------|-------------|
| **Checkbox** | Select/deselect this configuration for running or analysis. |
| **#** | Row number. |
| **Run name** | Auto-generated 5-part label (e.g. `Few-Shot - Schema-Completed - GPT‑4o‑mini - Gold-vocab - None`). |
| **Strategy** | Zero-Shot / One-Shot / Few-Shot. |
| **Pipeline mode** | Strict / Guided / Schema-Completed / Fully Reasoned. |
| **LLM** | Provider and model. |
| **Eval settings** | Gold-vocab / None. |
| **Advanced** | DSR (DeepSeek Reasoner) / None. |
| **Actions** | View last result, view analysis, or remove the row. |

**Select all** checkbox in the header selects/deselects all rows at once.

Use the **trash icon** in the Actions column to remove an unwanted row.

![Screenshot: Runs table with several configurations added](screenshots/17_runs_table.png)

---

### 5.3 Running the batch

1. Tick the checkboxes next to the configurations you want to run (or use **Select all**).
2. Click **Run selected**. You are redirected to the **Batch progress** page.
3. The status bar shows how many runs are selected (e.g. "3 selected").

![Screenshot: Run selected button with selection count](screenshots/18_run_selected.png)


---

## 6. Batch progress page

The batch progress page shows one status row per configuration:

| Element | Description |
|---------|-------------|
| **Run label** | The 5-part label for this configuration. |
| **Run ID** | Assigned when the run starts. |
| **Status badge** | Pending / Running / Completed / Failed / Cancelled. |

Runs execute **sequentially** — one at a time. The current run shows its progress; pending runs wait in the queue.

**Cancel** stops all running and pending jobs. Completed runs are preserved.

When all runs complete, an **Analyze** button appears to open the batch analysis.

![Screenshot: Batch progress page with some runs completed and one running](screenshots/19_batch_progress.png)


---

## 7. Analyzing and comparing results

TripleGen provides several ways to analyze and compare runs.

### 7.1 Analyze selected (comparison dashboard)

From the **Run comparison** page (not during a batch), you can compare existing results without re-running:

1. Tick the configurations you want to compare.
2. Click **Analyze selected**.
3. A modal opens with a comparison table showing the **last completed run** for each configuration.

The table includes:

| Column | Description |
|--------|-------------|
| **Run** | Row number. |
| **F1** | F1 score (harmonic mean of precision and recall). |
| **Precision** | Class precision. |
| **Recall** | Class recall. |
| **Coverage** | Class coverage. |
| **View** | Opens the full Results page for that run. |

The **best run** (by F1) is highlighted with a "best" badge. An interactive bar chart visualises the comparison, and a precision-vs-recall scatter plot shows where each run sits.

You can filter the table by run name, and toggle between Top 10 / Top 20 / All views.

![Screenshot: Analyze selected modal with comparison table and chart](screenshots/20_analyze_selected.png)


---

### 7.2 Per-run analysis

In the runs table, each row has action buttons:

| Button | Icon | What it does |
|--------|------|-------------|
| **Last result** | clipboard | Opens a modal showing the summary of the most recent completed run for this configuration. |
| **Analysis** | chart | Opens a modal showing metrics for **all** past runs with this exact configuration — useful for studying LLM non-determinism across repeated runs. |

![Screenshot: Per-run action buttons — Last result and Analysis](screenshots/21_per_run_actions.png)


![Screenshot: Analysis modal for a single configuration showing multiple past runs](screenshots/22_per_config_analysis.png)


---

### 7.3 Batch analysis page

After a batch completes from the **Batch progress** page, click **Analyze** to open the full batch analysis:

- A comparison table lists every run in the batch with its metrics.
- The best run by F1 is highlighted.
- **View** links open the full Results page for any run.

---

## 8. Understanding the summary file

Every run produces a `summary.txt` — the primary human-readable report. It contains:

| Section | Contents |
|---------|----------|
| **1. Metadata** | Run ID, timestamp, prompting method, pipeline mode, LLM provider, reasoning LLM, improvements applied, evaluation settings. |
| **2. Input paper(s)** | Filenames of all ingested source documents. For pasted text, shows your Paper title (or `corpus.txt` if none was entered). |
| **3. Improvement counts** | Per-feature tallies: classes/relations/hierarchy added, removed, and inferred at each stage. |
| **4. Final ontology metrics** | Coverage, precision, recall, errors, structural metrics, clinical-only sub-metrics, relation precision/recall. |
| **5. Extraction-only metrics** | Same metrics before any improvement step — your raw LLM baseline, useful for seeing what the extraction alone achieved. |
| **6. Per-stage ablation table** | Compact table: n_classes, n_relations, n_hierarchy, coverage, precision, recall at every pipeline stage. |
| **7. Concept counts** | Total classes, relations, and hierarchy edges in the final evaluated ontology. |
| **8. Full listings** | Every class, relation, and hierarchy edge with labels and evidence. |

![Screenshot: Summary file content in the browser](screenshots/24_summary_file.png)


---

## 9. Understanding run labels

Every run is assigned a **5-part label** that uniquely identifies its configuration:

```
Strategy - PipelineMode - LLM - EvalSettings - Advanced
```

**Examples:**

| Label | Meaning |
|-------|---------|
| `Few-Shot - Schema-Completed - GPT‑4o‑mini - Gold-vocab - None` | Few-Shot (3-phase) extraction, Schema-Completed mode, OpenAI GPT-4o-mini, gold-vocab evaluation, no advanced options. |
| `Zero-Shot - Fully Reasoned - Anthropic - Gold-vocab - None` | Zero-Shot baseline with full post-processing, using Anthropic Claude. |
| `Few-Shot - Schema-Completed - GPT‑4o‑mini - Gold-vocab - DSR` | Few-Shot with DeepSeek Reasoner for schema-guided completion and LLM reasoning. |

This label appears in the comparison table, run list, batch progress, summary file, and metadata.

---

### Where are my run files?

All outputs are saved under `runs/<run_id>/`:

```
runs/<run_id>/
├── metadata.json              # Config, input papers, environment
├── generated/
│   ├── ontology.json          # Full generated ontology
│   ├── ontology_restricted.json  # Gold-filtered (when Gold-vocab is on)
│   └── summary.txt            # Human-readable report
├── evaluation/
│   ├── metrics.json           # All metrics (by_stage, relations, etc.)
│   ├── improvement_counts.json
│   ├── axiom_violations.json
│   └── hallucinated_classes.json
└── prompts/
    ├── prompt_chunk_0000_phase1.txt
    ├── prompt_chunk_0000_phase2.txt
    ├── sgc_prompt.txt          # Schema-guided completion prompt
    ├── sgc_response.txt        # SGC raw LLM response
    ├── sgc_diagnostic.json     # SGC parsing pipeline counts
    └── ...
```