"""Interactive Flask UI for the ontology framework."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class RunCancelledError(Exception):
    """Raised when the user cancels a run from the progress page."""
    pass

# Sanitize JSON strings that may contain invalid control chars (e.g. from old runs or LLM output).
_CONTROL_CHAR = re.compile(r"[\x00-\x1f]")


def _safe_json_loads(text: str):
    """Parse JSON after replacing control characters so Invalid control character errors are avoided."""
    if not text:
        return None
    cleaned = _CONTROL_CHAR.sub(" ", text)
    return json.loads(cleaned)


# Add project root so "from src. ..." works when run as python web/app.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Runs directory: always under project root so runs are found whether app is run from web/ or project root
RUNS_DIR = _project_root / "runs"

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _safe_run_path(run_id: str) -> Path | None:
    """Validate run_id and return its path under RUNS_DIR, or None if invalid.

    Prevents path-traversal attacks by rejecting ids with '..' or '/' and
    verifying the resolved path stays under RUNS_DIR.
    """
    if not run_id or not _SAFE_ID_RE.match(run_id):
        return None
    candidate = (RUNS_DIR / run_id).resolve()
    if not str(candidate).startswith(str(RUNS_DIR.resolve())):
        return None
    return candidate

# ── BrainIT Paper Catalog ───────────────────────────────────────────────────
_PAPERS_BASE = Path(os.getenv(
    "BRAINIT_PAPERS_DIR",
    str(_project_root / "resources" / "BrainIT Papers"),
))
BRAINIT_PAPER_CATALOG: Dict[int, Dict[str, str]] = {
    1:  {"path": str(_PAPERS_BASE / "1.Pipre 2003 - The BrainIT group concept and core dataset definition.txt"),
         "short": "Piper 2003",      "label": "Paper 1 \u2014 Piper 2003 \u2014 The BrainIT Group: Concept and Core Dataset Definition"},
    2:  {"path": str(_PAPERS_BASE / "2. Moss 2013 \u2014 Trusting Intensive Care Unit .txt"),
         "short": "Moss 2013",       "label": "Paper 2 \u2014 Moss 2013 \u2014 Trusting Intensive Care Unit"},
    3:  {"path": str(_PAPERS_BASE / "3. Stell A, Piper I, Moss L (2018) Automated Measurement of Adherence to Traumati.txt"),
         "short": "Stell 2018",      "label": "Paper 3 \u2014 Stell 2018 \u2014 Automated Measurement of Adherence to TBI Guidelines"},
    4:  {"path": str(_PAPERS_BASE / "4.Shaw M, Moss LBrainIT Group (2024) Exploration of the relationship between part.txt"),
         "short": "Shaw 2024",       "label": "Paper 4 \u2014 Shaw 2024 \u2014 Exploration of PbtO2 and ICP Relationship"},
    5:  {"path": str(_PAPERS_BASE / "5. Georgatzis et al. 2016 \u2014 Artefact in Physiological Data Collected from Patients with Brain Injury.txt"),
         "short": "Georgatzis 2016", "label": "Paper 5 \u2014 Georgatzis 2016 \u2014 Artefact in Physiological Data (FSLDS Approach)"},
    6:  {"path": str(_PAPERS_BASE / "6.-Visualizing-the-pressure-and-time-burden-.txt"),
         "short": "G\u00fciza 2015a", "label": "Paper 6 \u2014 G\u00fciza 2015 \u2014 Visualizing Pressure and Time Burden (copy a)"},
    7:  {"path": str(_PAPERS_BASE / "7. Donald et al. 2012 \u2014 Trigger characteristics of EUSIG-defined hypotensive events.txt"),
         "short": "Donald 2012",     "label": "Paper 7 \u2014 Donald 2012 \u2014 Trigger Characteristics of EUSIG-Defined Hypotensive Events"},
    8:  {"path": str(_PAPERS_BASE / "8.Depreitere et al. 2018 \u2014 Cerebral Perfusion Pressure Variabili.txt"),
         "short": "Depreitere 2018", "label": "Paper 8 \u2014 Depreitere 2018 \u2014 Cerebral Perfusion Pressure Variability"},
    9:  {"path": str(_PAPERS_BASE / "9.Donald R, Howells T, Piper I, et al. (2019) Forewarning of hyp.txt"),
         "short": "Donald 2019",     "label": "Paper 9 \u2014 Donald 2019 \u2014 Forewarning of Hypotensive Events (Bayesian ANN)"},
    10: {"path": str(_PAPERS_BASE / "10.Decraene et al. 2023 \u2014 Decompressive craniectomy as a secondt.txt"),
         "short": "Decraene 2023",   "label": "Paper 10 \u2014 Decraene 2023 \u2014 Decompressive Craniectomy as Second/Third-Tier Intervention"},
    11: {"path": str(_PAPERS_BASE / "11.G\u00fciza et al. (2015) - Visualizing the pressure and time burde.txt"),
         "short": "G\u00fciza 2015b", "label": "Paper 11 \u2014 G\u00fciza 2015 \u2014 Visualizing Pressure and Time Burden (copy b)"},
}


def _resolve_paper_corpus(paper_id: int) -> Optional[str]:
    """Read a BrainIT paper by ID, copy to an isolated corpus directory, return the file path.
    Uses a separate base dir (`data/corpus_papers/`) so leftover files never
    contaminate the upload directory (`data/corpus_ui/`).
    Returns None if paper_id is invalid or file doesn't exist."""
    entry = BRAINIT_PAPER_CATALOG.get(paper_id)
    if not entry:
        return None
    src = Path(entry["path"])
    if not src.exists():
        return None
    corpus_dir = Path("data") / "corpus_papers" / f"paper_{paper_id}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for f in corpus_dir.glob("*"):
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    dest = corpus_dir / src.name
    shutil.copy2(str(src), str(dest))
    return str(dest)


# Paper group definitions for multi-paper runs
PAPER_GROUPS: Dict[str, List[int]] = {
    "group_1": [1, 2],
    "group_2": [3, 4, 5, 6, 7],
    "group_3": [8, 9, 10, 11],
}

PAPER_GROUP_LABELS: Dict[str, str] = {
    "group_1": "Group 1 (Papers 1-2)",
    "group_2": "Group 2 (Papers 3-7)",
    "group_3": "Group 3 (Papers 8-11)",
}


def _resolve_paper_group_corpus(group_id: str) -> Optional[str]:
    """Copy all papers in a group into a single corpus directory. Returns the directory path.
    The corpus loader handles directories by loading all .txt/.pdf files inside."""
    paper_ids = PAPER_GROUPS.get(group_id)
    if not paper_ids:
        return None
    corpus_dir = Path("data") / "corpus_papers" / group_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Clear old files
    for f in corpus_dir.glob("*"):
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    # Copy all papers in the group
    for pid in paper_ids:
        entry = BRAINIT_PAPER_CATALOG.get(pid)
        if not entry:
            continue
        src = Path(entry["path"])
        if not src.exists():
            continue
        dest = corpus_dir / src.name
        shutil.copy2(str(src), str(dest))
    # Verify at least one file was copied
    if not any(corpus_dir.iterdir()):
        return None
    return str(corpus_dir)

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from src.experiments.config import StrategyConfig
from src.experiments.run_experiments import run_one
from src.analysis.load_runs import load_run
from src.analysis.report import write_comparison, write_plot

# Async run status (run_id -> status dict). Access under run_status_lock.
run_statuses: Dict[str, Dict[str, Any]] = {}
run_status_lock = threading.Lock()

# Batch comparison: batch_id -> { run_ids: [], labels: [], status, corpus_path, error? }
batch_states: Dict[str, Dict[str, Any]] = {}
batch_lock = threading.Lock()

# Load .env from project root so OPENAI_API_KEY is set no matter where you start the app
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

app = Flask(__name__)

# Default gold standard: BrainIT Core 2003 (Piper et al.)
_DEFAULT_GOLD_PATH = (
    Path(__file__).resolve().parent.parent / "resources" / "brainit_core_2003.ttl"
)

# Pipeline diagram — HTML version is the primary source
_PIPELINE_HTML_PATH = _project_root / "docs" / "pipeline_diagram.html"
_PIPELINE_PNG_PATH  = _project_root / "web" / "static" / "Generated_image.png"
_PIPELINE_SVG_PATH  = _project_root / "web" / "static" / "pipeline_v2.svg"

# Active strategies for UI and comparison dashboard
_STRATEGY_LABELS = {
    "baseline": "Zero-Shot",
    "one_shot": "One-Shot",
    "phased_3step": "Few-Shot",
    # Legacy — kept for backfilling old run labels; not shown in UI
    "simple_fewshot": "Few-Shot I (legacy)",
    "phased_2step": "Few-Shot II (legacy)",
}

# Abbreviations for run name: PromptingMethod - PipelineMode - LLMModel - ReasoningLLM
_STRATEGY_SHORT = {
    "baseline": "Zero-Shot",
    "one_shot": "One-Shot",
    "phased_3step": "Few-Shot",
    # Legacy labels preserved so old run names display correctly
    "simple_fewshot": "Few-Shot I",
    "phased_2step": "Few-Shot II",
}
_LLM_MODEL_ABBREV = {
    "gpt-4o-mini": "GPT",
    "gpt-4o": "GPT4o",
    "claude-3-haiku": "C.H",
    "claude-haiku-4-5": "C.H",
    "gemini-2.0-flash": "G.F",
    "llama-3.1-8b-instant": "L.3",
    "mistralai/Mistral-7B-Instruct-v0.1": "M7B",
    "deepseek-chat": "D.S",
}
# Full display names — must match JS modelLabels / providerLabels in all templates
_LLM_MODEL_FULL = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o": "GPT-4o",
    "claude-3-haiku": "Claude 3 Haiku",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7B",
    "deepseek-chat": "DeepSeek-V3",
}
_LLM_PROVIDER_FULL = {
    "openai": "GPT-4o-mini",
    "openai-4o": "GPT-4o",
    "anthropic": "Anthropic",
    "google": "Google",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "deepseek": "DeepSeek",
    "local": "Local",
}
_REASONING_LLM_ABBREV = {
    ("openai", None): "GPT",
    ("openai", ""): "GPT",
    ("openai-4o", None): "GPT4o",
    ("openai-4o", ""): "GPT4o",
    ("openai", "gpt-4o"): "GPT4o",
    ("deepseek", "deepseek-reasoner"): "D.S.R",
}


def _pipeline_mode_label(opts: Dict[str, Any]) -> str:
    """Derive the pipeline mode name from config flags for compact run labelling.
    Schema-Completed = gold-schema injection (SGC or symbolic reasoner).
    Strict = text-only (text-grounded completion is always built-in).
    Controlled = any non-default cleanup/preprocessing toggle was explicitly set.
    Must stay in sync with JS pipelineModeLabel() in comparison_dashboard.html and comparison_progress.html.
    """
    _cleanup_keys = ("cleanup_dedupe", "cleanup_scope_pruning", "cleanup_evidence_pruning",
                     "cleanup_structural", "cleanup_axioms")
    has_custom_cleanup = any(k in opts and opts[k] is not True for k in _cleanup_keys)
    has_tgl_off = opts.get("text_grounded_completion") is False
    if has_custom_cleanup or has_tgl_off:
        return "Controlled"
    sgc = opts.get("schema_guided_completion", False)
    sym = opts.get("symbolic_reasoner", False)
    if sgc or sym:
        return "Schema-Completed"
    # Guided = has guardrails (NER, candidate terms, or vocab) but no SGC/reasoner
    med = opts.get("medical_ner_anchor", False)
    cand = opts.get("candidate_terms", False)
    vocab = opts.get("prompt_vocab_guardrails", False)
    if med or cand or vocab:
        return "Guided"
    return "Strict"


def _format_run_label(run_opts: Dict[str, Any]) -> str:
    """Build run name: Strategy - PipelineMode - LLM - EvalSettings - Advanced - Paper.
    Must stay in sync with JS formatRunName() in comparison_dashboard.html and comparison_progress.html.
    e.g. 'Few-Shot - Schema-Completed - GPT-4o-mini - Gold-vocab - None - Piper 2010(3)'
    Never raises.
    """
    try:
        opts = _safe_config_for_label(run_opts)
        strategy = str(opts.get("strategy", "baseline")).strip() or "baseline"
        prompting = _STRATEGY_SHORT.get(strategy, _STRATEGY_LABELS.get(strategy, strategy))
        mode = _pipeline_mode_label(opts)
        # LLM full name
        model_id = (str(opts.get("llm_model") or "").strip()) or None
        provider = (str(opts.get("llm_provider") or "")).strip() or "openai"
        llm = _LLM_MODEL_FULL.get(model_id) if model_id else None
        if llm is None:
            llm = _LLM_PROVIDER_FULL.get(provider, provider)
        # Eval settings
        eval_s = "Gold-vocab" if opts.get("eval_restrict_to_gold") else "None"
        # Advanced (Reasoning LLM if DSR + Medical NER flag)
        imp_provider = (str(opts.get("improvements_llm_provider") or "").strip() or None)
        imp_model = (str(opts.get("improvements_llm_model") or "")).strip() or ""
        has_llm_imp = opts.get("schema_guided_completion")
        adv_parts = []
        if has_llm_imp and imp_provider == "deepseek" and "reasoner" in imp_model.lower():
            adv_parts.append("DSR")
        adv = "+".join(adv_parts) if adv_parts else "None"
        # Paper identifier
        paper_id = opts.get("paper_id")
        paper_suffix = ""
        if paper_id is not None:
            paper_id_str = str(paper_id).strip()
            if paper_id_str in PAPER_GROUP_LABELS:
                paper_suffix = f" - {PAPER_GROUP_LABELS[paper_id_str]}"
            else:
                try:
                    pid = int(paper_id)
                    entry = BRAINIT_PAPER_CATALOG.get(pid)
                    if entry:
                        paper_suffix = f" - {entry['short']}({pid})"
                except (ValueError, TypeError):
                    pass
        return f"{prompting} - {mode} - {llm} - {eval_s} - {adv}{paper_suffix}"
    except Exception:
        return "Zero-Shot - Strict - GPT-4o-mini - None - None"


def _make_combo(id_: str, strategy: str, llm_provider: str, llm_model: str,
                medical: bool, candidate: bool, sgc: bool, sym: bool,
                vocab: bool, eval_gold: bool, imp_provider: str, imp_model: str,
                row_group: str, paper_id: int | str | None = None,
                embedding_scope_fallback: bool = False) -> Dict:
    combo = {
        "id": id_,
        "strategy": strategy,
        "strategy_label": _STRATEGY_LABELS.get(strategy, strategy),
        "medical_ner_anchor": medical,
        "candidate_terms": candidate,
        "schema_guided_completion": sgc,
        "symbolic_reasoner": sym,
        "prompt_vocab_guardrails": vocab,
        "eval_restrict_to_gold": eval_gold,
        "improvements_llm_provider": imp_provider or None,
        "improvements_llm_model": imp_model or None,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "row_group": row_group,
        "scope_filter": True,
        "paper_id": paper_id,
        "embedding_scope_fallback": embedding_scope_fallback,
    }
    combo["label"] = _format_run_label(combo)
    return combo


def _build_comparison_combinations() -> list:
    """Build comparison combinations for the dashboard.

    Dissertation experiment matrix (39 runs):
      A — Core Ablation: GPT-4o-mini × 7 strategy–mode combos × 3 paper groups (21 runs)
      B — Cross-LLM Best: Few-Shot + Schema-Completed × 3 LLMs × 3 groups (9 runs)
      C — Cross-LLM Raw:  Zero-Shot + Strict × 3 LLMs × 3 groups (9 runs)
    """
    out = []

    # Paper group definitions: group label → list of paper IDs
    GROUPS = [
        ("G1", [1, 2]),       # Foundational
        ("G2", [3, 4, 5, 6, 7]),  # Clinical Analysis
        ("G3", [8, 9, 10, 11]),   # Advanced Monitoring
    ]

    # ── Mode presets (strategy, medical, candidate, sgc, sym, vocab, eval_gold, imp_prov, imp_model, emb_fallback) ──
    # Strict:           Zero-Shot only, no guardrails, no gold filter, embedding fallback ON
    STRICT  = ("baseline",      False, False, False, False, False, False, "", "", True)
    # Guided:           all strategies, NER + candidate terms ON, no gold-vocab, no SGC
    GUIDED  = (None,            True,  True,  False, False, False, False, "", "", False)
    # Schema-Completed: all strategies, NER + candidate terms ON, no gold-vocab, SGC + reasoner ON
    SCHEMA  = (None,            True,  True,  True,  True,  False, False, "openai", "", False)

    def _add(id_pfx, strategy, mode_tuple, llm_prov, llm_model, group_label, row_group):
        """Add one run for the entire paper group."""
        _, med, cand, sgc, sym, vocab, eg, ip, im, emb = mode_tuple
        group_key = {"G1": "group_1", "G2": "group_2", "G3": "group_3"}[group_label]
        out.append(_make_combo(
            id_pfx, strategy, llm_prov, llm_model,
            med, cand, sgc, sym, vocab, eg, ip, im,
            row_group, paper_id=group_key, embedding_scope_fallback=emb,
        ))

    # ── Batch A: Core Ablation (GPT-4o-mini, 7 combos × 3 groups = 21 runs) ──
    GPT = ("openai", "gpt-4o-mini")
    for gl, _pids in GROUPS:
        # 1. Zero-Shot + Strict
        _add(f"A_zs_strict_{gl}", "baseline", STRICT, *GPT, gl, "batch_a")
        # 2. Zero-Shot + Guided
        _add(f"A_zs_guided_{gl}", "baseline", GUIDED, *GPT, gl, "batch_a")
        # 3. Zero-Shot + Schema-Completed
        _add(f"A_zs_schema_{gl}", "baseline", SCHEMA, *GPT, gl, "batch_a")
        # 4. One-Shot + Guided
        _add(f"A_os_guided_{gl}", "one_shot", GUIDED, *GPT, gl, "batch_a")
        # 5. One-Shot + Schema-Completed
        _add(f"A_os_schema_{gl}", "one_shot", SCHEMA, *GPT, gl, "batch_a")
        # 6. Few-Shot + Guided
        _add(f"A_fs_guided_{gl}", "phased_3step", GUIDED, *GPT, gl, "batch_a")
        # 7. Few-Shot + Schema-Completed
        _add(f"A_fs_schema_{gl}", "phased_3step", SCHEMA, *GPT, gl, "batch_a")

    # ── Batch B: Cross-LLM Best Config (Few-Shot + Schema-Completed × 3 LLMs × 3 groups = 9 runs) ──
    CROSS_LLMS = [
        ("anthropic", "claude-haiku-4-5"),
        ("google",    "gemini-2.0-flash"),
        ("deepseek",  "deepseek-chat"),
    ]
    for llm_prov, llm_model in CROSS_LLMS:
        for gl, _pids in GROUPS:
            _add(f"B_fs_schema_{llm_prov}_{gl}", "phased_3step", SCHEMA, llm_prov, llm_model, gl, "batch_b")

    # ── Batch C: Cross-LLM Raw Baseline (Zero-Shot + Strict × 3 LLMs × 3 groups = 9 runs) ──
    for llm_prov, llm_model in CROSS_LLMS:
        for gl, _pids in GROUPS:
            _add(f"C_zs_strict_{llm_prov}_{gl}", "baseline", STRICT, llm_prov, llm_model, gl, "batch_c")

    return out


COMPARISON_COMBINATIONS = _build_comparison_combinations()


def get_default_gold_path() -> str:
    """Path to gold standard: env GOLD_STANDARD_PATH or resources/brainit_core_2003.ttl if it exists."""
    env_path = os.getenv("GOLD_STANDARD_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    if _DEFAULT_GOLD_PATH.exists():
        return str(_DEFAULT_GOLD_PATH)
    return ""


def _read_default_paper() -> str:
    """Read default paper text from path in .env DEFAULT_PAPER_PATH (relative to project root). Returns empty string if unset or file missing."""
    path_str = os.getenv("DEFAULT_PAPER_PATH", "").strip()
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_absolute():
        p = _project_root / p
    if not p.exists() or not p.is_file():
        return ""
    try:
        suf = p.suffix.lower()
        if suf == ".pdf":
            return extract_text_from_pdf(p)
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                return p.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _get_default_paper_name() -> str:
    """Return the filename (with extension) of DEFAULT_PAPER_PATH, or '' if unset/missing."""
    path_str = os.getenv("DEFAULT_PAPER_PATH", "").strip()
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_absolute():
        p = _project_root / p
    return p.name if p.exists() else Path(path_str).name


def write_corpus_file(content: str, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    for f in base_dir.glob("*"):
        if f.is_file() and f.suffix.lower() in (".txt", ".pdf"):
            try:
                f.unlink()
            except OSError:
                pass
    corpus_path = base_dir / "corpus.txt"
    corpus_path.write_text(content, encoding="utf-8")
    return corpus_path


def _safe_filename(name: str) -> str:
    """Sanitize filename to avoid path traversal and invalid chars."""
    if not name or not name.strip():
        return "document"
    base = os.path.basename(name.strip())
    safe = re.sub(r"[^\w\s\-\.]", "", base) or "document"
    return safe[:200]  # limit length


def save_uploaded_files_to_corpus_dir(uploaded_files: List, corpus_dir: Path) -> Path:
    """Save multiple uploaded files to corpus directory. Clears existing .txt/.pdf and subdirs first. Returns corpus_dir."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for f in corpus_dir.iterdir():
        if f.is_dir():
            try:
                shutil.rmtree(f)
            except OSError:
                pass
        elif f.is_file() and f.suffix.lower() in (".txt", ".pdf"):
            try:
                f.unlink()
            except OSError:
                pass
    seen: Dict[str, int] = {}
    for uf in uploaded_files:
        if not uf or not getattr(uf, "filename", None):
            continue
        base = _safe_filename(uf.filename)
        if not base.endswith((".txt", ".pdf")):
            ext = os.path.splitext(uf.filename)[1].lower()
            if ext in (".txt", ".pdf"):
                base = base + ext
            else:
                base = base + ".txt"
        if base in seen:
            seen[base] += 1
            stem, suf = os.path.splitext(base)
            base = f"{stem}_{seen[base]}{suf}"
        else:
            seen[base] = 1
        out_path = corpus_dir / base
        try:
            uf.save(str(out_path))
        except Exception:
            pass
    return corpus_dir


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from a PDF file."""
    try:
        import PyPDF2
    except ImportError:
        raise RuntimeError(
            "PyPDF2 is not installed. Install it with: pip install PyPDF2"
        )
    
    text_content = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
        return "\n\n".join(text_content)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text from an uploaded file (text or PDF)."""
    if not uploaded_file:
        return ""
    
    # Get file extension
    filename = uploaded_file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    # Create temporary file with a unique name
    # Use mkstemp for better Windows compatibility
    fd, tmp_path_str = tempfile.mkstemp(suffix=ext)
    tmp_path = Path(tmp_path_str)
    
    try:
        # Save uploaded file to temporary location
        uploaded_file.save(tmp_path_str)
        
        # Close the file descriptor immediately to release the lock (Windows requirement)
        os.close(fd)
        
        # Now read the file content
        if ext == '.pdf':
            content = extract_text_from_pdf(tmp_path)
        elif ext == '.txt':
            # Try multiple encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            for encoding in encodings:
                try:
                    content = tmp_path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                # Fallback with error handling
                content = tmp_path.read_text(encoding='utf-8', errors='replace')
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .txt and .pdf are supported.")
        
        return content
    finally:
        # Clean up temporary file (ensure it's closed first)
        try:
            if tmp_path.exists():
                # On Windows, wait a moment and retry if file is locked
                import time
                for _ in range(3):
                    try:
                        tmp_path.unlink()
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.1)
        except Exception:
            # Ignore cleanup errors - temp file will be cleaned up by OS eventually
            pass


def load_metrics(run_root: Path) -> Optional[dict]:
    metrics_path = run_root / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return _safe_json_loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_run_ids(base_dir: str | Path = None) -> list[str]:
    root = Path(base_dir) if base_dir is not None else RUNS_DIR
    if not root.exists():
        return []
    run_ids = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not (p / "metadata.json").exists():
            continue
        if not (p / "evaluation" / "metrics.json").exists():
            continue
        run_ids.append(p.name)
    return sorted(run_ids, reverse=True)


def load_analysis(run_root: Path) -> Optional[dict]:
    analysis_dir = run_root / "analysis"
    if not analysis_dir.exists():
        return None
    summary_path = analysis_dir / "summary.txt"
    comparison_path = analysis_dir / "comparison.csv"
    plot_path = analysis_dir / "comparison.png"
    return {
        "summary": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        "comparison_csv": comparison_path if comparison_path.exists() else None,
        "comparison_plot": plot_path if plot_path.exists() else None,
    }


# Config registry: map config fingerprint -> list of run_ids (so we can show "last result" and "analysis" per config)
CONFIG_REGISTRY_PATH = RUNS_DIR / "_config_registry.json"

_CONFIG_KEYS = (
    "strategy", "medical_ner_anchor", "candidate_terms", "scope_filter", "llm_provider", "llm_model",
    "schema_guided_completion", "symbolic_reasoner", "prompt_vocab_guardrails",
    "eval_restrict_to_gold",
    "improvements_llm_provider", "improvements_llm_model",
    "paper_id",
    "text_grounded_completion", "chunk_clinical_filter", "embedding_scope_fallback",
    "clinical_only_routing", "require_label_in_evidence",
    "filter_to_gold_vocabulary", "strict_relations",
    "cleanup_dedupe", "cleanup_scope_pruning", "cleanup_evidence_pruning",
    "cleanup_structural", "cleanup_axioms",
)

# Defaults for every key used by _format_run_label so we never fail on missing/dirty config
_LABEL_DEFAULTS = {
    "strategy": "baseline",
    "medical_ner_anchor": False,
    "candidate_terms": False,
    "scope_filter": False,
    "prompt_vocab_guardrails": False,
    "llm_provider": "openai",
    "llm_model": None,
    "schema_guided_completion": False,
    "symbolic_reasoner": False,
    "eval_restrict_to_gold": False,
    "improvements_llm_provider": None,
    "improvements_llm_model": None,
    "paper_id": None,
    "text_grounded_completion": True,
    "chunk_clinical_filter": True,
    "embedding_scope_fallback": False,
    "clinical_only_routing": True,
    "require_label_in_evidence": True,
    "filter_to_gold_vocabulary": False,
    "strict_relations": False,
    "cleanup_dedupe": True,
    "cleanup_scope_pruning": True,
    "cleanup_evidence_pruning": True,
    "cleanup_structural": True,
    "cleanup_axioms": True,
}


def _safe_config_for_label(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a config dict with all keys needed for _format_run_label set to safe defaults. Never raises."""
    if not isinstance(config, dict):
        return dict(_LABEL_DEFAULTS)
    out = dict(_LABEL_DEFAULTS)
    for k in _LABEL_DEFAULTS:
        if k in config and config[k] is not None:
            default = _LABEL_DEFAULTS[k]
            if isinstance(default, bool):
                v = config[k]
                if isinstance(v, bool):
                    out[k] = v
                elif isinstance(v, str):
                    out[k] = v.strip().lower() in ("1", "true", "yes", "on")
                else:
                    out[k] = bool(v)
            else:
                out[k] = config[k] if isinstance(config[k], str) else (str(config[k]).strip() or _LABEL_DEFAULTS[k])
        elif k in config:
            out[k] = _LABEL_DEFAULTS[k]
    if "paper_id" in config and config["paper_id"] is not None:
        out["paper_id"] = config["paper_id"]
    return out


def _config_fingerprint(run_opts: Dict[str, Any]) -> str:
    """Stable hash of run options so we can group runs by config."""
    canonical = {}
    for k in _CONFIG_KEYS:
        v = run_opts.get(k)
        if v is None:
            canonical[k] = None
        elif isinstance(v, bool):
            canonical[k] = v
        else:
            canonical[k] = str(v).strip() if v else None
    canonical["strategy"] = run_opts.get("strategy") or "baseline"
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _load_config_registry() -> Dict[str, List[str]]:
    if not CONFIG_REGISTRY_PATH.exists():
        return {}
    try:
        data = _safe_json_loads(CONFIG_REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _run_opts_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build run_opts from a run's metadata.json config (for fingerprinting). Returns None if config missing/invalid."""
    try:
        config = metadata.get("config")
        if not isinstance(config, dict):
            return None
        return _run_opts_from_config(config)
    except Exception:
        return None


def _backfill_run_ids_for_fingerprint(fp: str) -> List[str]:
    """Return run_ids from registry for fp, plus any runs on disk with matching config (recovers lost registry entries)."""
    registry = _load_config_registry()
    run_ids = registry.get(fp) or []
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    seen = set(run_ids)
    if not RUNS_DIR.exists():
        return sorted(run_ids)
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        metadata_path = run_dir / "metadata.json"
        metrics_path = run_dir / "evaluation" / "metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            continue
        try:
            meta = _safe_json_loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                continue
            run_opts = _run_opts_from_metadata(meta)
            if run_opts is None:
                continue
            run_fp = _config_fingerprint(_normalize_config_for_fingerprint(run_opts))
            if run_fp == fp and run_id not in seen:
                seen.add(run_id)
                run_ids.append(run_id)
        except Exception:
            continue
    return sorted(run_ids)


def _save_config_registry(registry: Dict[str, List[str]]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


_registry_lock = threading.Lock()


def _register_run_for_config(run_opts: Dict[str, Any], run_id: str) -> None:
    """Append run_id to the list of runs for this config (so we can show last result / analysis)."""
    fp = _config_fingerprint(_normalize_config_for_fingerprint(run_opts))
    with _registry_lock:
        registry = _load_config_registry()
        registry.setdefault(fp, []).append(run_id)
        _save_config_registry(registry)


def _run_opts_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build run_opts dict from app config (for single-run registration).
    Must include every key in _CONFIG_KEYS so fingerprints stay consistent."""
    strategies = config.get("strategies") or []
    strategy = strategies[0].get("prompt_strategy", "baseline") if strategies else "baseline"
    return {
        "strategy": strategy,
        "medical_ner_anchor": config.get("medical_ner_anchor", False),
        "candidate_terms": config.get("candidate_terms", False),
        "scope_filter": config.get("scope_filter", False),
        "llm_provider": config.get("llm_provider") or "openai",
        "llm_model": config.get("llm_model"),
        "schema_guided_completion": config.get("schema_guided_completion", False),
        "symbolic_reasoner": config.get("symbolic_reasoner", False),
        "prompt_vocab_guardrails": config.get("prompt_vocab_guardrails", False),
        "eval_restrict_to_gold": config.get("eval_restrict_to_gold", False),
        "improvements_llm_provider": config.get("improvements_llm_provider"),
        "improvements_llm_model": config.get("improvements_llm_model"),
        "paper_id": config.get("paper_id"),
        "text_grounded_completion": config.get("text_grounded_completion", True),
        "chunk_clinical_filter": config.get("chunk_clinical_filter", True),
        "clinical_only_routing": config.get("clinical_only_routing", True),
        "require_label_in_evidence": config.get("require_label_in_evidence", True),
        "filter_to_gold_vocabulary": config.get("filter_to_gold_vocabulary", False),
        "strict_relations": config.get("strict_relations", False),
        "cleanup_dedupe": config.get("cleanup_dedupe", True),
        "cleanup_scope_pruning": config.get("cleanup_scope_pruning", True),
        "cleanup_evidence_pruning": config.get("cleanup_evidence_pruning", True),
        "cleanup_structural": config.get("cleanup_structural", True),
        "cleanup_axioms": config.get("cleanup_axioms", True),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        default_gold_path=get_default_gold_path(),
        run_ids=list_run_ids(),
        gold_mode_default="restricted",
    )


@app.route("/controlled-experiment")
def controlled_experiment():
    paper_options = [{"id": pid, "label": entry["label"], "short": entry["short"]}
                     for pid, entry in sorted(BRAINIT_PAPER_CATALOG.items())]
    return render_template(
        "controlled_experiment.html",
        default_gold_path=get_default_gold_path(),
        run_ids=list_run_ids(),
        gold_mode_default="restricted",
        paper_options=paper_options,
    )


@app.route("/run", methods=["POST"])
def run():
    from_page = request.form.get("from_page", "")

    # Get input method (paste or upload)
    input_method = request.form.get("input_method", "paste")
    use_default_paper = request.form.get("use_default_paper") == "on"

    # Controlled experiment page may supply a paper_id instead of pasted text
    paper_id_raw = request.form.get("paper_id", "").strip() if from_page == "controlled_experiment" else ""

    # Get text content based on input method
    text_input = ""
    uploaded_files = []
    paper_name = request.form.get("paper_name", "").strip()

    # Resolve paper_id first (controlled experiment page)
    if paper_id_raw:
        try:
            pid = int(paper_id_raw)
            resolved = _resolve_paper_corpus(pid)
            if resolved:
                text_input = "__paper__"  # sentinel handled below
                entry = BRAINIT_PAPER_CATALOG.get(pid, {})
                if not paper_name:
                    paper_name = entry.get("short", f"Paper {pid}")
        except (ValueError, TypeError):
            paper_id_raw = ""

    if not text_input:
        if input_method == "paste":
            text_input = request.form.get("text_input", "").strip()
        elif input_method == "upload":
            uploaded_files = request.files.getlist("file_input") or []
            uploaded_files = [f for f in uploaded_files if f and getattr(f, "filename", None)]
            if uploaded_files:
                text_input = "__multi_file__"
        if use_default_paper and not text_input:
            text_input = _read_default_paper()
            if text_input and not paper_name:
                paper_name = _get_default_paper_name()

    strategy = request.form.get("strategy", "baseline")
    llm_provider = (request.form.get("llm_provider") or "").strip() or "openai"
    if not llm_provider or llm_provider == "none":
        llm_provider = "openai"
    llm_model = (request.form.get("llm_model") or "").strip() or None
    gold_mode = request.form.get("gold_mode", "restricted")
    gold_path = request.form.get("gold_path", "").strip()
    gold_text_input = request.form.get("gold_text_input", "").strip()
    if gold_mode == "restricted" and not gold_path:
        gold_path = get_default_gold_path()

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    def err_response(msg: str, status: int = 400):
        if is_ajax:
            return jsonify({"error": msg}), status
        err_tmpl = "controlled_experiment.html" if from_page == "controlled_experiment" else "index.html"
        err_ctx: Dict[str, Any] = {
            "error": msg,
            "default_gold_path": get_default_gold_path(),
            "run_ids": list_run_ids(),
            "gold_mode_default": "restricted",
        }
        if from_page == "controlled_experiment":
            err_ctx["paper_options"] = [
                {"id": pid, "label": e["label"], "short": e["short"]}
                for pid, e in sorted(BRAINIT_PAPER_CATALOG.items())
            ]
        return render_template(err_tmpl, **err_ctx)

    if not text_input:
        error_msg = "Please paste corpus text, upload one or more files, or select a BrainIT paper."
        if input_method == "upload":
            error_msg = "Please select at least one file to upload (.txt or .pdf)."
        return err_response(error_msg)

    if gold_mode == "restricted" and not gold_path:
        return err_response(
            "Restricted mode requires a gold standard path. Set GOLD_STANDARD_PATH or use resources/brainit_core_2003.ttl."
        )
    if gold_mode == "isolated" and not gold_text_input:
        return err_response(
            "Isolation mode requires gold standard corpus text. Paste text into the gold corpus box."
        )

    os.environ["GOLD_STANDARD_MODE"] = gold_mode

    # Use a unique per-run corpus directory to avoid race conditions where a
    # background thread from a previous run holds file locks (Windows) and
    # prevents cleanup, causing leftover files to be processed alongside new ones.
    corpus_base = Path("data") / "corpus_ui"
    corpus_base.mkdir(parents=True, exist_ok=True)
    # Best-effort cleanup of old per-run subdirectories (ignore locked ones)
    for old in corpus_base.iterdir():
        if old.is_dir():
            try:
                shutil.rmtree(old)
            except OSError:
                pass
    # Also remove any legacy top-level files from before per-run isolation
    for old in corpus_base.iterdir():
        if old.is_file():
            try:
                old.unlink()
            except OSError:
                pass
    corpus_dir = corpus_base / uuid4().hex[:10]
    corpus_path_for_run = None
    resolved_paper_id = None

    if text_input == "__paper__" and paper_id_raw:
        pid = int(paper_id_raw)
        resolved = _resolve_paper_corpus(pid)
        if resolved:
            corpus_path_for_run = resolved
            resolved_paper_id = pid
        else:
            return err_response(f"Could not resolve BrainIT paper #{pid}.")
    elif text_input == "__multi_file__" and uploaded_files:
        save_uploaded_files_to_corpus_dir(uploaded_files, corpus_dir)
        corpus_path_for_run = str(corpus_dir)
    else:
        corpus_file = write_corpus_file(text_input, corpus_dir)
        corpus_path_for_run = str(corpus_file)

    gold_corpus_path = None
    if gold_mode == "isolated":
        gold_dir = Path("data") / "gold_corpus_ui"
        gold_file = write_corpus_file(gold_text_input, gold_dir)
        gold_corpus_path = str(gold_file.parent)

    schema_guided_completion = request.form.get("schema_guided_completion") == "on"
    symbolic_reasoner = request.form.get("symbolic_reasoner") == "on"
    prompt_vocab_guardrails = request.form.get("prompt_vocab_guardrails") == "on"
    eval_restrict_to_gold = request.form.get("eval_restrict_to_gold") == "on"
    medical_ner_anchor = request.form.get("medical_ner_anchor") == "on"
    candidate_terms = request.form.get("candidate_terms") == "on"
    scope_filter = request.form.get("scope_filter") == "on"
    improvements_llm_provider = (request.form.get("improvements_llm_provider") or "").strip()
    improvements_llm_model = (request.form.get("improvements_llm_model") or "").strip()

    # Controlled-experiment flags. When from_page is controlled_experiment, unchecked
    # checkboxes are absent from form data → False. From other pages the fields are
    # never sent, so default to legacy on-by-default behavior.
    _is_ctrl = from_page == "controlled_experiment"
    def _flag(key, legacy_default="on"):
        if _is_ctrl:
            return request.form.get(key) == "on"
        return request.form.get(key, legacy_default) == "on"

    text_grounded_completion = _flag("text_grounded_completion")
    chunk_clinical_filter = _flag("chunk_clinical_filter")
    embedding_scope_fallback = _flag("embedding_scope_fallback", legacy_default="")
    clinical_only_routing = _flag("clinical_only_routing")
    require_label_in_evidence = _flag("require_label_in_evidence")
    filter_to_gold_vocabulary = _flag("filter_to_gold_vocabulary", legacy_default="")
    strict_relations = _flag("strict_relations", legacy_default="")
    cleanup_dedupe = _flag("cleanup_dedupe")
    cleanup_scope_pruning = _flag("cleanup_scope_pruning")
    cleanup_evidence_pruning = _flag("cleanup_evidence_pruning")
    cleanup_structural = _flag("cleanup_structural")
    cleanup_axioms = _flag("cleanup_axioms")

    config = {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "corpus_path": corpus_path_for_run,
        "gold_standard_path": gold_path or None,
        "gold_standard_mode": gold_mode,
        "gold_corpus_path": gold_corpus_path,
        "runs_dir": str(RUNS_DIR),
        "schema_guided_completion": schema_guided_completion,
        "symbolic_reasoner": symbolic_reasoner,
        "prompt_vocab_guardrails": prompt_vocab_guardrails,
        "eval_restrict_to_gold": eval_restrict_to_gold,
        "medical_ner_anchor": medical_ner_anchor,
        "candidate_terms": candidate_terms,
        "scope_filter": scope_filter,
        "improvements_llm_provider": improvements_llm_provider or None,
        "improvements_llm_model": improvements_llm_model or None,
        "paper_name": paper_name or None,
        "text_grounded_completion": text_grounded_completion,
        "chunk_clinical_filter": chunk_clinical_filter,
        "embedding_scope_fallback": embedding_scope_fallback,
        "clinical_only_routing": clinical_only_routing,
        "require_label_in_evidence": require_label_in_evidence,
        "filter_to_gold_vocabulary": filter_to_gold_vocabulary,
        "strict_relations": strict_relations,
        "cleanup_dedupe": cleanup_dedupe,
        "cleanup_scope_pruning": cleanup_scope_pruning,
        "cleanup_evidence_pruning": cleanup_evidence_pruning,
        "cleanup_structural": cleanup_structural,
        "cleanup_axioms": cleanup_axioms,
        "strategies": [
            {
                "name": f"ui-{strategy}",
                "prompt_strategy": strategy,
                "corpus_path": corpus_path_for_run,
                "gold_standard_path": gold_path or None,
                "gold_standard_mode": gold_mode,
                "gold_corpus_path": gold_corpus_path,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
        ],
    }
    if resolved_paper_id is not None:
        config["paper_id"] = resolved_paper_id

    strategy_cfg = StrategyConfig(
        name=config["strategies"][0]["name"],
        prompt_strategy=strategy,
        llm_provider=llm_provider,
        llm_model=llm_model,
        corpus_path=corpus_path_for_run,
        gold_standard_path=gold_path or None,
        gold_standard_mode=gold_mode,
        gold_corpus_path=gold_corpus_path,
    )

    # Async run with progress bar (when requested via X-Requested-With: XMLHttpRequest)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        run_id = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        with run_status_lock:
            run_statuses[run_id] = {
                "status": "running",
                "current": 0,
                "total": 0,
                "message": "Starting…",
                "run_id": run_id,
                "error": None,
                "cancel_requested": False,
            }

        def job():
            def progress(c: int, t: int, m: str) -> None:
                with run_status_lock:
                    if run_id in run_statuses and run_statuses[run_id].get("cancel_requested"):
                        raise RunCancelledError("Run cancelled by user")
                    if run_id in run_statuses:
                        run_statuses[run_id].update(current=c, total=t, message=m)

            try:
                run_one(config, strategy_cfg, run_id=run_id, progress_callback=progress)
                with run_status_lock:
                    if run_id in run_statuses:
                        run_statuses[run_id].update(
                            status="completed", run_id=run_id, message="Done."
                        )
                _register_run_for_config(_run_opts_from_config(config), run_id)
            except RunCancelledError:
                run_dir = RUNS_DIR / run_id
                if run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except OSError:
                        pass
                with run_status_lock:
                    if run_id in run_statuses:
                        run_statuses[run_id].update(
                            status="cancelled", run_id=run_id, message="Cancelled. Generated files deleted."
                        )
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                with run_status_lock:
                    if run_id in run_statuses:
                        run_statuses[run_id].update(
                            status="failed",
                            error=f"{e}\n\n{tb}",
                        )

        threading.Thread(target=job, daemon=True).start()
        return jsonify({
            "run_id": run_id,
            "status_url": url_for("run_status", run_id=run_id),
            "progress_url": url_for("run_progress", run_id=run_id),
        })

    try:
        run_root = run_one(config, strategy_cfg)
        _register_run_for_config(_run_opts_from_config(config), run_root.name)
        return redirect(url_for("results", run_id=run_root.name))
    except RuntimeError as e:
        msg = str(e)
        if "rdflib" in msg.lower() or "ttl" in msg.lower():
            return render_template(
                "index.html",
                error=f"Gold standard (TTL) requires rdflib. {msg} Alternatively, use 'Public' gold mode to run without a TTL file.",
                default_gold_path=get_default_gold_path(),
                run_ids=list_run_ids(),
            )
        if "openai" in msg.lower() or "openai_api_key" in msg.lower():
            return render_template(
                "index.html",
                error="That run failed: the openai package is not installed (or OPENAI_API_KEY is missing). Install with: pip install openai and set OPENAI_API_KEY in your .env file.",
                default_gold_path=get_default_gold_path(),
                run_ids=list_run_ids(),
            )
        raise


def _build_run_config(
    corpus_path: str,
    run_opts: Dict[str, Any],
    gold_path: str,
    gold_mode: str,
    gold_corpus_path: Optional[str],
    paper_name: Optional[str] = None,
) -> tuple:
    """Build config dict and StrategyConfig for one run (used by single run and comparison)."""
    strategy = run_opts.get("strategy", "baseline")
    llm_provider = (run_opts.get("llm_provider") or "").strip() or "openai"
    llm_model = (run_opts.get("llm_model") or "").strip() or None
    medical_ner_anchor = run_opts.get("medical_ner_anchor") is True
    candidate_terms = run_opts.get("candidate_terms") is True
    scope_filter = run_opts.get("scope_filter", False) is True
    config = {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "corpus_path": corpus_path,
        "gold_standard_path": gold_path or None,
        "gold_standard_mode": gold_mode,
        "gold_corpus_path": gold_corpus_path,
        "runs_dir": str(RUNS_DIR),
        "schema_guided_completion": run_opts.get("schema_guided_completion", False),
        "symbolic_reasoner": run_opts.get("symbolic_reasoner", False),
        "prompt_vocab_guardrails": run_opts.get("prompt_vocab_guardrails", False),
        "eval_restrict_to_gold": run_opts.get("eval_restrict_to_gold", False),
        "medical_ner_anchor": medical_ner_anchor,
        "candidate_terms": candidate_terms,
        "scope_filter": scope_filter,
        "improvements_llm_provider": run_opts.get("improvements_llm_provider") or None,
        "improvements_llm_model": run_opts.get("improvements_llm_model") or None,
        "paper_name": (paper_name or run_opts.get("paper_name") or "").strip() or None,
        "text_grounded_completion": run_opts.get("text_grounded_completion", True),
        "chunk_clinical_filter": run_opts.get("chunk_clinical_filter", True),
        "embedding_scope_fallback": run_opts.get("embedding_scope_fallback", False),
        "clinical_only_routing": run_opts.get("clinical_only_routing", scope_filter),
        "require_label_in_evidence": run_opts.get("require_label_in_evidence", True),
        "filter_to_gold_vocabulary": run_opts.get("filter_to_gold_vocabulary", False),
        "strict_relations": run_opts.get("strict_relations", False),
        "cleanup_dedupe": run_opts.get("cleanup_dedupe", True),
        "cleanup_scope_pruning": run_opts.get("cleanup_scope_pruning", True),
        "cleanup_evidence_pruning": run_opts.get("cleanup_evidence_pruning", True),
        "cleanup_structural": run_opts.get("cleanup_structural", True),
        "cleanup_axioms": run_opts.get("cleanup_axioms", True),
        "strategies": [
            {
                "name": f"ui-{strategy}",
                "prompt_strategy": strategy,
                "corpus_path": corpus_path,
                "gold_standard_path": gold_path or None,
                "gold_standard_mode": gold_mode,
                "gold_corpus_path": gold_corpus_path,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
        ],
    }
    strategy_cfg = StrategyConfig(
        name=config["strategies"][0]["name"],
        prompt_strategy=strategy,
        llm_provider=llm_provider,
        llm_model=llm_model,
        corpus_path=corpus_path,
        gold_standard_path=gold_path or None,
        gold_standard_mode=gold_mode,
        gold_corpus_path=gold_corpus_path,
    )
    return config, strategy_cfg


@app.route("/comparison")
def comparison_dashboard():
    """Comparison dashboard: list all run-config combinations; user selects which to include."""
    paper_options = [{"id": pid, "label": entry["label"], "short": entry["short"]}
                     for pid, entry in sorted(BRAINIT_PAPER_CATALOG.items())]
    return render_template(
        "comparison_dashboard.html",
        default_gold_path=get_default_gold_path(),
        run_ids=list_run_ids(),
        comparison_combinations=COMPARISON_COMBINATIONS,
        paper_options=paper_options,
    )


@app.route("/run-comparison", methods=["POST"])
def run_comparison():
    """Start a batch of comparison runs (same corpus, different configs). Runs execute sequentially."""
    input_method = request.form.get("input_method", "paste")
    use_default_paper = request.form.get("use_default_paper") == "on"
    text_input = ""
    uploaded_files = []
    paper_name = request.form.get("paper_name", "").strip()  # optional user-supplied title (paste mode)
    if input_method == "paste":
        text_input = request.form.get("text_input", "").strip()
    elif input_method == "upload":
        uploaded_files = request.files.getlist("file_input") or []
        uploaded_files = [f for f in uploaded_files if f and getattr(f, "filename", None)]
        if uploaded_files:
            # Always use __multi_file__ sentinel so original filenames are preserved on disk.
            text_input = "__multi_file__"
    if use_default_paper and not text_input:
        text_input = _read_default_paper()
        if text_input and not paper_name:
            paper_name = _get_default_paper_name()
    from_page = request.form.get("from_page", "")
    err_template = "comparison_dashboard.html" if from_page == "comparison_dashboard" else "index.html"
    _paper_opts = [{"id": pid, "label": e["label"], "short": e["short"]}
                   for pid, e in sorted(BRAINIT_PAPER_CATALOG.items())]
    err_ctx = {"default_gold_path": get_default_gold_path(), "run_ids": list_run_ids()}
    if err_template == "comparison_dashboard.html":
        err_ctx["comparison_combinations"] = COMPARISON_COMBINATIONS
        err_ctx["paper_options"] = _paper_opts

    runs_json_early = request.form.get("comparison_runs", "[]")
    all_runs_have_paper = False
    try:
        _early_list = _safe_json_loads(runs_json_early) if runs_json_early else []
        if isinstance(_early_list, list) and _early_list:
            all_runs_have_paper = all(
                isinstance(r, dict) and r.get("paper_id") not in (None, "", 0, "0")
                for r in _early_list
            )
    except Exception:
        pass

    if not text_input and not all_runs_have_paper:
        return render_template(err_template, error="Please paste corpus text, upload one or more files, enable Default paper, or select a paper for each run.", **err_ctx)

    gold_mode = request.form.get("gold_mode", "restricted")
    gold_path = request.form.get("gold_path", "").strip()
    gold_text_input = request.form.get("gold_text_input", "").strip()
    if gold_mode == "restricted" and not gold_path:
        gold_path = get_default_gold_path()
    if gold_mode == "isolated" and not gold_text_input:
        return render_template(err_template, error="Isolation mode requires gold standard corpus text.", **err_ctx)
    os.environ["GOLD_STANDARD_MODE"] = gold_mode
    gold_corpus_path = None
    if gold_mode == "isolated":
        gold_dir = Path("data") / "gold_corpus_ui"
        gold_file = write_corpus_file(gold_text_input, gold_dir)
        gold_corpus_path = str(gold_file.parent)

    runs_json = request.form.get("comparison_runs", "[]")
    try:
        runs_list = _safe_json_loads(runs_json) if runs_json else []
        if runs_list is None:
            runs_list = []
        if not isinstance(runs_list, list):
            raise ValueError("comparison_runs must be a JSON array")
    except (json.JSONDecodeError, ValueError):
        return render_template(err_template, error="Invalid comparison runs configuration.", **err_ctx)
    if not runs_list or len(runs_list) > 100:
        err_template = "comparison_dashboard.html" if request.form.get("from_page") == "comparison_dashboard" else "index.html"
        err_ctx = {"error": "Select between 1 and 100 runs for comparison.", "default_gold_path": get_default_gold_path(), "run_ids": list_run_ids()}
        if err_template == "comparison_dashboard.html":
            err_ctx["comparison_combinations"] = COMPARISON_COMBINATIONS
            err_ctx["paper_options"] = _paper_opts
        return render_template(err_template, **err_ctx)

    corpus_base = Path("data") / "corpus_ui"
    corpus_base.mkdir(parents=True, exist_ok=True)
    for old in corpus_base.iterdir():
        if old.is_dir():
            try:
                shutil.rmtree(old)
            except OSError:
                pass
    for old in corpus_base.iterdir():
        if old.is_file():
            try:
                old.unlink()
            except OSError:
                pass
    corpus_dir = corpus_base / uuid4().hex[:10]
    if text_input == "__multi_file__" and uploaded_files:
        save_uploaded_files_to_corpus_dir(uploaded_files, corpus_dir)
        corpus_path = str(corpus_dir)
    elif text_input:
        corpus_file = write_corpus_file(text_input, corpus_dir)
        corpus_path = str(corpus_file)
    else:
        corpus_path = ""

    batch_id = f"batch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    labels = []
    run_configs_stored = []
    for i, r in enumerate(runs_list):
        try:
            normalized = _normalize_config_for_fingerprint(r) if isinstance(r, dict) else {}
            labels.append(_format_run_label(normalized))
        except Exception:
            # Never trust frontend short label (e.g. r.get("label") -> "Few-Shot III")
            labels.append(f"Run {i + 1}")
        run_configs_stored.append(r if isinstance(r, dict) else {})
    with batch_lock:
        batch_states[batch_id] = {
            "run_ids": [None] * len(runs_list),
            "labels": labels,
            "run_configs": run_configs_stored,
            "status": "running",
            "corpus_path": corpus_path,
            "error": None,
            "cancel_requested": False,
        }

    def job():
        with batch_lock:
            state = batch_states.get(batch_id)
        if not state:
            return
        run_ids = state["run_ids"]
        for i, run_opts in enumerate(runs_list):
            with batch_lock:
                if batch_id in batch_states and batch_states[batch_id].get("cancel_requested"):
                    break
            run_id = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            with batch_lock:
                if batch_id in batch_states:
                    batch_states[batch_id]["run_ids"][i] = run_id
            with run_status_lock:
                run_statuses[run_id] = {
                    "status": "running",
                    "current": 0,
                    "total": 0,
                    "message": state["labels"][i],
                    "run_id": run_id,
                    "error": None,
                    "cancel_requested": False,
                    "pause_requested": False,
                    "skip_requested": False,
                }
            try:
                run_corpus_path = corpus_path
                run_paper_name = paper_name
                paper_id_raw = run_opts.get("paper_id") if isinstance(run_opts, dict) else None
                if paper_id_raw is not None:
                    try:
                        paper_id_str = str(paper_id_raw).strip()
                        if paper_id_str in PAPER_GROUPS:
                            # Paper group → multi-paper corpus directory
                            resolved = _resolve_paper_group_corpus(paper_id_str)
                            if resolved:
                                run_corpus_path = resolved
                                run_paper_name = PAPER_GROUP_LABELS.get(paper_id_str, paper_id_str)
                            else:
                                raise ValueError(f"Could not resolve paper group '{paper_id_str}'")
                        else:
                            # Individual paper ID
                            pid = int(paper_id_raw)
                            resolved = _resolve_paper_corpus(pid)
                            if resolved:
                                run_corpus_path = resolved
                                entry = BRAINIT_PAPER_CATALOG.get(pid)
                                run_paper_name = entry["short"] if entry else run_paper_name
                            else:
                                raise ValueError(f"Could not resolve BrainIT paper #{pid}")
                    except (ValueError, TypeError) as exc:
                        with run_status_lock:
                            run_statuses[run_id] = {"status": "error", "error": str(exc)}
                        continue
                config, strategy_cfg = _build_run_config(
                    run_corpus_path, run_opts, gold_path, gold_mode, gold_corpus_path,
                    paper_name=run_paper_name,
                )
                run_one(config, strategy_cfg, run_id=run_id, progress_callback=lambda c, t, m: _batch_progress(run_id, c, t, m))
                with run_status_lock:
                    if run_id in run_statuses:
                        run_statuses[run_id].update(status="completed", message="Done.")
                _register_run_for_config(_normalize_config_for_fingerprint(run_opts), run_id)
            except RunCancelledError:
                # Distinguish per-run skip from batch-level cancel
                was_skip = False
                with run_status_lock:
                    if run_id in run_statuses and run_statuses[run_id].get("skip_requested"):
                        was_skip = True
                run_dir = RUNS_DIR / run_id
                if run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except OSError:
                        pass
                if was_skip:
                    with run_status_lock:
                        if run_id in run_statuses:
                            run_statuses[run_id].update(
                                status="skipped", run_id=run_id, message="Skipped. Generated files deleted."
                            )
                    continue
                with run_status_lock:
                    if run_id in run_statuses:
                        run_statuses[run_id].update(
                            status="cancelled", run_id=run_id, message="Cancelled. Generated files deleted."
                        )
                with batch_lock:
                    if batch_id in batch_states:
                        batch_states[batch_id]["status"] = "cancelled"
                return
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                with run_status_lock:
                    if run_id in run_statuses:
                        run_statuses[run_id].update(
                            status="failed",
                            error=f"{e}\n\n{tb}",
                        )
                # Continue to the next run instead of aborting the batch
                continue
        with batch_lock:
            if batch_id in batch_states:
                batch_states[batch_id]["status"] = "cancelled" if batch_states[batch_id].get("cancel_requested") else "completed"

    def _batch_progress(run_id: str, c: int, t: int, m: str) -> None:
        with run_status_lock:
            entry = run_statuses.get(run_id)
            if not entry:
                return
            if entry.get("cancel_requested"):
                raise RunCancelledError("Batch cancelled by user")
            if entry.get("skip_requested"):
                raise RunCancelledError("Run skipped by user")
            entry.update(current=c, total=t, message=m)
        # Pause spin-wait (outside lock so status polling still works)
        import time as _time
        while True:
            with run_status_lock:
                entry = run_statuses.get(run_id)
                if not entry:
                    return
                if entry.get("cancel_requested"):
                    raise RunCancelledError("Batch cancelled by user")
                if entry.get("skip_requested"):
                    raise RunCancelledError("Run skipped by user")
                if not entry.get("pause_requested"):
                    break
                entry["status"] = "paused"
            _time.sleep(1)

    threading.Thread(target=job, daemon=True).start()
    return redirect(url_for("comparison_progress", batch_id=batch_id))


@app.route("/run-comparison/<batch_id>/status")
def comparison_status(batch_id: str):
    """JSON status for the batch progress page. Always return run_configs aligned to run_ids (1:1)."""
    with batch_lock:
        state = dict(batch_states.get(batch_id, {}))
    if not state:
        return jsonify({"error": "Batch not found", "status": "unknown"}), 404
    run_ids = state.get("run_ids") or []
    stored_labels = state.get("labels") or []
    run_configs = state.get("run_configs") or []
    if not isinstance(run_configs, list):
        run_configs = []
    # Pad or truncate run_configs to match run_ids (strict 1:1)
    if len(run_configs) < len(run_ids):
        run_configs = run_configs + ([{}] * (len(run_ids) - len(run_configs)))
    else:
        run_configs = run_configs[: len(run_ids)]
    # Rebuild labels from run_configs when possible so we never send short frontend labels
    rebuilt_labels = []
    for i, cfg in enumerate(run_configs):
        try:
            if isinstance(cfg, dict) and cfg:
                rebuilt_labels.append(
                    _format_run_label(_normalize_config_for_fingerprint(cfg))
                )
            else:
                rebuilt_labels.append(
                    stored_labels[i] if i < len(stored_labels) else f"Run {i + 1}"
                )
        except Exception:
            rebuilt_labels.append(
                stored_labels[i] if i < len(stored_labels) else f"Run {i + 1}"
            )
    run_statuses_snapshot = {}
    with run_status_lock:
        for rid in run_ids:
            if rid:
                run_statuses_snapshot[rid] = dict(run_statuses.get(rid, {"status": "unknown"}))
    return jsonify({
        "batch_id": batch_id,
        "run_ids": run_ids,
        "labels": rebuilt_labels,
        "run_configs": run_configs,
        "status": state.get("status", "unknown"),
        "error": state.get("error"),
        "run_statuses": run_statuses_snapshot,
    })


@app.route("/run-comparison/<batch_id>/cancel", methods=["POST"], endpoint="comparison_cancel")
def comparison_cancel(batch_id: str):
    """Request cancellation of the comparison batch. Stops the current run, deletes its files, and does not start further runs."""
    with batch_lock:
        state = dict(batch_states.get(batch_id, {}))
    if not state:
        return jsonify({"ok": False, "error": "Batch not found"}), 404
    if state.get("status") not in ("running",):
        return jsonify({"ok": False, "error": "Batch is not running"}), 400
    with batch_lock:
        if batch_id in batch_states:
            batch_states[batch_id]["cancel_requested"] = True
    run_ids = state.get("run_ids") or []
    with run_status_lock:
        for rid in run_ids:
            if rid and rid in run_statuses and run_statuses[rid].get("status") in ("running", "paused"):
                run_statuses[rid]["cancel_requested"] = True
                run_statuses[rid]["pause_requested"] = False
                break
    return jsonify({"ok": True, "message": "Cancellation requested. Current run will stop and its generated files will be deleted."})


@app.route("/run-comparison/<batch_id>/run/<run_id>/pause", methods=["POST"])
def batch_run_pause(batch_id: str, run_id: str):
    """Pause a currently running run in a batch."""
    with batch_lock:
        state = batch_states.get(batch_id)
    if not state:
        return jsonify({"ok": False, "error": "Batch not found"}), 404
    with run_status_lock:
        entry = run_statuses.get(run_id)
        if not entry:
            return jsonify({"ok": False, "error": "Run not found"}), 404
        if entry.get("status") not in ("running",):
            return jsonify({"ok": False, "error": "Run is not running"}), 400
        entry["pause_requested"] = True
    return jsonify({"ok": True})


@app.route("/run-comparison/<batch_id>/run/<run_id>/resume", methods=["POST"])
def batch_run_resume(batch_id: str, run_id: str):
    """Resume a paused run in a batch."""
    with batch_lock:
        state = batch_states.get(batch_id)
    if not state:
        return jsonify({"ok": False, "error": "Batch not found"}), 404
    with run_status_lock:
        entry = run_statuses.get(run_id)
        if not entry:
            return jsonify({"ok": False, "error": "Run not found"}), 404
        if not entry.get("pause_requested") and entry.get("status") != "paused":
            return jsonify({"ok": False, "error": "Run is not paused"}), 400
        entry["pause_requested"] = False
        entry["status"] = "running"
    return jsonify({"ok": True})


@app.route("/run-comparison/<batch_id>/run/<run_id>/skip", methods=["POST"])
def batch_run_skip(batch_id: str, run_id: str):
    """Skip a currently running or paused run in a batch."""
    with batch_lock:
        state = batch_states.get(batch_id)
    if not state:
        return jsonify({"ok": False, "error": "Batch not found"}), 404
    with run_status_lock:
        entry = run_statuses.get(run_id)
        if not entry:
            return jsonify({"ok": False, "error": "Run not found"}), 404
        if entry.get("status") not in ("running", "paused"):
            return jsonify({"ok": False, "error": "Run is not running or paused"}), 400
        entry["skip_requested"] = True
        entry["pause_requested"] = False
    return jsonify({"ok": True})


@app.route("/run-comparison/<batch_id>/run/<run_id>/delete", methods=["POST"])
def batch_run_delete(batch_id: str, run_id: str):
    """Delete a completed/failed/skipped run's files from disk."""
    with batch_lock:
        state = batch_states.get(batch_id)
    if not state:
        return jsonify({"ok": False, "error": "Batch not found"}), 404
    with run_status_lock:
        entry = run_statuses.get(run_id)
        if not entry:
            return jsonify({"ok": False, "error": "Run not found"}), 404
        if entry.get("status") not in ("completed", "failed", "skipped"):
            return jsonify({"ok": False, "error": "Run must be completed, failed, or skipped to delete"}), 400
    run_dir = _safe_run_path(run_id)
    if not run_dir:
        return jsonify({"ok": False, "error": "Invalid run ID"}), 400
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError as e:
            return jsonify({"ok": False, "error": f"Could not delete: {e}"}), 500
    with run_status_lock:
        if run_id in run_statuses:
            run_statuses[run_id].update(status="deleted", message="Deleted.")
    return jsonify({"ok": True})


def _f1_from_metrics(metrics: Optional[dict]) -> float:
    """F1 from precision and recall; 0 if missing."""
    if not metrics:
        return 0.0
    p = metrics.get("precision") or 0.0
    r = metrics.get("recall") or 0.0
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _existing_run_ids(run_ids: List[str]) -> List[str]:
    """Filter to run_ids that exist on disk with evaluation/metrics.json."""
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    return [rid for rid in run_ids if (RUNS_DIR / rid / "evaluation" / "metrics.json").exists()]


@app.route("/api/config-runs", methods=["POST"])
def api_config_runs():
    """Return run_ids and last_run_id for the given run config (from comparison row). Only includes runs that exist on disk."""
    data = request.get_json(silent=True) or {}
    config = data.get("config") or {}
    fp = _config_fingerprint(_normalize_config_for_fingerprint(config))
    run_ids = _backfill_run_ids_for_fingerprint(fp)
    run_ids = _existing_run_ids(run_ids)
    last_run_id = run_ids[-1] if run_ids else None
    return jsonify({"run_ids": run_ids, "last_run_id": last_run_id})


@app.route("/api/config-last-summary", methods=["POST"])
def api_config_last_summary():
    """Return the last summary for this config, or found=false with message. Uses last run that exists on disk."""
    data = request.get_json(silent=True) or {}
    config = data.get("config") or {}
    fp = _config_fingerprint(_normalize_config_for_fingerprint(config))
    run_ids = _backfill_run_ids_for_fingerprint(fp)
    run_ids = _existing_run_ids(run_ids)
    if not run_ids:
        return jsonify({"found": False, "message": "No data saved for this run."})
    run_id = run_ids[-1]
    summary_path = RUNS_DIR / run_id / "generated" / "summary.txt"
    if not summary_path.exists():
        return jsonify({"found": False, "message": "No summary found for the last run.", "run_id": run_id})
    try:
        summary = summary_path.read_text(encoding="utf-8")
    except Exception:
        return jsonify({"found": False, "message": "Could not read summary.", "run_id": run_id})
    return jsonify({"found": True, "run_id": run_id, "summary": summary})


@app.route("/api/config-analysis", methods=["POST"])
def api_config_analysis():
    """Return analysis for all runs with this config (run_id, label, metrics, f1, etc.).
    Uses registry and backfills from runs on disk. Only includes runs that exist on disk (have metrics).
    """
    data = request.get_json(silent=True) or {}
    config = data.get("config") or {}
    fp = _config_fingerprint(_normalize_config_for_fingerprint(config))
    run_ids = _backfill_run_ids_for_fingerprint(fp)
    if not run_ids:
        return jsonify({"found": False, "message": "No data saved for this run."})
    rows = []

    def _pair_f1(pp, rr):
        try:
            ppf = float(pp); rrf = float(rr)
        except (TypeError, ValueError):
            return None
        return (2 * ppf * rrf / (ppf + rrf)) if (ppf + rrf) else 0.0

    for i, run_id in enumerate(run_ids):
        run_root = RUNS_DIR / run_id
        if not run_root.exists() or not (run_root / "evaluation" / "metrics.json").exists():
            continue
        metrics = load_metrics(run_root)
        f1 = _f1_from_metrics(metrics)
        m = metrics or {}
        rel_block = m.get("relations") or {}
        hier_block = m.get("hierarchy") or {}
        rel_f1 = _pair_f1(rel_block.get("precision"), rel_block.get("recall"))
        hier_f1 = _pair_f1(hier_block.get("precision"), hier_block.get("recall"))
        parts = [x for x in (f1, rel_f1, hier_f1) if x is not None]
        overall_f1 = (sum(parts) / len(parts)) if parts else None
        err = m.get("errors") or {}
        struct = m.get("structural") or {}
        he = struct.get("hierarchy_edges")
        rows.append({
            "run_id": run_id,
            "label": f"Run {len(rows) + 1}",
            "metrics": m,
            "f1": f1,
            "overall_f1": overall_f1,
            "hier_f1": hier_f1,
            "rel_f1": rel_f1,
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "coverage": m.get("coverage"),
            "hallucinations": err.get("hallucinations"),
            "hierarchy_edges": int(he) if he is not None else None,
        })
    if not rows:
        return jsonify({
            "found": False,
            "message": "No run data on disk for this config. The registry lists runs that are missing—copy the run folders from backup into the runs/ directory (e.g. runs/20260224-190026-284a5d34/), not just _config_registry.json.",
        })
    return jsonify({"found": True, "rows": rows})


def _normalize_config_for_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure config has explicit values for all _CONFIG_KEYS so fingerprint is stable (e.g. eval_restrict_to_gold bool)."""
    out = dict(config)
    for k in _CONFIG_KEYS:
        if k not in out:
            out[k] = False if k == "eval_restrict_to_gold" else None
        elif k == "eval_restrict_to_gold":
            v = out[k]
            if isinstance(v, bool):
                pass
            elif v is None:
                out[k] = False
            elif isinstance(v, str):
                out[k] = v.strip().lower() in ("1", "true", "yes", "on")
            else:
                out[k] = bool(v)
    return out


@app.route("/api/configs-last-runs", methods=["POST"])
def api_configs_last_runs():
    """Return last run metrics for each config (one row per config, same order). No run -> row with run_id null."""
    data = request.get_json(silent=True) or {}
    configs = data.get("configs") or []
    if not configs:
        return jsonify({"rows": []})
    registry = _load_config_registry()
    rows = []
    for config in configs:
        normalized = _normalize_config_for_fingerprint(config)
        fp = _config_fingerprint(normalized)
        run_ids = _backfill_run_ids_for_fingerprint(fp)
        run_ids = _existing_run_ids(run_ids)
        run_id = run_ids[-1] if run_ids else None
        if not run_id:
            rows.append({
                "run_id": None,
                "metrics": None,
                "f1": None,
                "precision": None,
                "recall": None,
                "coverage": None,
                "overall_f1": None,
                "hier_f1": None,
                "rel_f1": None,
            })
            continue
        run_root = RUNS_DIR / run_id
        metrics = load_metrics(run_root) if run_root.exists() else None
        f1 = _f1_from_metrics(metrics)
        m = metrics or {}

        def _pair_f1(pp, rr):
            try:
                ppf = float(pp); rrf = float(rr)
            except (TypeError, ValueError):
                return None
            return (2 * ppf * rrf / (ppf + rrf)) if (ppf + rrf) else 0.0

        rel_block = m.get("relations") or {}
        hier_block = m.get("hierarchy") or {}
        rel_f1 = _pair_f1(rel_block.get("precision"), rel_block.get("recall"))
        hier_f1 = _pair_f1(hier_block.get("precision"), hier_block.get("recall"))
        parts = [x for x in (f1, rel_f1, hier_f1) if x is not None]
        overall_f1 = (sum(parts) / len(parts)) if parts else None

        rows.append({
            "run_id": run_id,
            "metrics": m,
            "f1": f1,
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "coverage": m.get("coverage"),
            "overall_f1": overall_f1,
            "hier_f1": hier_f1,
            "rel_f1": rel_f1,
        })
    return jsonify({"rows": rows})


@app.route("/run-comparison/<batch_id>/progress")
def comparison_progress(batch_id: str):
    """Progress page for a batch of comparison runs."""
    with batch_lock:
        if batch_id not in batch_states:
            return "Batch not found", 404
    return render_template("comparison_progress.html", batch_id=batch_id)


@app.route("/run-comparison/<batch_id>/analyze", endpoint="comparison_analyze")
def comparison_analyze(batch_id: str):
    """Compare all runs in the batch and highlight the best one (by Overall F1)."""
    with batch_lock:
        state = dict(batch_states.get(batch_id, {}))
    if not state:
        return "Batch not found", 404
    run_ids = state.get("run_ids") or []
    labels = state.get("labels") or []
    rows = []
    best_idx = -1
    best_f1 = -1.0

    def _pair_f1(pp, rr):
        try:
            ppf = float(pp); rrf = float(rr)
        except (TypeError, ValueError):
            return None
        return (2 * ppf * rrf / (ppf + rrf)) if (ppf + rrf) else 0.0

    for i, run_id in enumerate(run_ids):
        if not run_id:
            continue
        run_root = RUNS_DIR / run_id
        metrics = load_metrics(run_root) if run_root.exists() else None
        f1 = _f1_from_metrics(metrics)
        m = metrics or {}
        rel_block = m.get("relations") or {}
        hier_block = m.get("hierarchy") or {}
        rel_f1 = _pair_f1(rel_block.get("precision"), rel_block.get("recall"))
        hier_f1 = _pair_f1(hier_block.get("precision"), hier_block.get("recall"))
        parts = [x for x in (f1, rel_f1, hier_f1) if x is not None]
        overall_f1 = (sum(parts) / len(parts)) if parts else None
        if overall_f1 is not None and overall_f1 > best_f1:
            best_f1 = overall_f1
            best_idx = len(rows)
        err = m.get("errors") or {}
        struct = m.get("structural") or {}
        he = struct.get("hierarchy_edges")
        rows.append({
            "run_id": run_id,
            "label": labels[i] if i < len(labels) else f"Run {i + 1}",
            "metrics": m,
            "f1": f1,
            "overall_f1": overall_f1,
            "hier_f1": hier_f1,
            "rel_f1": rel_f1,
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "coverage": m.get("coverage"),
            "hallucinations": err.get("hallucinations"),
            "hierarchy_edges": int(he) if he is not None else None,
        })
    return render_template(
        "comparison_analyze.html",
        batch_id=batch_id,
        rows=rows,
        best_index=best_idx,
    )


@app.route("/run/<run_id>/status")
def run_status(run_id: str):
    """Return current run status (for progress polling)."""
    with run_status_lock:
        data = dict(run_statuses.get(run_id, {"status": "unknown"}))
    # If the server was restarted (debug reloader / crash), in-memory state is lost.
    # Provide a best-effort fallback based on run artifacts on disk so the UI can recover.
    if data.get("status") == "unknown":
        run_root = _safe_run_path(run_id)
        if not run_root:
            return jsonify({"status": "unknown", "error": "Invalid run ID"}), 400
        if (run_root / "generated" / "summary.txt").exists() and (run_root / "evaluation" / "metrics.json").exists():
            return jsonify({"status": "completed", "run_id": run_id, "message": "Done."})
        if (run_root / "metadata.json").exists():
            return jsonify({"status": "running", "run_id": run_id, "message": "Run in progress (status restored)."})
    return jsonify(data)


@app.route("/run/<run_id>/progress")
def run_progress(run_id: str):
    """Show progress page that polls status and redirects when done."""
    return render_template("progress.html", run_id=run_id)


@app.route("/pipeline", methods=["GET"])
def pipeline_image():
    """Serve the pipeline diagram for the UI modal (HTML preferred, then PNG/SVG fallback)."""
    if _PIPELINE_HTML_PATH.exists():
        return send_file(
            _PIPELINE_HTML_PATH.resolve(),
            mimetype="text/html",
        )
    if _PIPELINE_PNG_PATH.exists():
        return send_file(
            _PIPELINE_PNG_PATH.resolve(),
            mimetype="image/png",
            as_attachment=False,
            download_name=_PIPELINE_PNG_PATH.name,
        )
    if _PIPELINE_SVG_PATH.exists():
        return send_file(
            _PIPELINE_SVG_PATH.resolve(),
            mimetype="image/svg+xml",
            as_attachment=False,
            download_name=_PIPELINE_SVG_PATH.name,
        )
    return "Pipeline diagram not found", 404


@app.route("/run/<run_id>/cancel", methods=["POST"], endpoint="run_cancel")
def run_cancel(run_id: str):
    """Request cancellation of a running experiment. Run folder is deleted when the job stops."""
    with run_status_lock:
        if run_id not in run_statuses:
            return jsonify({"ok": False, "error": "Run not found"}), 404
        if run_statuses[run_id].get("status") != "running":
            return jsonify({"ok": False, "error": "Run is not running"}), 400
        run_statuses[run_id]["cancel_requested"] = True
    return jsonify({"ok": True, "message": "Cancellation requested. Run will stop and generated files will be deleted."})


@app.route("/analyze", methods=["POST"])
def analyze():
    baseline_id = (request.form.get("baseline_run_id") or "").strip()
    improved_id = (request.form.get("improved_run_id") or "").strip()
    if not baseline_id or not improved_id:
        return render_template(
            "index.html",
            error="Select both baseline and improved runs for comparison.",
            default_gold_path=get_default_gold_path(),
            run_ids=list_run_ids(),
        )
    baseline_path = _safe_run_path(baseline_id)
    improved_path = _safe_run_path(improved_id)
    if not baseline_path or not improved_path:
        return render_template(
            "index.html",
            error="Invalid run ID.",
            default_gold_path=get_default_gold_path(),
            run_ids=list_run_ids(),
        )
    try:
        baseline = load_run(baseline_path)
        improved = load_run(improved_path)
        write_comparison(improved.root, baseline.metrics, improved.metrics)
        write_plot(improved.root, baseline.metrics, improved.metrics)
        return redirect(url_for("results", run_id=improved.run_id))
    except FileNotFoundError as exc:
        return render_template(
            "index.html",
            error=str(exc),
            default_gold_path=get_default_gold_path(),
            run_ids=list_run_ids(),
        )


@app.route("/results/<run_id>", methods=["GET"])
def results(run_id: str):
    run_root = _safe_run_path(run_id)
    if not run_root:
        return "Invalid run ID", 400
    metrics = load_metrics(run_root)
    analysis = load_analysis(run_root)
    prompts_dir = run_root / "prompts"
    prompt_files = sorted(prompts_dir.glob("prompt_chunk_*.txt")) if prompts_dir.exists() else []
    artifacts = {
        "ontology": run_root / "generated" / "ontology.json",
        "summary": run_root / "generated" / "summary.txt",
        "evaluation": run_root / "evaluation" / "metrics.json",
        "prompts_dir": prompts_dir.resolve() if prompts_dir.exists() else prompts_dir,
        "prompt_files": prompt_files,
    }
    return render_template(
        "results.html",
        run_id=run_id,
        metrics=metrics,
        artifacts=artifacts,
        analysis=analysis,
    )


@app.route("/api/run/<run_id>/cluster", methods=["POST"])
def api_cluster_results(run_id: str):
    """Cluster extracted ontology classes using semantic embeddings + Ward's hierarchical clustering."""
    run_root = _safe_run_path(run_id)
    if not run_root:
        return jsonify({"error": "Invalid run ID"}), 400
    ontology_path = run_root / "generated" / "ontology.json"
    if not ontology_path.exists():
        return jsonify({"error": "Ontology not found — run the pipeline first"}), 404
    try:
        data = json.loads(ontology_path.read_text(encoding="utf-8"))
        classes = data.get("classes", [])
        if not classes:
            return jsonify({"error": "No classes found in ontology"}), 400
        from src.analysis.cluster_results import cluster_classes
        result = cluster_classes(classes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Clustering failed: {str(e)}"}), 500


@app.route("/api/run/<run_id>/ontology-graph", methods=["GET"])
def api_ontology_graph(run_id: str):
    """Return ontology as Cytoscape.js-compatible graph elements (nodes + edges)."""
    run_root = _safe_run_path(run_id)
    if not run_root:
        return jsonify({"error": "Invalid run ID"}), 400
    ontology_path = run_root / "generated" / "ontology.json"
    if not ontology_path.exists():
        return jsonify({"error": "Ontology not found"}), 404
    try:
        data = json.loads(ontology_path.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"error": "Could not parse ontology"}), 500

    nodes = []
    edges = []
    class_labels: set = set()

    for c in data.get("classes", []):
        label = c.get("label", "")
        if not label or label in class_labels:
            continue
        class_labels.add(label)
        nodes.append({"data": {
            "id": label, "label": label, "type": "class",
            "definition": c.get("definition") or "",
            "stratum": c.get("stratum") or "core",
            "evidence": c.get("evidence") or "",
        }})

    def _ensure_node(lbl: str):
        if lbl and lbl not in class_labels:
            class_labels.add(lbl)
            nodes.append({"data": {
                "id": lbl, "label": lbl, "type": "class",
                "definition": "", "stratum": "inferred", "evidence": "",
            }})

    for i, r in enumerate(data.get("relations", [])):
        domain = r.get("domain", "")
        range_ = r.get("range", "")
        _ensure_node(domain)
        _ensure_node(range_)
        if domain and range_:
            edges.append({"data": {
                "id": f"rel_{i}", "source": domain, "target": range_,
                "label": r.get("label", ""), "type": "relation",
                "definition": r.get("definition") or "",
                "evidence": r.get("evidence") or "",
            }})

    for i, h in enumerate(data.get("hierarchy", [])):
        sub = h.get("subClass", "")
        sup = h.get("superClass", "")
        _ensure_node(sub)
        _ensure_node(sup)
        if sub and sup:
            edges.append({"data": {
                "id": f"hier_{i}", "source": sub, "target": sup,
                "label": "subClassOf", "type": "hierarchy",
                "evidence": h.get("evidence") or "",
            }})

    return jsonify({
        "nodes": nodes, "edges": edges,
        "stats": {
            "classes": len(data.get("classes", [])),
            "relations": len(data.get("relations", [])),
            "hierarchy": len(data.get("hierarchy", [])),
        },
    })


@app.route("/results/<run_id>/artifact/<artifact_type>", methods=["GET"])
def artifact_file(run_id: str, artifact_type: str):
    """Serve an artifact file for viewing/download (ontology, summary, evaluation)."""
    run_root = _safe_run_path(run_id)
    if not run_root or not run_root.exists() or not (run_root / "metadata.json").exists():
        return "Run not found", 404
    paths = {
        "ontology": run_root / "generated" / "ontology.json",
        "summary": run_root / "generated" / "summary.txt",
        "evaluation": run_root / "evaluation" / "metrics.json",
    }
    path = paths.get(artifact_type)
    if not path or not path.exists():
        return "File not found", 404
    mimetypes = {
        "ontology": "application/json",
        "summary": "text/plain; charset=utf-8",
        "evaluation": "application/json",
    }
    try:
        return send_file(
            path.resolve(),
            mimetype=mimetypes.get(artifact_type, "application/octet-stream"),
            as_attachment=False,
            download_name=path.name,
        )
    except FileNotFoundError:
        return "File not found", 404


# ── Cluster Results API ──────────────────────────────────────────────
@app.route("/api/run/<run_id>/cluster", methods=["POST"])
def api_cluster(run_id):
    """Run semantic clustering on the ontology classes of a completed run."""
    from src.analysis.cluster_results import cluster_classes

    run_path = _safe_run_path(run_id)
    if run_path is None:
        return jsonify({"error": "Invalid run ID"}), 400

    ontology_file = run_path / "generated" / "ontology.json"
    if not ontology_file.exists():
        return jsonify({"error": "ontology.json not found for this run"}), 404

    try:
        ontology = _safe_json_loads(ontology_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": f"Could not parse ontology.json: {exc}"}), 500

    classes = ontology.get("classes", [])
    if not classes:
        return jsonify({"error": "No classes found in ontology.json"}), 400

    try:
        result = cluster_classes(classes)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Clustering failed: {exc}"}), 500


# ── Ontology Engineering ──────────────────────────────────────────────
# Three-stage pipeline: Merge → Cluster → LLM Reconstruct

_OE_DIR = _project_root / "data" / "ontology_engineering"


def _save_oe_run_as_full_run(
    reconstructed: Dict[str, Any],
    source_run_ids: List[str],
    provider: str,
    model: Optional[str],
    session_id: str,
    n_clusters: int,
    log_entries: Optional[List[Dict[str, Any]]] = None,
    session_dir: Optional[Path] = None,
) -> Optional[str]:
    """Persist a reconstructed (merge + cluster + LLM) ontology as a standard run
    under ``runs/<run_id>/`` with full evaluation against the gold standard.

    Mirrors the artifact layout produced by ``run_experiments.run_one`` so the
    existing results page (`/results/<run_id>`) works without modification.
    Returns the new run_id, or None on failure.
    """
    from src.experiments.run_registry import create_run_dirs
    from src.experiments.metadata import build_metadata, write_metadata
    from src.experiments.artifacts import write_run_summary
    from src.experiments.run_experiments import _flatten_metrics_for_table
    from src.ontology.export import ontology_from_dict, write_ontology_json
    from src.evaluation.gold_standard import load_gold_standard
    from src.evaluation.align import align_entities
    from src.evaluation.metrics import (
        compute_coverage,
        compute_precision_recall,
        compute_structural_metrics,
        compute_relation_metrics,
        compute_hierarchy_metrics,
    )
    from src.evaluation.errors import error_taxonomy
    from src.evaluation.report import write_metrics, write_table

    try:
        ontology = ontology_from_dict(reconstructed)
    except Exception:
        traceback.print_exc()
        return None

    run_paths = create_run_dirs(base_dir=str(RUNS_DIR), run_id=None)

    # Primary ontology artefact
    write_ontology_json(run_paths.generated / "ontology.json", ontology)

    # Save cluster-completion prompts (one file per cluster)
    if log_entries:
        prompts_dir = run_paths.prompts
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for entry in log_entries:
            cluster_id = entry.get("cluster_id", "unknown")
            cluster_name = entry.get("cluster_name", "unknown")
            prompt_text = entry.get("prompt", "")
            if prompt_text:
                safe_name = str(cluster_name).replace(" ", "_").replace("/", "_")[:60]
                filename = f"cluster_{cluster_id}_{safe_name}.txt"
                (prompts_dir / filename).write_text(
                    prompt_text, encoding="utf-8"
                )

    # Copy cluster report from OE session into run artifacts
    if session_dir:
        cluster_src = Path(session_dir) / "cluster_data.json"
        if cluster_src.exists():
            import shutil
            shutil.copy2(str(cluster_src), str(run_paths.generated / "cluster_data.json"))

    # Synthetic config & strategy so write_run_summary renders a coherent header.
    # Keys mirror the shape produced by StrategyConfig runs, but describe this
    # run as an Ontology Engineering (cluster-completion) reconstruction.
    oe_config: Dict[str, Any] = {
        "llm_provider": provider or "",
        "llm_model": model or "",
        "strategies": [{"prompt_strategy": "ontology_engineering"}],
        "method": "ontology_engineering",
        "source_runs": source_run_ids,
        "n_clusters": n_clusters,
        "session_id": session_id,
    }

    class _OEStrategy:
        prompt_strategy = "ontology_engineering"
        gold_standard_path = get_default_gold_path() or None

    strategy = _OEStrategy()

    # Evaluation — best effort, gold may be absent.
    metrics: Dict[str, Any] = {}
    gold_path = get_default_gold_path()
    gold: Optional[Dict[str, Any]] = None
    if gold_path:
        try:
            gold = load_gold_standard(gold_path)
        except Exception:
            traceback.print_exc()
            gold = None

    if gold:
        try:
            gold_classes = gold.get("classes", [])
            generated_classes = [c.__dict__ for c in ontology.classes]
            alignment = align_entities(generated=generated_classes, gold=gold_classes)
            matched = alignment["matched_exact"] + alignment["matched_semantic"]
            unmatched = alignment["unmatched"]
            unique_gold_matched = len(alignment.get("gold_labels_matched", set()))

            metrics = {
                "coverage": compute_coverage(unique_gold_matched, gold_classes),
                **compute_precision_recall(
                    len(matched), generated_classes, unique_gold_matched, gold_classes
                ),
                "errors": error_taxonomy(
                    unmatched, [],
                    relations=[r.__dict__ for r in ontology.relations],
                ),
                "structural": compute_structural_metrics(ontology),
            }
            if gold.get("relations"):
                metrics["relations"] = compute_relation_metrics(
                    [r.__dict__ for r in ontology.relations],
                    gold.get("relations", []),
                )
            if gold.get("hierarchy"):
                metrics["hierarchy"] = compute_hierarchy_metrics(
                    ontology.hierarchy, gold.get("hierarchy", []),
                )

            hallucinated = [
                {"label": u.get("label"), "definition": u.get("definition")}
                for u in unmatched
            ]
            (run_paths.evaluation / "hallucinated_classes.json").write_text(
                json.dumps(hallucinated, indent=2), encoding="utf-8"
            )
        except Exception:
            traceback.print_exc()

    # Metrics files (written even when empty so the results page renders cleanly)
    write_metrics(run_paths.evaluation / "metrics.json", metrics)
    table_rows = _flatten_metrics_for_table(metrics) if metrics else []
    write_table(run_paths.evaluation / "table.csv", table_rows)

    # Metadata — treat the source runs as "input papers" so the results page lists them.
    source_docs = [{"source": "", "id": rid} for rid in source_run_ids]
    metadata = build_metadata(
        oe_config, run_paths.run_id, source_run_ids, docs=source_docs
    )
    metadata["oe_session_id"] = session_id
    metadata["oe_source_runs"] = source_run_ids
    metadata["oe_method"] = "merge + cluster + LLM reconstruction"
    write_metadata(run_paths.root / "metadata.json", metadata)

    # Human-readable summary (falls back to a minimal summary on failure)
    try:
        write_run_summary(
            run_paths.generated,
            run_paths.run_id,
            ontology,
            metrics,
            oe_config,
            strategy,
            docs=source_docs,
            timestamp_utc=metadata.get("timestamp_utc"),
        )
    except Exception:
        traceback.print_exc()
        (run_paths.generated / "summary.txt").write_text(
            "\n".join([
                "Ontology Engineering — reconstructed run",
                f"Run ID: {run_paths.run_id}",
                f"Source runs: {', '.join(source_run_ids)}",
                f"LLM: {provider} {model or ''}".strip(),
                f"Clusters processed: {n_clusters}",
                f"Classes: {len(ontology.classes)}",
                f"Relations: {len(ontology.relations)}",
                f"Hierarchy edges: {len(ontology.hierarchy)}",
            ]),
            encoding="utf-8",
        )

    return run_paths.run_id


def _list_runs_with_ontology() -> List[Dict[str, Any]]:
    """Return runs that have a generated/ontology.json file, with metadata for the UI.

    Each row also carries the latest evaluation metrics (f1/precision/recall/coverage)
    so the Ontology Engineering page can show a "Last result" column without an
    extra round-trip.
    """
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        ontology_path = p / "generated" / "ontology.json"
        if not ontology_path.exists():
            continue
        meta = {}
        meta_path = p / "metadata.json"
        if meta_path.exists():
            try:
                meta = _safe_json_loads(meta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
        # Strategy is stored under metadata.config.strategies[0].prompt_strategy for
        # normal runs; fall back to top-level `strategy` for older/legacy runs.
        strategy = meta.get("strategy", "") or (
            (meta.get("config") or {}).get("strategies", [{}])[0].get("prompt_strategy", "")
        )
        papers = meta.get("input_papers") or meta.get("papers") or []
        if isinstance(papers, list):
            names = []
            for pp in papers[:3]:
                if isinstance(pp, dict):
                    names.append((pp.get("stem") or pp.get("name", ""))[:25])
                elif isinstance(pp, str):
                    names.append(Path(pp).stem[:25])
            papers_short = ", ".join(names)
        else:
            papers_short = str(papers)[:40]
        # Count classes
        n_classes = 0
        try:
            onto_data = _safe_json_loads(ontology_path.read_text(encoding="utf-8")) or {}
            n_classes = len(onto_data.get("classes", []))
        except Exception:
            pass
        # Latest metrics (optional — runs without evaluation still show up).
        # We also compute hierarchy F1, relation F1, and an Overall F1 (mean of
        # class/hier/rel F1) so the Stage-1 run list can rank by the same signal
        # the Compare Metrics modal uses.
        metrics = load_metrics(p)
        f1_val: Optional[float] = None
        precision_val: Optional[float] = None
        recall_val: Optional[float] = None
        coverage_val: Optional[float] = None
        rel_f1_val: Optional[float] = None
        hier_f1_val: Optional[float] = None
        overall_f1_val: Optional[float] = None

        def _pair_f1(pp: Any, rr: Any) -> Optional[float]:
            try:
                ppf = float(pp); rrf = float(rr)
            except (TypeError, ValueError):
                return None
            return (2 * ppf * rrf / (ppf + rrf)) if (ppf + rrf) else 0.0

        if metrics:
            precision_val = metrics.get("precision")
            recall_val = metrics.get("recall")
            coverage_val = metrics.get("coverage")
            f1_val = _f1_from_metrics(metrics)
            rel_block = metrics.get("relations") or {}
            hier_block = metrics.get("hierarchy") or {}
            rel_f1_val = _pair_f1(rel_block.get("precision"), rel_block.get("recall"))
            hier_f1_val = _pair_f1(hier_block.get("precision"), hier_block.get("recall"))
            parts = [x for x in (f1_val, rel_f1_val, hier_f1_val) if x is not None]
            overall_f1_val = (sum(parts) / len(parts)) if parts else None
        # Derive pipeline mode from config flags
        cfg = meta.get("config") or {}
        if cfg.get("schema_guided_completion"):
            pipeline_mode = "schema"
        elif cfg.get("prompt_vocab_guardrails") or cfg.get("medical_ner_anchor") or cfg.get("eval_restrict_to_gold"):
            pipeline_mode = "guided"
        elif cfg.get("method") == "ontology_engineering":
            pipeline_mode = "oe"
        else:
            pipeline_mode = "strict"

        runs.append({
            "id": p.name,
            "strategy": strategy,
            "papers": papers_short,
            "n_classes": n_classes,
            "has_metrics": metrics is not None,
            "f1": f1_val,
            "precision": precision_val,
            "recall": recall_val,
            "coverage": coverage_val,
            "rel_f1": rel_f1_val,
            "hier_f1": hier_f1_val,
            "overall_f1": overall_f1_val,
            "pipeline_mode": pipeline_mode,
            "label": f"{p.name} — {_STRATEGY_LABELS.get(strategy, strategy)} ({n_classes} classes)",
        })
    return runs


@app.route("/ontology-engineering")
def ontology_engineering():
    """Ontology Engineering page — merge, cluster, reconstruct pipeline."""
    runs = _list_runs_with_ontology()
    return render_template("ontology_engineering.html", runs=runs)


@app.route("/api/ontology-engineering/merge", methods=["POST"])
def api_oe_merge():
    """Merge selected ontologies from multiple runs into one."""
    from src.ontology.merge import merge_ontologies as _merge

    data = request.get_json(force=True)
    run_ids = data.get("run_ids", [])
    if not run_ids or len(run_ids) < 1:
        return jsonify({"error": "Select at least one run to merge"}), 400

    ontology_dicts = []
    for rid in run_ids:
        run_path = _safe_run_path(rid)
        if not run_path:
            return jsonify({"error": f"Invalid run ID: {rid}"}), 400
        onto_file = run_path / "generated" / "ontology.json"
        if not onto_file.exists():
            return jsonify({"error": f"No ontology.json for run {rid}"}), 404
        try:
            ontology_dicts.append(
                _safe_json_loads(onto_file.read_text(encoding="utf-8"))
            )
        except Exception as e:
            return jsonify({"error": f"Could not read ontology for {rid}: {e}"}), 500

    session_id = f"oe-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    session_dir = _OE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        merged = _merge(ontology_dicts, metadata={
            "source_runs": run_ids,
            "method": "merge",
        })
    except Exception as e:
        return jsonify({"error": f"Merge failed: {e}"}), 500

    # Save artifacts
    (session_dir / "merged_ontology.json").write_text(
        json.dumps(merged, indent=2, default=str), encoding="utf-8"
    )
    stats = {
        "source_runs": run_ids,
        "total_classes": len(merged.get("classes", [])),
        "total_relations": len(merged.get("relations", [])),
        "total_hierarchy": len(merged.get("hierarchy", [])),
    }
    (session_dir / "merge_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    return jsonify({"ok": True, "session_id": session_id, "stats": stats})


@app.route("/api/ontology-engineering/cluster", methods=["POST"])
def api_oe_cluster():
    """Cluster the merged ontology classes."""
    from src.analysis.cluster_results import cluster_classes

    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return jsonify({"error": "Invalid session ID"}), 400

    session_dir = _OE_DIR / session_id
    merged_path = session_dir / "merged_ontology.json"
    if not merged_path.exists():
        return jsonify({"error": "Merged ontology not found — run merge first"}), 404

    try:
        merged = _safe_json_loads(merged_path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"Could not read merged ontology: {e}"}), 500

    classes = merged.get("classes", [])
    if len(classes) < 5:
        return jsonify({"error": f"Need at least 5 classes to cluster (found {len(classes)})"}), 400

    try:
        result = cluster_classes(classes)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Clustering failed: {e}"}), 500

    # Save cluster data
    (session_dir / "cluster_data.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    return jsonify(result)


@app.route("/api/ontology-engineering/reconstruct", methods=["POST"])
def api_oe_reconstruct():
    """Start async LLM cluster-to-ontology completion."""
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    provider = data.get("provider", "openai")
    model = data.get("model", "") or None

    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return jsonify({"error": "Invalid session ID"}), 400

    session_dir = _OE_DIR / session_id
    merged_path = session_dir / "merged_ontology.json"
    cluster_path = session_dir / "cluster_data.json"

    if not merged_path.exists() or not cluster_path.exists():
        return jsonify({"error": "Merged ontology or cluster data not found"}), 404

    try:
        merged = _safe_json_loads(merged_path.read_text(encoding="utf-8"))
        cluster_data = _safe_json_loads(cluster_path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"Could not read data: {e}"}), 500

    task_id = f"oe-recon-{uuid4().hex[:8]}"

    with run_status_lock:
        run_statuses[task_id] = {
            "status": "running",
            "current": 0,
            "total": len(cluster_data.get("clusters", [])),
            "message": "Starting reconstruction...",
            "task_id": task_id,
            "session_id": session_id,
            "error": None,
        }

    def job():
        from src.ontology.cluster_completion import run_cluster_completion

        def progress(current, total, message):
            with run_status_lock:
                if task_id in run_statuses:
                    run_statuses[task_id].update(
                        current=current, total=total, message=message,
                    )

        try:
            result, log_entries = run_cluster_completion(
                merged_ontology=merged,
                cluster_data=cluster_data,
                provider=provider,
                model=model,
                progress_callback=progress,
            )

            # Save session artefacts (used by the inline graph preview on the OE page)
            (session_dir / "reconstructed_ontology.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            (session_dir / "reconstruction_log.json").write_text(
                json.dumps(log_entries, indent=2, default=str), encoding="utf-8"
            )

            # Promote this reconstruction to a standard run with evaluation so it
            # appears in comparisons, /results/, and the normal run listings.
            source_run_ids = list(
                (merged.get("metadata") or {}).get("source_runs", [])
            )
            new_run_id: Optional[str] = None
            try:
                new_run_id = _save_oe_run_as_full_run(
                    reconstructed=result,
                    source_run_ids=source_run_ids,
                    provider=provider,
                    model=model,
                    session_id=session_id,
                    n_clusters=len(cluster_data.get("clusters", [])),
                    log_entries=log_entries,
                    session_dir=session_dir,
                )
            except Exception:
                traceback.print_exc()

            with run_status_lock:
                if task_id in run_statuses:
                    update = {
                        "status": "completed",
                        "message": (
                            f"Reconstruction complete. Saved as run {new_run_id}."
                            if new_run_id
                            else "Reconstruction complete."
                        ),
                        "current": run_statuses[task_id]["total"],
                    }
                    if new_run_id:
                        update["run_id"] = new_run_id
                        update["results_url"] = f"/results/{new_run_id}"
                    run_statuses[task_id].update(**update)
        except Exception as e:
            traceback.print_exc()
            with run_status_lock:
                if task_id in run_statuses:
                    run_statuses[task_id].update(
                        status="failed", error=str(e), message=f"Failed: {e}",
                    )

    threading.Thread(target=job, daemon=True).start()

    return jsonify({
        "task_id": task_id,
        "session_id": session_id,
        "status_url": f"/api/ontology-engineering/reconstruct/{task_id}/status",
    })


@app.route("/api/ontology-engineering/reconstruct/<task_id>/status")
def api_oe_reconstruct_status(task_id):
    """Poll reconstruction task status."""
    with run_status_lock:
        data = dict(run_statuses.get(task_id, {"status": "unknown"}))
    return jsonify(data)


@app.route("/api/ontology-engineering/sessions")
def api_oe_sessions():
    """List completed OE sessions (those with a reconstructed ontology)."""
    if not _OE_DIR.exists():
        return jsonify([])
    sessions = []
    for p in sorted(_OE_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        recon_path = p / "reconstructed_ontology.json"
        if not recon_path.exists():
            continue
        merge_stats = {}
        ms_path = p / "merge_stats.json"
        if ms_path.exists():
            try:
                merge_stats = _safe_json_loads(ms_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
        # Count classes in reconstructed ontology
        n_classes = 0
        try:
            recon = _safe_json_loads(recon_path.read_text(encoding="utf-8")) or {}
            n_classes = len(recon.get("classes", []))
        except Exception:
            pass
        # Find the promoted run_id if it exists
        run_id = None
        source_runs = merge_stats.get("source_runs", [])
        for rp in sorted(RUNS_DIR.iterdir(), reverse=True) if RUNS_DIR.exists() else []:
            meta_path = rp / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = _safe_json_loads(meta_path.read_text(encoding="utf-8")) or {}
                if meta.get("oe_session_id") == p.name:
                    run_id = rp.name
                    source_runs = meta.get("oe_source_runs", source_runs)
                    break
            except Exception:
                continue
        sessions.append({
            "session_id": p.name,
            "n_classes": n_classes,
            "total_source_classes": merge_stats.get("total_classes", 0),
            "source_runs": source_runs,
            "run_id": run_id,
        })
    return jsonify(sessions)


@app.route("/api/ontology-engineering/cluster-sessions")
def api_oe_cluster_sessions():
    """List OE sessions that have cluster data."""
    if not _OE_DIR.exists():
        return jsonify([])
    sessions = []
    for p in sorted(_OE_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        cluster_path = p / "cluster_data.json"
        if not cluster_path.exists():
            continue
        merge_stats = {}
        ms_path = p / "merge_stats.json"
        if ms_path.exists():
            try:
                merge_stats = _safe_json_loads(ms_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
        # Read cluster summary
        n_clusters = 0
        total_classes = 0
        try:
            cd = _safe_json_loads(cluster_path.read_text(encoding="utf-8")) or {}
            n_clusters = cd.get("optimal_k", len(cd.get("clusters", [])))
            total_classes = cd.get("total_classes", 0)
        except Exception:
            pass
        source_runs = merge_stats.get("source_runs", [])
        sessions.append({
            "session_id": p.name,
            "n_clusters": n_clusters,
            "total_classes": total_classes or merge_stats.get("total_classes", 0),
            "source_runs": source_runs,
        })
    return jsonify(sessions)


@app.route("/api/ontology-engineering/<session_id>/cluster-data")
def api_oe_cluster_data(session_id):
    """Return cluster data for a given OE session."""
    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    cluster_path = _OE_DIR / session_id / "cluster_data.json"
    if not cluster_path.exists():
        return jsonify({"error": "Cluster data not found"}), 404
    try:
        data = _safe_json_loads(cluster_path.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"error": "Could not parse cluster data"}), 500
    return jsonify(data)


@app.route("/api/ontology-engineering/<session_id>/result")
def api_oe_result(session_id):
    """Return the reconstructed ontology as Cytoscape-compatible graph elements."""
    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return jsonify({"error": "Invalid session ID"}), 400

    session_dir = _OE_DIR / session_id
    recon_path = session_dir / "reconstructed_ontology.json"
    if not recon_path.exists():
        return jsonify({"error": "Reconstructed ontology not found"}), 404

    try:
        data = _safe_json_loads(recon_path.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"error": "Could not parse reconstructed ontology"}), 500

    # Build Cytoscape graph elements (same logic as api_ontology_graph)
    nodes = []
    edges = []
    class_labels: set = set()

    for c in data.get("classes", []):
        label = c.get("label", "")
        if not label or label in class_labels:
            continue
        class_labels.add(label)
        is_inferred = label.startswith("[inferred]")
        nodes.append({"data": {
            "id": label, "label": label, "type": "class",
            "definition": c.get("definition") or "",
            "stratum": "inferred" if is_inferred else (c.get("stratum") or "core"),
            "evidence": c.get("evidence") or "",
        }})

    def _ensure_node(lbl: str):
        if lbl and lbl not in class_labels:
            class_labels.add(lbl)
            nodes.append({"data": {
                "id": lbl, "label": lbl, "type": "class",
                "definition": "", "stratum": "inferred", "evidence": "",
            }})

    for i, r in enumerate(data.get("relations", [])):
        domain = r.get("domain", "")
        range_ = r.get("range", "")
        _ensure_node(domain)
        _ensure_node(range_)
        if domain and range_:
            edges.append({"data": {
                "id": f"rel_{i}", "source": domain, "target": range_,
                "label": r.get("label", ""), "type": "relation",
                "definition": r.get("definition") or "",
                "evidence": r.get("evidence") or "",
            }})

    for i, h in enumerate(data.get("hierarchy", [])):
        sub = h.get("subClass", "")
        sup = h.get("superClass", "")
        _ensure_node(sub)
        _ensure_node(sup)
        if sub and sup:
            edges.append({"data": {
                "id": f"hier_{i}", "source": sub, "target": sup,
                "label": "subClassOf", "type": "hierarchy",
                "evidence": h.get("evidence") or "",
            }})

    return jsonify({
        "nodes": nodes, "edges": edges,
        "stats": {
            "classes": len(data.get("classes", [])),
            "relations": len(data.get("relations", [])),
            "hierarchy": len(data.get("hierarchy", [])),
            "inferred_classes": sum(
                1 for c in data.get("classes", [])
                if (c.get("label") or "").startswith("[inferred]")
            ),
        },
    })


@app.route("/api/ontology-engineering/<session_id>/ttl")
def api_oe_ttl(session_id):
    """Return the reconstructed ontology as Turtle (TTL) format."""
    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return "Invalid session ID", 400

    recon_path = _OE_DIR / session_id / "reconstructed_ontology.json"
    if not recon_path.exists():
        return "Reconstructed ontology not found", 404

    try:
        data = _safe_json_loads(recon_path.read_text(encoding="utf-8"))
    except Exception:
        return "Could not parse reconstructed ontology", 500

    from src.ontology.export import ontology_to_ttl
    ttl = ontology_to_ttl(data)
    return app.response_class(
        ttl,
        mimetype="text/turtle",
        headers={"Content-Disposition": f"attachment; filename={session_id}.ttl"},
    )


# ── Qualitative LLM-as-judge evaluation for OE runs ──────────────────

_QEVAL_ID_RE = re.compile(r"^[\w.-]+$")


def _resolve_qeval_target(target_id: str):
    """Resolve a judge-target id to (ontology_path, artifacts_dir, kind).

    Accepts either an OE session id (``oe-...``) or a regular run id. For an
    OE session we read from ``<oe>/reconstructed_ontology.json`` and write
    verdicts alongside it. For a regular run we read from
    ``runs/<run_id>/generated/ontology.json`` and write verdicts under
    ``runs/<run_id>/qualitative_eval/`` so the run artefact layout stays
    self-contained.

    Returns ``(ontology_path, artifacts_dir, kind)`` where kind is either
    ``"oe"`` or ``"run"``. Raises FileNotFoundError / ValueError on bad input.
    """
    if not target_id or not _QEVAL_ID_RE.match(target_id):
        raise ValueError("Invalid id")
    if target_id.startswith("oe-"):
        session_dir = _OE_DIR / target_id
        recon_path = session_dir / "reconstructed_ontology.json"
        if not recon_path.exists():
            raise FileNotFoundError("Reconstructed ontology not found — run reconstruction first")
        return recon_path, session_dir, "oe"
    run_dir = RUNS_DIR / target_id
    onto_path = run_dir / "generated" / "ontology.json"
    if not onto_path.exists():
        raise FileNotFoundError("Run has no generated/ontology.json")
    artifacts_dir = run_dir / "qualitative_eval"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return onto_path, artifacts_dir, "run"


@app.route("/api/ontology-engineering/judgeable-runs")
def api_oe_judgeable_runs():
    """List regular pipeline runs that have an ontology ready to judge."""
    out = []
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not p.is_dir():
                continue
            onto = p / "generated" / "ontology.json"
            if not onto.exists():
                continue
            meta = {}
            mp = p / "metadata.json"
            if mp.exists():
                try:
                    meta = _safe_json_loads(mp.read_text(encoding="utf-8")) or {}
                except Exception:
                    meta = {}
            # Skip promoted OE runs — they're already covered by the OE list.
            if meta.get("oe_session_id"):
                continue
            n_classes = 0
            n_relations = 0
            n_hierarchy = 0
            try:
                onto_data = _safe_json_loads(onto.read_text(encoding="utf-8")) or {}
                n_classes = len(onto_data.get("classes", []))
                n_relations = len(onto_data.get("relations", []))
                n_hierarchy = len(onto_data.get("hierarchy", []))
            except Exception:
                pass
            has_verdicts = (p / "qualitative_eval" / "qualitative_summary.json").exists()
            paper_name = ""
            cfg = meta.get("config") or {}
            paper_name = cfg.get("paper_name") or meta.get("note") or ""
            out.append({
                "run_id": p.name,
                "n_classes": n_classes,
                "n_triples": n_relations + n_hierarchy,
                "has_verdicts": has_verdicts,
                "paper_name": paper_name,
            })
    return jsonify(out)


@app.route("/api/ontology-engineering/qualitative-eval", methods=["POST"])
def api_oe_qualitative_eval():
    """Start an async qualitative judgement pass over an ontology.

    Accepts either an OE session id or a regular run id as ``session_id``.
    """
    data = request.get_json(force=True) or {}
    session_id = (data.get("session_id") or "").strip()
    provider = (data.get("provider") or "openai").strip()
    model = (data.get("model") or "").strip() or None

    try:
        recon_path, session_dir, _kind = _resolve_qeval_target(session_id)
    except ValueError:
        return jsonify({"error": "Invalid session/run ID"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    try:
        ontology = _safe_json_loads(recon_path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"Could not read ontology: {e}"}), 500

    from src.evaluation.qualitative_judge import collect_triples

    triples = collect_triples(ontology)
    if not triples:
        return jsonify({"error": "No relations or hierarchy edges to judge"}), 400

    task_id = f"oe-qeval-{uuid4().hex[:8]}"
    with run_status_lock:
        run_statuses[task_id] = {
            "status": "running",
            "current": 0,
            "total": len(triples),
            "message": f"Starting qualitative eval on {len(triples)} triples...",
            "task_id": task_id,
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "error": None,
        }

    def job():
        from src.evaluation.qualitative_judge import (
            run_qualitative_eval,
            verdicts_to_csv,
        )

        def progress(current, total, message):
            with run_status_lock:
                if task_id in run_statuses:
                    run_statuses[task_id].update(current=current, total=total, message=message)

        try:
            verdicts, summary, prompts_log = run_qualitative_eval(
                ontology=ontology,
                provider=provider,
                model=model,
                progress_callback=progress,
            )
            # Save artefacts
            (session_dir / "qualitative_verdicts.jsonl").write_text(
                "\n".join(json.dumps(v, ensure_ascii=False) for v in verdicts),
                encoding="utf-8",
            )
            (session_dir / "qualitative_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8",
            )
            (session_dir / "qualitative_prompts.jsonl").write_text(
                "\n".join(json.dumps(p, ensure_ascii=False) for p in prompts_log),
                encoding="utf-8",
            )
            (session_dir / "qualitative_verdicts.csv").write_text(
                verdicts_to_csv(verdicts), encoding="utf-8",
            )
            with run_status_lock:
                if task_id in run_statuses:
                    run_statuses[task_id].update(
                        status="completed",
                        message=f"Qualitative eval complete ({summary['n_triples']} triples).",
                        current=run_statuses[task_id]["total"],
                        summary=summary,
                    )
        except Exception as e:
            traceback.print_exc()
            with run_status_lock:
                if task_id in run_statuses:
                    run_statuses[task_id].update(
                        status="failed", error=str(e), message=f"Failed: {e}",
                    )

    threading.Thread(target=job, daemon=True).start()

    return jsonify({
        "task_id": task_id,
        "session_id": session_id,
        "n_triples": len(triples),
        "status_url": f"/api/ontology-engineering/qualitative-eval/{task_id}/status",
    })


@app.route("/api/ontology-engineering/qualitative-eval/<task_id>/status")
def api_oe_qualitative_eval_status(task_id):
    """Poll qualitative-eval task status."""
    with run_status_lock:
        data = dict(run_statuses.get(task_id, {"status": "unknown"}))
    return jsonify(data)


@app.route("/api/ontology-engineering/<session_id>/qualitative-eval")
def api_oe_qualitative_eval_results(session_id):
    """Return saved qualitative eval results for an OE session or regular run."""
    try:
        _recon, session_dir, _kind = _resolve_qeval_target(session_id)
    except ValueError:
        return jsonify({"error": "Invalid session/run ID"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    summary_path = session_dir / "qualitative_summary.json"
    verdicts_path = session_dir / "qualitative_verdicts.jsonl"
    if not summary_path.exists() or not verdicts_path.exists():
        return jsonify({"error": "No qualitative evaluation on file for this session"}), 404

    try:
        summary = _safe_json_loads(summary_path.read_text(encoding="utf-8"))
        verdicts = [
            _safe_json_loads(line)
            for line in verdicts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as e:
        return jsonify({"error": f"Could not read eval artefacts: {e}"}), 500

    return jsonify({"summary": summary, "verdicts": verdicts})


@app.route("/api/ontology-engineering/<session_id>/qualitative-eval.csv")
def api_oe_qualitative_eval_csv(session_id):
    """Download qualitative verdicts as CSV (OE session or regular run)."""
    try:
        _recon, session_dir, _kind = _resolve_qeval_target(session_id)
    except ValueError:
        return "Invalid session/run ID", 400
    except FileNotFoundError as e:
        return str(e), 404

    csv_path = session_dir / "qualitative_verdicts.csv"
    if not csv_path.exists():
        return "No qualitative evaluation on file for this session", 404

    return app.response_class(
        csv_path.read_text(encoding="utf-8"),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={session_id}_qualitative.csv"},
    )


@app.route("/api/ontology-engineering/<session_id>/apply-judge-edits", methods=["POST"])
def api_oe_apply_judge_edits(session_id):
    """Apply user-approved judge verdicts to the reconstructed ontology.

    Request body:
        {"approved_indexes": [int, ...], "allow_new_classes": bool (optional)}

    Writes the cleaned ontology to a NEW OE session folder named
    '<session_id>-cleaned' (or '<session_id>-cleaned-N' on collision) so
    the cleaned run appears independently in the OE session list and can
    be loaded / downloaded / re-evaluated like any other session. Essential
    files from the source session are copied over, and a
    'cleaning_metadata.json' file records the source + approval decisions.

    Only verdicts whose triple_index is in approved_indexes are applied;
    every other triple passes through unchanged (even rejects).
    """
    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return jsonify({"error": "Invalid session ID"}), 400

    session_dir = _OE_DIR / session_id
    recon_path = session_dir / "reconstructed_ontology.json"
    verdicts_path = session_dir / "qualitative_verdicts.jsonl"
    if not recon_path.exists():
        return jsonify({"error": "No reconstructed ontology for this session"}), 404
    if not verdicts_path.exists():
        return jsonify({"error": "No qualitative evaluation on file for this session"}), 404

    body = request.get_json(silent=True) or {}
    approved = body.get("approved_indexes") or []
    if not isinstance(approved, list):
        return jsonify({"error": "approved_indexes must be a list"}), 400
    try:
        approved_set = {int(i) for i in approved}
    except (TypeError, ValueError):
        return jsonify({"error": "approved_indexes must contain integers"}), 400
    allow_new_classes = bool(body.get("allow_new_classes", False))

    try:
        ontology = _safe_json_loads(recon_path.read_text(encoding="utf-8"))
        verdicts = [
            _safe_json_loads(line)
            for line in verdicts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as e:
        return jsonify({"error": f"Could not read session artefacts: {e}"}), 500

    if not approved_set:
        return jsonify({"error": "No triples were approved — check at least one row before applying."}), 400

    # Keep only approved verdicts; unapproved triples fall through unchanged.
    filtered_verdicts = [v for v in verdicts if int(v.get("triple_index", -1)) in approved_set]

    try:
        from src.evaluation.apply_judge_edits import apply_edits
        cleaned, stats = apply_edits(ontology, filtered_verdicts, allow_new_classes=allow_new_classes)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"apply_edits failed: {e}"}), 500

    # ── Create a NEW session folder for the cleaned run ──
    new_session_id = f"{session_id}-cleaned"
    new_dir = _OE_DIR / new_session_id
    if new_dir.exists():
        n = 2
        while (_OE_DIR / f"{session_id}-cleaned-{n}").exists():
            n += 1
        new_session_id = f"{session_id}-cleaned-{n}"
        new_dir = _OE_DIR / new_session_id

    try:
        new_dir.mkdir(parents=True, exist_ok=False)
        # The cleaned ontology IS the new session's reconstructed ontology.
        (new_dir / "reconstructed_ontology.json").write_text(
            json.dumps(cleaned, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Copy session-essential files so the new folder behaves like a
        # fully-featured OE session (graph, cluster view, prior eval).
        for fname in (
            "reconstruction_log.json",
            "merged_ontology.json",
            "cluster_data.json",
            "merge_stats.json",
            "qualitative_verdicts.jsonl",
            "qualitative_summary.json",
            "qualitative_prompts.jsonl",
            "qualitative_verdicts.csv",
        ):
            src = session_dir / fname
            if src.exists():
                shutil.copy2(src, new_dir / fname)
        # Cleaning metadata: source session + what the user approved.
        cleaning_meta = {
            "source_session_id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "approved_count": len(filtered_verdicts),
            "total_verdicts": len(verdicts),
            "approved_indexes": sorted(approved_set),
            "allow_new_classes": allow_new_classes,
            "stats": stats,
        }
        (new_dir / "cleaning_metadata.json").write_text(
            json.dumps(cleaning_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not write cleaned session: {e}"}), 500

    stats["approved_count"] = len(filtered_verdicts)
    stats["total_verdicts"] = len(verdicts)
    stats["new_session_id"] = new_session_id
    stats["download_url"] = f"/api/ontology-engineering/{new_session_id}/reconstructed-ontology"
    return jsonify(stats)


@app.route("/api/ontology-engineering/<session_id>/reconstructed-ontology")
def api_oe_download_reconstructed(session_id):
    """Download an OE session's reconstructed ontology as a JSON file."""
    if not session_id or not re.match(r"^oe-[\w-]+$", session_id):
        return "Invalid session ID", 400

    path = _OE_DIR / session_id / "reconstructed_ontology.json"
    if not path.exists():
        return "No reconstructed ontology on file for this session", 404

    return app.response_class(
        path.read_text(encoding="utf-8"),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename={session_id}.json"},
    )


# ── Generic run-metrics endpoints (used by Ontology Engineering "Analyze") ──

def _run_label_from_metadata(run_id: str, run_root: Path) -> str:
    """Build a short human-readable label for a run from its metadata."""
    label = run_id
    meta_path = run_root / "metadata.json"
    if not meta_path.exists():
        return label
    try:
        meta = _safe_json_loads(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return label
    strat = (
        meta.get("strategy")
        or ((meta.get("config") or {}).get("strategies", [{}])[0] or {}).get("prompt_strategy")
        or ""
    )
    short_id = run_id[:16]
    if strat:
        return f"{short_id}… — {_STRATEGY_LABELS.get(strat, strat)}"
    return short_id


@app.route("/api/runs-metrics", methods=["POST"])
def api_runs_metrics():
    """Return metrics for each of the given run_ids (one row per run, in order).

    Request body: ``{"run_ids": ["...","..."]}``. Response shape mirrors
    ``api_configs_last_runs`` so the same dashboard renderer can reuse it.
    Runs without saved metrics produce a row with ``run_id`` populated but all
    metric fields set to ``None``.
    """
    data = request.get_json(silent=True) or {}
    run_ids = data.get("run_ids") or []

    def _f1(p, r):
        if p is None or r is None:
            return None
        try:
            p = float(p); r = float(r)
        except (TypeError, ValueError):
            return None
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    rows: List[Dict[str, Any]] = []
    for rid in run_ids:
        run_root = _safe_run_path(rid)
        if not run_root or not run_root.exists():
            rows.append({
                "run_id": rid,
                "label": rid,
                "metrics": None,
                "f1": None,
                "precision": None,
                "recall": None,
                "coverage": None,
                "hallucinations": None,
                "hierarchy_edges": None,
                "rel_f1": None,
                "hier_f1": None,
                "overall_f1": None,
            })
            continue
        metrics = load_metrics(run_root)
        m = metrics or {}
        err = m.get("errors") or {}
        struct = m.get("structural") or {}
        he = struct.get("hierarchy_edges")

        # Structural F1s — rel_f1 and hier_f1 come straight from metrics.json's
        # relations/hierarchy sub-objects. overall_f1 is the mean of the three
        # available F1s (class, hierarchy, relation), skipping missing ones.
        class_f1 = _f1_from_metrics(metrics) if metrics else None
        rel = m.get("relations") or {}
        hier = m.get("hierarchy") or {}
        rel_f1 = _f1(rel.get("precision"), rel.get("recall"))
        hier_f1 = _f1(hier.get("precision"), hier.get("recall"))
        f1_parts = [x for x in (class_f1, rel_f1, hier_f1) if x is not None]
        overall_f1 = (sum(f1_parts) / len(f1_parts)) if f1_parts else None

        rows.append({
            "run_id": rid,
            "label": _run_label_from_metadata(rid, run_root),
            "metrics": m if metrics else None,
            "f1": class_f1,
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "coverage": m.get("coverage"),
            "hallucinations": err.get("hallucinations"),
            "hierarchy_edges": int(he) if he is not None else None,
            "rel_f1": rel_f1,
            "hier_f1": hier_f1,
            "overall_f1": overall_f1,
        })
    return jsonify({"rows": rows})


@app.route("/api/run/<run_id>/summary", methods=["GET"])
def api_run_summary(run_id: str):
    """Return the generated ``summary.txt`` for a single run."""
    run_root = _safe_run_path(run_id)
    if not run_root or not run_root.exists():
        return jsonify({"found": False, "message": "Run not found."}), 404
    summary_path = run_root / "generated" / "summary.txt"
    if not summary_path.exists():
        return jsonify({
            "found": False,
            "run_id": run_id,
            "message": "No summary saved for this run.",
        })
    try:
        summary = summary_path.read_text(encoding="utf-8")
    except Exception:
        return jsonify({
            "found": False,
            "run_id": run_id,
            "message": "Could not read summary.",
        })
    return jsonify({"found": True, "run_id": run_id, "summary": summary})


if __name__ == "__main__":
    # Keep debug features, but disable the auto-reloader.
    # The reloader restarts the process on file changes, which would kill background runs
    # and clear in-memory progress state (making the progress page appear "stuck").
    app.run(debug=True, use_reloader=False)
