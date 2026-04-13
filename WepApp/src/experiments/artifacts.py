"""Helpers for writing run artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..ontology.export import write_ontology_json
from ..prompting.strategy_order import STRATEGY_LABELS

# Academically clearer display names for improvement toggles (config keys unchanged for compatibility)
IMPROVEMENT_LABELS = {
    "text_grounded_completion": "Text-grounded completion (built-in)",
    "schema_guided_completion": "Schema-guided completion",
    "builtin_cleanup": "Built-in cleanup (always active)",
    "orphan_rescue": "Orphan Rescue (connect isolated classes)",
    "post_orphan_scope_cleanup": "Post-orphan scope cleanup",
    "llm_refinement": "LLM CoT Refinement (semantic quality review)",
    "symbolic_reasoner": "Rule-based Reasoning Layer (optional)",
}

# Display label for LLM provider in run summary (provider + model name)
LLM_PROVIDER_LABELS = {
    "openai": "openai GPT-4o-mini",
    "openai-4o": "openai GPT-4o",
    "anthropic": "anthropic Claude Haiku 4.5",
    "google": "google Gemini 2.5 Flash",
    "groq": "groq Llama 3.1 8B (free)",
    "huggingface": "Hugging Face Mistral 7B (free)",
    "deepseek": "DeepSeek (deepseek-chat)",
}


def _deepseek_model_label(model: str | None) -> str | None:
    """Return a short label for DeepSeek model for run summary."""
    if not model or not str(model).strip():
        return None
    m = str(model).strip().lower()
    if m == "deepseek-reasoner":
        return "DeepSeek Reasoner (R1)"
    if m == "deepseek-chat":
        return "DeepSeek (chat)"
    return str(model).strip()


def _format_llm_label(provider: str, model: str | None) -> str:
    provider_key = (provider or "").strip()
    if not provider_key:
        return "—"
    if provider_key == "local":
        if model and str(model).strip():
            p = str(model).strip().rstrip("\\/")
            tail = p.split("\\")[-1].split("/")[-1]
            if tail.lower().endswith(".gguf"):
                tail = tail[:-5]
            return f"local {tail}" if tail else "local"
        return "local"
    if provider_key == "deepseek" and model:
        deepseek_label = _deepseek_model_label(model)
        if deepseek_label:
            return deepseek_label
    if model and str(model).strip():
        return f"{provider_key} {str(model).strip()}"
    return LLM_PROVIDER_LABELS.get(provider_key, provider_key)


def write_generated(run_dir: Path, ontology, summary_name: str = "summary.txt") -> None:
    """Write ontology.json only. summary.txt is written later by write_run_summary (metadata + metrics + ontology)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_ontology_json(run_dir / "ontology.json", ontology)


def _pipeline_mode_label(config: Dict[str, Any]) -> str:
    """Derive the pipeline mode name from config flags.
    Must stay in sync with app.py _pipeline_mode_label() and JS pipelineModeLabel().
    Logic: Controlled = any non-default cleanup/preprocessing toggle was set;
           Schema-Completed = sgc or sym (gold-schema injection);
           Strict = text-only (text-grounded completion is always built-in).
    """
    _cleanup_keys = ("cleanup_dedupe", "cleanup_scope_pruning", "cleanup_evidence_pruning",
                     "cleanup_structural", "cleanup_axioms")
    has_custom_cleanup = any(k in config and config[k] is not True for k in _cleanup_keys)
    has_tgl_off = config.get("text_grounded_completion") is False
    if has_custom_cleanup or has_tgl_off:
        return "Controlled"
    sgc = bool(config.get("schema_guided_completion"))
    sym_r = bool(config.get("symbolic_reasoner"))
    if sgc or sym_r:
        return "Schema-Completed"
    return "Strict"


def _fmt_pct(val: Any) -> str:
    """Format a float 0-1 as percentage string."""
    try:
        return f"{float(val):.2%}"
    except (TypeError, ValueError):
        return "—"


def write_run_summary(
    run_dir: Path,
    run_id: str,
    ontology,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    strategy,
    summary_name: str = "summary.txt",
    improvement_counts: Dict[str, Any] | None = None,
    docs: list | None = None,
    timestamp_utc: str | None = None,
) -> None:
    """
    Write a single, readable run summary: metadata, method, pipeline mode, evaluation
    settings, input papers, improvements, metrics (final + extraction baseline + by_stage),
    improvement counts, and brief ontology listing.
    """
    from datetime import datetime as _dt

    strategy_id = getattr(strategy, "prompt_strategy", None) or config.get("strategies", [{}])[0].get("prompt_strategy", "baseline")
    method_label = STRATEGY_LABELS.get(strategy_id, strategy_id)

    llm_provider_key = config.get("llm_provider", "")
    llm_model = config.get("llm_model")
    llm_provider = _format_llm_label(llm_provider_key, llm_model)
    improvements_llm = None
    if config.get("schema_guided_completion") or config.get("llm_refinement"):
        imp_provider = config.get("improvements_llm_provider") or llm_provider_key
        imp_model = config.get("improvements_llm_model")
        if (imp_model and str(imp_model).strip()) or (imp_provider and imp_provider != llm_provider_key):
            improvements_llm = _format_llm_label(str(imp_provider), str(imp_model) if imp_model else None)

    improvements = []
    for key, label in IMPROVEMENT_LABELS.items():
        if config.get(key):
            improvements.append(label)
    improvements_str = ", ".join(improvements) if improvements else "None"

    pipeline_mode = _pipeline_mode_label(config)

    preprocess_flags = []
    if config.get("scope_filter"):
        preprocess_flags.append("Scope filter (clinical-only chunks)")
    if config.get("medical_ner_anchor"):
        preprocess_flags.append("Medical NER anchor (experimental)")
    preprocess_str = ", ".join(preprocess_flags) if preprocess_flags else "None"

    eval_flags = []
    if config.get("eval_restrict_to_gold"):
        eval_flags.append("Gold-vocabulary-only evaluation")
    eval_str = ", ".join(eval_flags) if eval_flags else "None"

    # Input papers list
    paper_lines: list[str] = []
    if docs:
        for d in docs:
            source = d.get("source", "")
            stem = d.get("id", "")
            from pathlib import Path as _Path
            friendly = _Path(source).name if source else (stem or "unknown")
            paper_lines.append(f"  • {friendly}")
    elif config.get("corpus_path"):
        paper_lines.append(f"  • {config['corpus_path']} (directory)")

    ts = timestamp_utc or _dt.utcnow().isoformat()

    err = metrics.get("errors") or {}
    if isinstance(err, dict):
        err_str = (
            f"hallucinations={err.get('hallucinations', 0)}, "
            f"schema_violations={err.get('schema_violations', 0)}, "
            f"omissions={err.get('omissions', 0)}"
        )
    else:
        err_str = str(err)

    struct = metrics.get("structural") or {}
    if isinstance(struct, dict):
        struct_str = (
            f"hierarchy_edges={struct.get('hierarchy_edges', 0)}, "
            f"hierarchy_coverage={struct.get('hierarchy_coverage', 0):.2f}, "
            f"relation_domain_range_rate={struct.get('relation_domain_range_rate', 0):.2f}"
        )
    else:
        struct_str = str(struct)

    n_class = len(ontology.classes)
    n_rel = len(ontology.relations)
    n_hier = len(ontology.hierarchy)

    lines = [
        "=" * 60,
        "RUN SUMMARY  —  TripleGen Ontology Engineering Framework",
        "=" * 60,
        "",
        "--- Metadata ---",
        f"Run ID:          {run_id}",
        f"Timestamp (UTC): {ts}",
        f"Method:          {method_label}",
        f"Pipeline mode:   {pipeline_mode}",
        f"LLM provider:    {llm_provider}",
        *([f"Reasoning LLM:   {improvements_llm}"] if improvements_llm else []),
        f"Improvements:    {improvements_str}",
        f"Preprocessing:   {preprocess_str}",
        f"Eval settings:   {eval_str}",
        "",
    ]

    if paper_lines:
        lines.append("--- Input paper(s) ---")
        lines.extend(paper_lines)
        lines.append("")

    if improvement_counts:
        lines.append("--- Improvement counts (for ablation / fairness) ---")
        for key, label in IMPROVEMENT_LABELS.items():
            counts = improvement_counts.get(key)
            if counts and isinstance(counts, dict):
                parts = [f"{k}={v}" for k, v in counts.items() if v is not None]
                if parts:
                    lines.append(f"  {label}: " + ", ".join(parts))
        lines.extend(["", ""])
        if improvement_counts.get("text_grounded_completion"):
            tgc = improvement_counts["text_grounded_completion"]
            added_c = tgc.get("classes_added", 0) or 0
            added_r = tgc.get("relations_added", 0) or 0
            added_h = tgc.get("hierarchy_added", 0) or 0
            if added_c or added_r or added_h:
                lines.append(
                    f"  Text-grounded completion (built-in): "
                    f"+{added_c} classes, +{added_r} relations, +{added_h} hierarchy edges "
                    "recovered from source text in second-pass extraction."
                )
                lines.append("")
        if improvement_counts.get("orphan_rescue"):
            orph = improvement_counts["orphan_rescue"]
            o_found = orph.get("orphans_found", 0) or 0
            o_rels = orph.get("relations_added", 0) or 0
            o_hier = orph.get("hierarchy_added", 0) or 0
            if o_found:
                lines.append(
                    f"  Orphan Rescue: {o_found} orphaned classes found → "
                    f"+{o_rels} relations, +{o_hier} hierarchy edges inferred from source text."
                )
                lines.append("")
        if improvement_counts.get("llm_refinement"):
            ref = improvement_counts["llm_refinement"]
            r_cls = ref.get("classes_removed", 0) or 0
            r_rel = ref.get("relations_removed", 0) or 0
            r_hier = ref.get("hierarchy_removed", 0) or 0
            r_corr = ref.get("hierarchy_corrected", 0) or 0
            if r_cls or r_rel or r_hier or r_corr:
                lines.append(
                    f"  LLM CoT Refinement: -{r_cls} classes, -{r_rel} relations, "
                    f"-{r_hier} hierarchy removed; {r_corr} hierarchy edges corrected."
                )
                lines.append("")

    # Final metrics
    lines.extend([
        "--- Metrics summary (final ontology) ---",
        f"  Coverage:        {_fmt_pct(metrics.get('coverage', 0))}",
        f"  Precision:       {_fmt_pct(metrics.get('precision', 0))}",
        f"  Recall:          {_fmt_pct(metrics.get('recall', 0))}",
        f"  Errors:          {err_str}",
        f"  Structural:      {struct_str}",
    ])
    # Clinical-only sub-metrics if present
    clin = metrics.get("clinical_only") or {}
    if clin:
        lines.append(
            f"  Clinical-only:   coverage={_fmt_pct(clin.get('coverage'))}, "
            f"precision={_fmt_pct(clin.get('precision'))}, "
            f"recall={_fmt_pct(clin.get('recall'))}"
        )
    # Relation metrics
    rel_m = metrics.get("relations") or {}
    if rel_m:
        lines.append(
            f"  Relations:       precision={_fmt_pct(rel_m.get('precision'))}, "
            f"recall={_fmt_pct(rel_m.get('recall'))}, "
            f"n_generated={rel_m.get('n_generated', 0)}, n_gold={rel_m.get('n_gold', 0)}"
        )
    # Hierarchy metrics
    hier_m = metrics.get("hierarchy") or {}
    if hier_m:
        lines.append(
            f"  Hierarchy:       precision={_fmt_pct(hier_m.get('precision'))}, "
            f"recall={_fmt_pct(hier_m.get('recall'))}, "
            f"matched={hier_m.get('matched', 0)}, n_generated={hier_m.get('n_generated', 0)}, n_gold={hier_m.get('n_gold', 0)}"
        )
    lines.append("")

    # Extraction-only baseline comparison
    ext = metrics.get("extraction_only") or {}
    if ext:
        ext_struct = ext.get("structural") or {}
        lines.extend([
            "--- Metrics at extraction only (pre-improvement baseline) ---",
            f"  Coverage:        {_fmt_pct(ext.get('coverage', 0))}",
            f"  Precision:       {_fmt_pct(ext.get('precision', 0))}",
            f"  Recall:          {_fmt_pct(ext.get('recall', 0))}",
        ])
        if ext_struct:
            lines.append(
                f"  Structural:      hierarchy_edges={ext_struct.get('hierarchy_edges', 0)}, "
                f"hierarchy_coverage={ext_struct.get('hierarchy_coverage', 0):.2f}, "
                f"relation_domain_range_rate={ext_struct.get('relation_domain_range_rate', 0):.2f}"
            )
        ext_rel = ext.get("relations") or {}
        if ext_rel:
            lines.append(
                f"  Relations:       precision={_fmt_pct(ext_rel.get('precision'))}, "
                f"recall={_fmt_pct(ext_rel.get('recall'))}, "
                f"n_generated={ext_rel.get('n_generated', 0)}"
            )
        lines.append("")

    # Per-stage ablation table
    by_stage = metrics.get("by_stage") or {}
    if by_stage:
        _STAGE_LABELS = {
            "extraction": "Extraction       ",
            "after_text_grounded": "+ Text-grounded  ",
            "after_sgc": "+ Schema-Guided  ",
            "after_cleanup": "+ Cleanup        ",
            "after_orphan_rescue": "+ Orphan Rescue  ",
            "after_llm_refinement": "+ LLM Refinement ",
            "after_rule_based": "+ Rule-based     ",
            "after_gold_filter": "= Gold-filtered  ",
        }
        lines.extend([
            "--- Per-stage ablation (pipeline progression) ---",
            f"  {'Stage':<22} {'Classes':>8} {'Rels':>6} {'Hier':>6} {'Coverage':>10} {'Precision':>10} {'Recall':>10}",
            "  " + "-" * 74,
        ])
        for stage_key, stage_label in _STAGE_LABELS.items():
            s = by_stage.get(stage_key)
            if s is None:
                continue
            cov = _fmt_pct(s.get("coverage")) if "coverage" in s else "—"
            prec = _fmt_pct(s.get("precision")) if "precision" in s else "—"
            rec = _fmt_pct(s.get("recall")) if "recall" in s else "—"
            lines.append(
                f"  {stage_label:<22} {s.get('n_classes', 0):>8} {s.get('n_relations', 0):>6} "
                f"{s.get('n_hierarchy', 0):>6} {cov:>10} {prec:>10} {rec:>10}"
            )
        lines.append("")

    lines.extend([
        "--- Concept counts (final ontology) ---",
        f"  Classes:         {n_class}",
        f"  Relations:       {n_rel}",
        f"  Hierarchy edges: {n_hier}",
        "",
        "--- Classes ---",
    ])
    for c in ontology.classes:
        defn = getattr(c, "definition", None) and str(c.definition).strip()
        lines.append(f"  • {c.label}" + (f" — {defn}" if defn else ""))
    lines.extend([
        "",
        "--- Relations ---",
    ])
    for r in ontology.relations:
        d = getattr(r, "domain", "") or ""
        rg = getattr(r, "range", "") or ""
        rdefn = getattr(r, "definition", None) and str(r.definition).strip()
        lines.append(f"  • {r.label}({d} → {rg})" + (f" — {rdefn}" if rdefn else ""))
    lines.extend([
        "",
        "--- Hierarchy (subclass → superclass) ---",
    ])
    for e in ontology.hierarchy:
        sub = e.get("subClass", "")
        sup = e.get("superClass", "")
        lines.append(f"  • {sub} ⊑ {sup}")
    lines.append("")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / summary_name).write_text("\n".join(lines), encoding="utf-8")
