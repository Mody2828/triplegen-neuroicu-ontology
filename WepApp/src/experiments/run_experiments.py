"""Run a single experiment config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import load_config, load_strategies
from .run_registry import create_run_dirs
from .metadata import build_metadata, write_metadata
from .artifacts import write_generated, write_run_summary
from .errors import write_failure_summary
from ..common.logging import log_event
from ..corpus.ingest import load_corpus, write_manifest
from ..corpus.chunk import chunk_documents
from ..corpus.scope_filter import filter_chunks_to_clinical
from ..prompting.llm_client import LLMClient
from ..prompting.run import run_strategy
from ..prompting.schema import filter_parsed_to_vocabulary, MIN_EVIDENCE_LENGTH_STRICT
from ..ontology.build import build_ontology
from ..ontology.export import ontology_from_dict, ontology_to_dict, write_ontology_json
from ..ontology.validate import validate_ontology
from ..evaluation.gold_standard import load_gold_standard
from ..evaluation.gold_standard_surrogate import build_surrogate
from ..evaluation.align import align_entities
from ..evaluation.metrics import (
    compute_coverage,
    compute_precision_recall,
    compute_structural_metrics,
    compute_metrics_from_classes,
    compute_relation_metrics,
    compute_hierarchy_metrics,
)
from ..evaluation.errors import error_taxonomy


def _flatten_metrics_for_table(metrics: Dict, prefix: str = "") -> List[Dict]:
    """Flatten nested metrics for CSV table: extraction_only.coverage, by_stratum.core.n_generated, etc."""
    rows = []
    for k, v in metrics.items():
        name = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and k != "errors":
            rows.extend(_flatten_metrics_for_table(v, prefix=f"{name}."))
        elif not isinstance(v, list):
            val = json.dumps(v) if isinstance(v, dict) else v
            rows.append({"metric": name, "value": val})
    return rows


def _capture_stage_metrics(
    ontology, gold_classes: List,
    gold_relations: Optional[List] = None,
    gold_hierarchy: Optional[List] = None,
) -> Dict:
    """
    Capture a compact metrics snapshot for the current ontology state.
    Used to build the by_stage ablation table: raw → SGC → Cleanup → LLM Reasoning → Rule-based → Gold-filtered.
    """
    classes_dict = [c.__dict__ for c in ontology.classes]
    stage: Dict = {
        "n_classes": len(ontology.classes),
        "n_relations": len(ontology.relations),
        "n_hierarchy": len(ontology.hierarchy),
    }
    if gold_classes:
        m = compute_metrics_from_classes(classes_dict, gold_classes, ontology)
        stage["coverage"] = round(m.get("coverage", 0.0), 4)
        stage["precision"] = round(m.get("precision", 0.0), 4)
        stage["recall"] = round(m.get("recall", 0.0), 4)
    if gold_relations:
        rel = compute_relation_metrics([r.__dict__ for r in ontology.relations], gold_relations)
        stage["relation_recall"] = round(rel.get("recall", 0.0), 4)
        stage["relation_precision"] = round(rel.get("precision", 0.0), 4)
    if gold_hierarchy:
        h = compute_hierarchy_metrics(ontology.hierarchy, gold_hierarchy)
        stage["hierarchy_recall"] = round(h.get("recall", 0.0), 4)
        stage["hierarchy_precision"] = round(h.get("precision", 0.0), 4)
    return stage


def _compute_by_stratum_metrics(ontology, gold_classes: List) -> Dict:
    """Compute metrics per stratum (core, governance, provenance). Core stratum aligned vs gold; others report n_generated."""
    from ..evaluation.align import align_entities

    by_stratum = {}
    for stratum_name in ("core", "governance", "provenance"):
        stratum_classes = [c.__dict__ for c in ontology.classes if getattr(c, "stratum", None) == stratum_name]
        if not stratum_classes:
            continue
        entry = {"n_generated": len(stratum_classes)}
        if stratum_name == "core" and gold_classes:
            align = align_entities(generated=stratum_classes, gold=gold_classes)
            matched = len(align["matched_exact"]) + len(align["matched_semantic"])
            gold_matched = len(align.get("gold_labels_matched", set()))
            entry["precision"] = matched / len(stratum_classes) if stratum_classes else 0.0
            entry["recall"] = min(1.0, gold_matched / len(gold_classes)) if gold_classes else 0.0
            entry["coverage"] = min(1.0, gold_matched / len(gold_classes)) if gold_classes else 0.0
        by_stratum[stratum_name] = entry
    return by_stratum
from ..evaluation.report import write_metrics, write_table


def run_one(
    config: Dict,
    strategy,
    run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    base_dir = config.get("runs_dir", "runs")
    run_paths = create_run_dirs(base_dir=base_dir, run_id=run_id)
    log_path = run_paths.root / "run.log"
    errors = []
    try:
        # Enforce that benchmark runs (those with a gold standard) always set scope_filter
        # explicitly in their config. This prevents silent non-comparable results between
        # UI runs (scope_filter=True by default) and CLI/config runs (previously False by default).
        # Second option chosen: mandate the key in config rather than silently defaulting.
        scope_filter_value = config.get("scope_filter")
        has_gold = bool(
            getattr(strategy, "gold_standard_path", None)
            or config.get("gold_standard_path")
        )
        if has_gold and scope_filter_value is None:
            raise ValueError(
                "CONFIG ERROR: 'scope_filter' is not set in your experiment config, "
                "but a gold standard is loaded. Benchmark runs against the BrainIT gold "
                "standard MUST set scope_filter explicitly. "
                "Add '\"scope_filter\": true' to your config JSON for clinical-only "
                "evaluation (recommended), or '\"scope_filter\": false' if you "
                "intentionally want unfiltered text (must document why in the dissertation)."
            )
        docs = load_corpus(
            strategy.corpus_path,
            scope_filter=bool(scope_filter_value),
            keep_raw=config.get("keep_raw_text", False),
        )
        write_manifest(run_paths.root / "corpus_manifest.json", docs)

        # Phase 28.1: Warn if multi-paper + baseline strategy
        strategy_name = getattr(strategy, "name", "") or config.get("strategy", "")
        if len(docs) > 1 and strategy_name in ("baseline", "zero_shot"):
            import warnings
            warnings.warn(
                f"Multi-paper run ({len(docs)} papers) with '{strategy_name}' strategy — "
                "this may produce noisy results. Consider using 'phased_3step' for multi-paper runs.",
                UserWarning,
                stacklevel=2,
            )

        # Phase 28.2: Increase chunk overlap for multi-paper inputs
        chunk_kwargs = {}
        if len(docs) > 1:
            chunk_kwargs["overlap_tokens"] = 250  # default is 150

        chunks = chunk_documents(docs, **chunk_kwargs)
        chunk_clinical_filter = config.get("chunk_clinical_filter", True)
        if config.get("scope_filter") and chunk_clinical_filter:
            n_before = len(chunks)
            from ..corpus.scope_filter import DEFAULT_MIN_CLINICAL_SCORE_STRICT
            min_clinical_score = config.get("min_clinical_score")
            if min_clinical_score is None:
                min_clinical_score = DEFAULT_MIN_CLINICAL_SCORE_STRICT
            chunks = filter_chunks_to_clinical(
                chunks,
                clinical_only=True,
                min_clinical_score=min_clinical_score,
                embedding_scope_fallback=config.get("embedding_scope_fallback", False),
            )
            if n_before != len(chunks):
                log_lines_chunk = [log_event("scope_filter_chunk_suppression", {"chunks_before": n_before, "chunks_after": len(chunks), "dropped": n_before - len(chunks)})]
            else:
                log_lines_chunk = []
        else:
            log_lines_chunk = []
        if progress_callback is not None:
            progress_callback(0, len(chunks), f"Loaded {len(docs)} document(s), {len(chunks)} chunk(s). Starting extraction…")
        log_lines = [log_event("corpus_loaded", {"documents": len(docs), "chunks": len(chunks)})]
        if config.get("scope_filter"):
            log_lines.append(log_event("scope_filter", {"enabled": True}))
        log_lines.extend(log_lines_chunk)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        llm_provider = getattr(strategy, "llm_provider", None)
        if not llm_provider:
            raise ValueError("llm_provider must be specified in strategy config. Supported: 'openai', 'anthropic', 'google', 'groq', 'huggingface'")
        llm_model = getattr(strategy, "llm_model", None)
        if llm_model is not None and not isinstance(llm_model, str):
            llm_model = str(llm_model).strip() or None
        elif isinstance(llm_model, str):
            llm_model = llm_model.strip() or None
        prompt_strategy = getattr(strategy, "prompt_strategy", "baseline")
        log_path.write_text(
            log_path.read_text(encoding="utf-8")
            + log_event("llm_config", {"provider": llm_provider, "model": llm_model, "prompt_strategy": prompt_strategy})
            + "\n",
            encoding="utf-8",
        )
        llm = LLMClient(provider=llm_provider, model=llm_model)
        improvements_llm_provider = (config.get("improvements_llm_provider") or "").strip() or llm_provider
        improvements_llm_model = (config.get("improvements_llm_model") or "").strip() or None
        improvements_llm = LLMClient(provider=improvements_llm_provider, model=improvements_llm_model)

        _VALID_GOLD_MODES = {"public", "restricted", "isolated"}
        gold_mode = (
            getattr(strategy, "gold_standard_mode", None)
            or config.get("gold_standard_mode")
            or os.getenv("GOLD_STANDARD_MODE", "public")
        )
        if gold_mode not in _VALID_GOLD_MODES:
            log_event("invalid_gold_mode", {"gold_mode": gold_mode, "fallback": "public"})
            gold_mode = "public"
        preloaded_gold = None
        allowed_classes = None
        allowed_relations = None
        relation_domains = None
        
        # Load gold standard for restricted mode OR for phased_3step (needs gold for phase 1 filter)
        gold_path = strategy.gold_standard_path or config.get("gold_standard_path")
        if not gold_path:
            # Try default gold path
            try:
                from web.app import get_default_gold_path
                gold_path = get_default_gold_path()
            except (ImportError, AttributeError):
                gold_path = None
        
        needs_gold_for_improvements = (
            config.get("schema_guided_completion")
            or config.get("symbolic_reasoner")
        )
        needs_vocab_from_gold = (
            config.get("prompt_vocab_guardrails")
            or config.get("filter_to_gold_vocabulary")
            or config.get("eval_restrict_to_gold")
        )
        if gold_path and Path(gold_path).exists():
            if gold_mode == "restricted" or needs_gold_for_improvements or needs_vocab_from_gold:
                preloaded_gold = load_gold_standard(gold_path)
                if needs_vocab_from_gold:
                    allowed_classes = [c.get("label") for c in preloaded_gold.get("classes", []) if c.get("label")]
                    allowed_relations = [r.get("label") for r in preloaded_gold.get("relations", []) if r.get("label")]
                    relation_domains = {
                        r.get("label"): (r.get("domain"), r.get("range"))
                        for r in preloaded_gold.get("relations", [])
                        if r.get("label")
                    }

        parsed_chunks = run_strategy(
            chunks,
            strategy.prompt_strategy,
            llm,
            prompt_save_dir=run_paths.prompts,
            allowed_classes=allowed_classes,
            allowed_relations=allowed_relations,
            relation_domains=relation_domains,
            gold_ontology=preloaded_gold,
            inject_vocab_guardrails=bool(config.get("prompt_vocab_guardrails", False)),
            filter_to_gold_vocabulary=bool(config.get("filter_to_gold_vocabulary", False)),
            inject_medical_ner_anchor=bool(config.get("medical_ner_anchor", False)),
            inject_candidate_terms=bool(config.get("candidate_terms", False)),
            strict_relations=bool(config.get("strict_relations", False)),
            require_label_in_evidence=bool(config.get("require_label_in_evidence", True)),
            clinical_only_routing=bool(
                config.get("clinical_only_routing")
                if "clinical_only_routing" in config
                else config.get("scope_filter", False)
            ),
            progress_callback=progress_callback,
        )

        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks), "Extraction complete. Merging chunk outputs into one ontology…")
        ontology = build_ontology(parsed_chunks, metadata={"strategy": strategy.name})
        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks), "Merge complete. Applying optional improvements…")

        # Extraction-only metrics (before text-grounded / schema_guided / cleanup) for ablation.
        # Relations are also captured here — BEFORE any post-processing modifies ontology.relations —
        # so that extraction_only relation recall reflects the raw extractor output, not the pipeline.
        extraction_only_metrics = None
        extraction_only_relations: List[Dict] = []
        if preloaded_gold:
            extraction_only_metrics = compute_metrics_from_classes(
                [c.__dict__ for c in ontology.classes],
                preloaded_gold.get("classes", []),
                ontology,
            )
            extraction_only_relations = [r.__dict__ for r in ontology.relations]

        # Improvement counts for ablation / fairness (report added vs extracted, inferred vs extracted)
        improvement_counts = {}

        # Per-stage metrics: track ontology state after each improvement so the dissertation
        # ablation table shows the full Raw → Text-grounded → SGC → Cleanup → Rule-based → Gold-filtered progression.
        _gold_classes_for_stage = preloaded_gold.get("classes", []) if preloaded_gold else []
        _gold_relations_for_stage = preloaded_gold.get("relations") if preloaded_gold else None
        _gold_hierarchy_for_stage = preloaded_gold.get("hierarchy") if preloaded_gold else None
        by_stage: Dict = {}
        # Stage 0: raw extraction (same snapshot as extraction_only, captured for symmetry)
        if preloaded_gold:
            by_stage["extraction"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        # Text-grounded completion: second-pass extraction giving LLM the full document + current
        # ontology. Controlled by config flag (default True for backward compatibility).
        corpus_text = "\n\n".join(c.get("text", "") for c in chunks)
        run_tgl = config.get("text_grounded_completion", True)
        if run_tgl and corpus_text.strip():
            from ..prompting.ontology_completion import run_text_grounded_completion
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Running text-grounded completion…")
            tgc_result = run_text_grounded_completion(
                ontology, llm, corpus_text,
                progress_callback=progress_callback,
                run_dir=str(run_paths.root),
            )
            improvement_counts["text_grounded_completion"] = tgc_result
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Text-grounded completion done.")
            by_stage["after_text_grounded"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        # Optional: schema-guided completion on the full merged ontology (any strategy)
        if config.get("schema_guided_completion") and preloaded_gold and improvements_llm:
            from ..prompting.ontology_completion import run_schema_guided_completion
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Running Schema-guided completion…")
            improvement_counts["schema_guided_completion"] = run_schema_guided_completion(
                ontology, preloaded_gold, improvements_llm, corpus_text=corpus_text,
                progress_callback=progress_callback,
                run_dir=str(run_paths.root),
            )
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Schema-guided completion done.")
            by_stage["after_sgc"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        # Save pre-cleanup snapshot for debugging (shows what extraction + TGC produced)
        write_ontology_json(run_paths.generated / "ontology_pre_cleanup.json", ontology)

        # Built-in cleanup: dedupe, domain/range, axiom, scope/evidence pruning.
        # Cleanup is guided by the domain scope file (resources/domain_scope.json),
        # which is independent of the gold standard (used for evaluation only).
        # In Strict mode, gold-label canonicalization and axiom constraints are skipped.
        _is_strict_mode = (
            not config.get("prompt_vocab_guardrails", False)
            and not config.get("eval_restrict_to_gold", False)
        )
        _should_cleanup = True
        if _should_cleanup:
            from ..ontology.reasoner import apply_builtin_cleanup, apply_symbolic_reasoner_enrichment
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Running built-in cleanup…")
            cleanup_config = {
                "cleanup_dedupe": config.get("cleanup_dedupe", True),
                "cleanup_scope_pruning": config.get("cleanup_scope_pruning", True),
                "cleanup_evidence_pruning": config.get("cleanup_evidence_pruning", True),
                "cleanup_structural": config.get("cleanup_structural", True),
                "cleanup_axioms": config.get("cleanup_axioms", True),
            }
            cleanup_result = apply_builtin_cleanup(
                ontology,
                preloaded_gold or {},
                cleanup_config=cleanup_config,
                gold_free=_is_strict_mode,
            )
            improvement_counts["builtin_cleanup"] = {
                k: v for k, v in cleanup_result.items()
                if k not in ("axiom_violations", "cleanup_audit_log")
            }
            improvement_counts["builtin_cleanup"].setdefault("hierarchy_added", 0)
            improvement_counts["builtin_cleanup"].setdefault("orphans_removed", 0)
            violations_list = cleanup_result.get("axiom_violations") or []
            if violations_list:
                improvement_counts["builtin_cleanup"]["axiom_violations_count"] = len(violations_list)
                (run_paths.evaluation / "axiom_violations.json").write_text(
                    json.dumps(violations_list, indent=2), encoding="utf-8"
                )
            _cleanup_audit = cleanup_result.get("cleanup_audit_log") or []
            if _cleanup_audit:
                (run_paths.root / "cleanup_removed.json").write_text(
                    json.dumps(_cleanup_audit, indent=2, default=str), encoding="utf-8"
                )
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Built-in cleanup done.")
            by_stage["after_cleanup"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        # Orphan Rescue Pass: after cleanup, find classes with no relations/hierarchy
        # and ask the LLM to infer connections using the source text.
        # Independently toggleable via config["orphan_rescue"] (defaults to text_grounded_completion).
        _orphan_rescue_enabled = config.get("orphan_rescue", config.get("text_grounded_completion", True))
        if corpus_text.strip() and _orphan_rescue_enabled:
            from ..prompting.ontology_completion import run_orphan_rescue
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Running orphan rescue…")
            orphan_result = run_orphan_rescue(
                ontology, llm, corpus_text,
                progress_callback=progress_callback,
                run_dir=str(run_paths.root),
            )
            improvement_counts["orphan_rescue"] = orphan_result
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks),
                    f"Orphan rescue: {orphan_result.get('orphans_found', 0)} orphans → "
                    f"+{orphan_result.get('relations_added', 0)} relations, "
                    f"+{orphan_result.get('hierarchy_added', 0)} hierarchy.")

            # Post-orphan-rescue scope cleanup: orphan rescue may introduce
            # out-of-scope classes as relation endpoints.
            from ..ontology.reasoner import apply_post_pass_scope_cleanup
            post_orphan_cleanup = apply_post_pass_scope_cleanup(ontology)
            improvement_counts["post_orphan_scope_cleanup"] = post_orphan_cleanup

            by_stage["after_orphan_rescue"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        # LLM Chain-of-Thought Refinement: semantic review of the full ontology.
        # The LLM reasons about each class, relation, and hierarchy edge to
        # identify non-clinical noise, semantic errors, and hierarchy inversions.
        # Independently toggleable via config["llm_refinement"] (defaults to text_grounded_completion).
        _llm_refinement_enabled = config.get("llm_refinement", config.get("text_grounded_completion", True))
        if corpus_text.strip() and _llm_refinement_enabled:
            from ..prompting.ontology_completion import run_ontology_refinement
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Running LLM refinement…")
            refinement_result = run_ontology_refinement(
                ontology, llm, corpus_text,
                progress_callback=progress_callback,
                run_dir=str(run_paths.root),
            )
            improvement_counts["llm_refinement"] = refinement_result
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks),
                    f"LLM refinement: -{refinement_result.get('classes_removed', 0)} classes, "
                    f"-{refinement_result.get('relations_removed', 0)} relations, "
                    f"-{refinement_result.get('hierarchy_removed', 0)} hierarchy, "
                    f"{refinement_result.get('hierarchy_corrected', 0)} corrections.")
            by_stage["after_llm_refinement"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        # Optional Rule-based Reasoning Layer (UI toggle): deterministic schema completion + orphan pruning.
        if preloaded_gold and config.get("symbolic_reasoner"):
            from ..ontology.reasoner import apply_symbolic_reasoner_enrichment
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Running Rule-based Reasoning Layer…")
            enrichment_result = apply_symbolic_reasoner_enrichment(ontology, preloaded_gold)
            improvement_counts["symbolic_reasoner"] = enrichment_result
            if progress_callback is not None:
                progress_callback(len(chunks), len(chunks), "Rule-based Reasoning Layer done.")
            by_stage["after_rule_based"] = _capture_stage_metrics(
                ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
            )

        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks), "Validating ontology and saving artefacts…")
        warnings = validate_ontology(ontology)
        if warnings:
            (run_paths.root / "warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
        # Always persist the raw ontology (unrestricted).
        write_ontology_json(run_paths.generated / "ontology_raw.json", ontology)

        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks), "Evaluating against gold standard…")
        if gold_mode == "isolated":
            gold_corpus_path = (
                getattr(strategy, "gold_corpus_path", None) or config.get("gold_corpus_path")
            )
            if not gold_corpus_path:
                raise RuntimeError("Gold standard isolation mode requires gold_corpus_path.")
            gold = _build_gold_from_corpus(gold_corpus_path, llm, prompt_save_dir=run_paths.prompts)
            (run_paths.evaluation / "gold_generated.json").write_text(
                json.dumps(gold, indent=2), encoding="utf-8"
            )
        elif gold_mode == "restricted" and strategy.gold_standard_path:
            gold = preloaded_gold or load_gold_standard(strategy.gold_standard_path)
        else:
            gold = build_surrogate()

        eval_ontology = ontology
        if config.get("eval_restrict_to_gold") and preloaded_gold and gold_mode == "restricted":
            filtered = filter_parsed_to_vocabulary(
                ontology_to_dict(ontology),
                allowed_classes or [],
                allowed_relations or [],
                require_evidence=True,
                min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                relation_domains=relation_domains,
                gold_hierarchy=(preloaded_gold.get("hierarchy") or []),
            )
            eval_ontology = ontology_from_dict(filtered)
            eval_ontology.metadata = dict(ontology.metadata)
            # Save restricted view for inspection; ontology.json always shows the full output.
            write_ontology_json(run_paths.generated / "ontology_restricted.json", eval_ontology)
            # Capture final stage after gold-vocab filtering so ablation table matches
            # the "Concept counts (final ontology)" section and relation/hierarchy numbers.
            if by_stage and preloaded_gold:
                by_stage["after_gold_filter"] = _capture_stage_metrics(
                    eval_ontology, _gold_classes_for_stage, _gold_relations_for_stage, _gold_hierarchy_for_stage
                )

        # Always write the full (unfiltered) ontology as ontology.json so the UI shows the
        # true system output. eval_restrict_to_gold only affects which classes count toward
        # precision/recall; it must not silently replace the primary artefact.
        write_generated(run_paths.generated, ontology)

        gold_classes = gold.get("classes", [])
        generated_classes = [c.__dict__ for c in eval_ontology.classes]

        # Full metrics (current behaviour)
        class_alignment = align_entities(
            generated=generated_classes, gold=gold_classes
        )
        matched = class_alignment["matched_exact"] + class_alignment["matched_semantic"]
        unmatched = class_alignment["unmatched"]
        gold_matched = class_alignment.get("gold_labels_matched", set())
        unique_gold_matched = len(gold_matched)
        metrics = {
            "coverage": compute_coverage(unique_gold_matched, gold_classes),
            **compute_precision_recall(len(matched), generated_classes, unique_gold_matched, gold_classes),
            "errors": error_taxonomy(
                unmatched,
                [],
                relations=[r.__dict__ for r in eval_ontology.relations],
            ),
            "structural": compute_structural_metrics(eval_ontology),
        }

        # Extraction-only metrics (before text-grounded / schema_guided / cleanup)
        if extraction_only_metrics is not None:
            metrics["extraction_only"] = {
                "coverage": extraction_only_metrics["coverage"],
                "precision": extraction_only_metrics["precision"],
                "recall": extraction_only_metrics["recall"],
                "structural": extraction_only_metrics["structural"],
            }

        # Clinical-only metrics (exclude governance classes to avoid pollution when gold is clinical)
        clinical_metrics = compute_metrics_from_classes(
            generated_classes, gold_classes, eval_ontology, exclude_governance=True
        )
        metrics["clinical_only"] = {
            "coverage": clinical_metrics["coverage"],
            "precision": clinical_metrics["precision"],
            "recall": clinical_metrics["recall"],
        }

        # Relation recall against gold (label-level, alias-normalized).
        # This fills the evaluation gap: the 4 gold relations were previously never measured.
        # Reported separately from structural metrics; does not affect class precision/recall.
        if gold.get("relations"):
            generated_relations = [r.__dict__ for r in eval_ontology.relations]
            metrics["relations"] = compute_relation_metrics(
                generated_relations,
                gold.get("relations", []),
            )
            # Also add to extraction_only block so ablation is complete.
            # Use the snapshot captured before any post-processing (SGC / LLM reasoning / cleanup)
            # so this truly reflects extractor-only relation recall.
            if extraction_only_metrics is not None and "extraction_only" in metrics:
                metrics["extraction_only"]["relations"] = compute_relation_metrics(
                    extraction_only_relations,
                    gold.get("relations", []),
                )

        # Hierarchy precision/recall against gold
        if gold.get("hierarchy"):
            metrics["hierarchy"] = compute_hierarchy_metrics(
                eval_ontology.hierarchy,
                gold.get("hierarchy", []),
            )

        # Per-stratum metrics (core / governance / provenance)
        by_stratum = _compute_by_stratum_metrics(eval_ontology, gold_classes)
        if by_stratum:
            metrics["by_stratum"] = by_stratum
        # Expansion strategy: report strict vs mapped separately so we don't overclaim (see strategies doc).
        if any(g.get("original_label") for g in generated_classes):
            strict_matched = [c for c in matched if not c.get("original_label")]
            mapped_matched = [c for c in matched if c.get("original_label")]
            metrics["expansion_strict_matched"] = len(strict_matched)
            metrics["expansion_mapped_matched"] = len(mapped_matched)
            metrics["expansion_proposed_new"] = len(unmatched)
        # Per-stage ablation: shows metrics at each pipeline stage for dissertation analysis.
        # Stages present depend on which improvements were enabled in this run.
        # Always has "extraction" and "after_text_grounded"; conditionally: after_sgc, after_cleanup, after_rule_based.
        if by_stage:
            metrics["by_stage"] = by_stage
        write_metrics(run_paths.evaluation / "metrics.json", metrics)
        table_rows = _flatten_metrics_for_table(metrics)
        write_table(run_paths.evaluation / "table.csv", table_rows)
        # Dump hallucinated (unmatched) classes for inspection and synonym tuning
        hallucinated = [{"label": u.get("label"), "definition": u.get("definition")} for u in unmatched]
        (run_paths.evaluation / "hallucinated_classes.json").write_text(
            json.dumps(hallucinated, indent=2), encoding="utf-8"
        )
        if improvement_counts:
            (run_paths.evaluation / "improvement_counts.json").write_text(
                json.dumps(improvement_counts, indent=2), encoding="utf-8"
            )

        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks), "Writing run summary…")
        metadata = build_metadata(config, run_paths.run_id, [d["id"] for d in docs], docs=docs)
        if improvement_counts:
            metadata["improvement_counts"] = improvement_counts
        write_metadata(run_paths.root / "metadata.json", metadata)
        write_run_summary(
            run_paths.generated,
            run_paths.run_id,
            eval_ontology,
            metrics,
            config,
            strategy,
            improvement_counts=improvement_counts or None,
            docs=docs,
            timestamp_utc=metadata.get("timestamp_utc"),
        )
    except Exception as exc:  # pragma: no cover - safety net for demo runs
        errors.append(str(exc))
        raise
    finally:
        if errors:
            write_failure_summary(run_paths.root, errors)
    return run_paths.root


def _build_gold_from_corpus(gold_corpus_path: str, llm: LLMClient, prompt_save_dir: Path) -> Dict:
    gold_docs = load_corpus(gold_corpus_path)
    gold_chunks = chunk_documents(gold_docs)
    # For isolated gold generation, keep guardrails on to reduce drift/hallucination.
    gold_parsed = run_strategy(
        gold_chunks,
        "baseline",
        llm,
        prompt_save_dir=prompt_save_dir,
        inject_vocab_guardrails=True,
        filter_to_gold_vocabulary=True,
    )
    gold_ontology = build_ontology(gold_parsed, metadata={"strategy": "gold_baseline"})
    return ontology_to_dict(gold_ontology)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for strategy in load_strategies(config):
        run_one(config, strategy)


if __name__ == "__main__":
    main()
