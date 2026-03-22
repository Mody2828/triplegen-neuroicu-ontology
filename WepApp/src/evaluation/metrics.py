"""Evaluation metrics."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Set

from ..ontology.model import Ontology


def _norm_rel(s: str) -> str:
    """Normalize a relation label: lowercase, strip all non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", (s or "").strip().lower())


def compute_relation_metrics(
    generated_relations: List[Dict],
    gold_relations: List[Dict],
) -> Dict:
    """
    Compute precision and recall for relation extraction against the gold standard.

    Matching strategy (label-only, normalized):
    - Generated relation labels are first mapped via RELATION_ALIASES_CORE to canonical
      gold-schema labels (e.g. "treats" → "targetsCondition", "has_source" → "hasMonitoringData").
    - Both the mapped label and the original label are checked against gold relation labels
      (all normalized: lowercase, non-alphanumeric stripped).
    - A generated relation is counted as matched if its (aliased) label equals a gold label.
    - Precision = matched generated / total generated (0 if no generated).
    - Recall    = unique gold labels covered / total gold (0 if no gold).
    - per_gold_relation: for each gold relation, whether it was found (True/False).
    - matched_generated: list of generated relation labels that matched.
    - unmatched_generated: list of generated relation labels that did not match.

    Domain/range matching is intentionally not enforced here — the structural metric
    relation_domain_range_rate already captures whether surviving relations have valid
    domain/range. Including domain/range in this check would penalise label-level recall
    for a separate structural failure.
    """
    from ..prompting.vocabulary import RELATION_ALIASES_CORE

    # Build a normalized gold label set and a map for per-relation breakdown
    gold_norm_to_label: Dict[str, str] = {}
    for r in gold_relations:
        lab = (r.get("label") or "").strip()
        if lab:
            gold_norm_to_label[_norm_rel(lab)] = lab

    if not gold_norm_to_label:
        return {
            "n_generated": len(generated_relations),
            "n_gold": 0,
            "precision": 0.0,
            "recall": 0.0,
            "per_gold_relation": {},
            "matched_generated": [],
            "unmatched_generated": [],
        }

    # Build alias map (normalized alias → normalized canonical)
    alias_norm: Dict[str, str] = {}
    for alias, canonical in RELATION_ALIASES_CORE.items():
        alias_norm[_norm_rel(alias)] = _norm_rel(canonical)

    gold_covered: Set[str] = set()
    matched_labels: List[str] = []
    unmatched_labels: List[str] = []

    for r in generated_relations:
        raw_label = (r.get("label") or "").strip()
        norm = _norm_rel(raw_label)
        # Try alias mapping first, then direct match
        canonical_norm = alias_norm.get(norm, norm)
        if canonical_norm in gold_norm_to_label:
            gold_covered.add(canonical_norm)
            matched_labels.append(raw_label)
        elif norm in gold_norm_to_label:
            gold_covered.add(norm)
            matched_labels.append(raw_label)
        else:
            unmatched_labels.append(raw_label)

    n_gen = len(generated_relations)
    n_gold = len(gold_norm_to_label)
    precision = len(matched_labels) / n_gen if n_gen else 0.0
    recall = len(gold_covered) / n_gold if n_gold else 0.0

    per_gold = {
        label: (_norm_rel(label) in gold_covered)
        for label in gold_norm_to_label.values()
    }

    return {
        "n_generated": n_gen,
        "n_gold": n_gold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "per_gold_relation": per_gold,
        "matched_generated": matched_labels,
        "unmatched_generated": unmatched_labels,
    }


def _filter_classes_for_clinical(generated: List[Dict], exclude_governance: bool) -> List[Dict]:
    """Exclude governance-vocab classes from generated when computing clinical-only metrics."""
    if not exclude_governance:
        return generated
    from ..prompting.vocabulary import ALLOWED_CLASSES_GOVERNANCE
    gov = ALLOWED_CLASSES_GOVERNANCE
    return [c for c in generated if (c.get("label") or "").strip() not in gov]


def compute_coverage(unique_gold_matched: int, gold: List[Dict]) -> float:
    """Coverage = fraction of gold classes that have at least one match (0 to 1)."""
    if not gold:
        return 0.0
    n = int(unique_gold_matched) if isinstance(unique_gold_matched, (int, float)) else len(unique_gold_matched)
    return min(1.0, n / len(gold))


def compute_precision_recall(
    matched_count: int,
    generated: List[Dict],
    unique_gold_matched: int,
    gold: List[Dict],
) -> Dict[str, float]:
    """Precision = matched generated / total generated. Recall = unique gold covered / total gold (capped at 1)."""
    n_matched = int(matched_count) if isinstance(matched_count, (int, float)) else len(matched_count)
    n_gold_covered = int(unique_gold_matched) if isinstance(unique_gold_matched, (int, float)) else len(unique_gold_matched)
    precision = n_matched / len(generated) if generated else 0.0
    recall = min(1.0, n_gold_covered / len(gold)) if gold else 0.0
    return {"precision": precision, "recall": recall}


def _norm_hierarchy_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (label or "").strip().lower())


def compute_hierarchy_metrics(
    generated_hierarchy: List[Dict],
    gold_hierarchy: List[Dict],
) -> Dict:
    """Compute precision and recall for hierarchy edges (subClass, superClass pairs)."""
    gold_edges = set()
    for e in gold_hierarchy:
        sub = _norm_hierarchy_key(e.get("subClass", ""))
        sup = _norm_hierarchy_key(e.get("superClass", ""))
        if sub and sup:
            gold_edges.add((sub, sup))

    if not gold_edges:
        return {"n_generated": len(generated_hierarchy), "n_gold": 0,
                "precision": 0.0, "recall": 0.0}

    gen_edges = set()
    for e in generated_hierarchy:
        sub = _norm_hierarchy_key(e.get("subClass", ""))
        sup = _norm_hierarchy_key(e.get("superClass", ""))
        if sub and sup:
            gen_edges.add((sub, sup))

    matched = gen_edges & gold_edges
    n_gen = len(gen_edges)
    precision = len(matched) / n_gen if n_gen else 0.0
    recall = len(matched) / len(gold_edges) if gold_edges else 0.0
    return {
        "n_generated": n_gen,
        "n_gold": len(gold_edges),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matched": len(matched),
    }


def compute_structural_metrics(ontology: Ontology) -> Dict[str, float]:
    total_relations = len(ontology.relations)
    relations_with_domains = sum(
        1 for r in ontology.relations if (r.domain or "").strip() and (r.range or "").strip()
    )
    relation_domain_range_rate = (
        relations_with_domains / total_relations if total_relations else 0.0
    )
    hierarchy_edges = len(ontology.hierarchy)
    hierarchy_nodes = set()
    for edge in ontology.hierarchy:
        if edge.get("subClass"):
            hierarchy_nodes.add(edge["subClass"])
        if edge.get("superClass"):
            hierarchy_nodes.add(edge["superClass"])
    hierarchy_coverage = len(hierarchy_nodes) / len(ontology.classes) if ontology.classes else 0.0
    return {
        "relation_domain_range_rate": relation_domain_range_rate,
        "hierarchy_edges": hierarchy_edges,
        "hierarchy_coverage": hierarchy_coverage,
    }


def compute_metrics_from_classes(
    generated_classes: List[Dict],
    gold_classes: List[Dict],
    ontology: Ontology,
    *,
    exclude_governance: bool = False,
    align_fn: Optional[Callable] = None,
    error_taxonomy_fn: Optional[Callable] = None,
) -> Dict:
    """
    Compute coverage, precision, recall, structural, and errors from generated classes vs gold.
    If exclude_governance=True, filters out ALLOWED_CLASSES_GOVERNANCE from generated before alignment
    (fixes governance pollution when gold is clinical-only).
    """
    from .align import align_entities
    from .errors import error_taxonomy

    align_fn = align_fn or align_entities
    error_taxonomy_fn = error_taxonomy_fn or error_taxonomy

    filtered = _filter_classes_for_clinical(generated_classes, exclude_governance)
    class_alignment = align_fn(
        generated=filtered, gold=gold_classes
    )
    matched = class_alignment["matched_exact"] + class_alignment["matched_semantic"]
    unmatched = class_alignment["unmatched"]
    gold_matched = class_alignment.get("gold_labels_matched", set())
    unique_gold_matched = len(gold_matched)

    return {
        "coverage": compute_coverage(unique_gold_matched, gold_classes),
        **compute_precision_recall(len(matched), filtered, unique_gold_matched, gold_classes),
        "structural": compute_structural_metrics(ontology),
        "errors": error_taxonomy_fn(
            unmatched,
            [],
            relations=[r.__dict__ for r in ontology.relations],
        ),
    }
