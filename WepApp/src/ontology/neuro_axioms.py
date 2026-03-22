"""
Neuro-ICU / BrainIT physiological and semantic type constraints (axiom-driven verification).

Used by the Rule-based Reasoning Layer to enforce:
- Forbidden hierarchy (is_a) pairs: e.g. Drug ⊑ Symptom is physiologically illegal.
- Allowed relation type pairs: e.g. (Therapy, Condition) may only use treats, targets, etc.;
  if the LLM proposes is_a between such types, the relation or hierarchy edge is removed and flagged.

Semantic types are inferred from the gold schema (each class gets the type of its top-level
ancestor). Forbidden/allowed rules are defined below and can be extended via config later.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# Normalise for matching
def _n(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def infer_semantic_types_from_gold(gold_schema: dict) -> Dict[str, str]:
    """
    Build class_label (normalised) -> semantic_type from gold schema.
    Semantic type = top-level ancestor in the hierarchy (root); roots are their own type.
    """
    classes = gold_schema.get("classes", [])
    hierarchy = gold_schema.get("hierarchy", [])
    labels = {c.get("label", "").strip() for c in classes if c.get("label")}
    norm_to_label: Dict[str, str] = {_n(c.get("label", "")): c.get("label", "").strip() for c in classes if c.get("label")}

    # subClass -> superClass edges
    child_to_parent: Dict[str, Set[str]] = {}
    for e in hierarchy:
        sub = (e.get("subClass") or "").strip()
        sup = (e.get("superClass") or "").strip()
        if not sub or not sup:
            continue
        sn, pn = _n(sub), _n(sup)
        if sn not in child_to_parent:
            child_to_parent[sn] = set()
        child_to_parent[sn].add(pn)

    def root(norm: str) -> str:
        """Return normalised label of the root (top-level ancestor) for this class.

        When a class has multiple parents, follow all paths and return the
        shallowest root (fewest hops). Cycles are detected and broken.
        """
        frontier = [(norm, 0)]
        visited: Set[str] = set()
        best_root = norm
        best_depth = float("inf")
        while frontier:
            current, depth = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            parents = child_to_parent.get(current)
            if not parents:
                if depth < best_depth:
                    best_depth = depth
                    best_root = current
                continue
            for p in parents:
                if p not in visited:
                    frontier.append((p, depth + 1))
        return best_root

    result: Dict[str, str] = {}
    for c in classes:
        label = (c.get("label") or "").strip()
        if not label:
            continue
        norm = _n(label)
        r = root(norm)
        result[norm] = norm_to_label.get(r, r)
    return result


# Forbidden is_a (subClass type, superClass type): physiologically illegal.
# E.g. (Therapy, Condition) = "Therapy ⊑ Condition" is forbidden.
# Type names are normalised root labels from infer_semantic_types_from_gold.
FORBIDDEN_HIERARCHY_TYPE_PAIRS: Set[Tuple[str, str]] = {
    # Original v1.0 constraints
    ("therapy", "condition"),
    ("monitoringdata", "therapy"),
    ("monitoringdata", "condition"),
    ("patient", "condition"),
    ("patient", "therapy"),
    ("patient", "monitoringdata"),
    ("condition", "therapy"),
    ("condition", "patient"),
    # v2.0 cross-root constraints — Clinical Assessment / Outcome
    ("clinicalassessment", "therapy"),
    ("clinicalassessment", "monitoringdata"),
    ("outcome", "therapy"),
    ("outcome", "condition"),
    ("outcome", "monitoringdata"),
    # v2.0 cross-root constraints — Nursing Intervention
    ("nursingintervention", "condition"),
    ("nursingintervention", "monitoringdata"),
    # v2.0 cross-root constraints — Laboratory Values
    ("laboratoryvalues", "therapy"),
    ("laboratoryvalues", "condition"),
    # v2.0 Moss 2013 framework — prevent nonsensical cross-root hierarchies
    ("session", "condition"),
    ("session", "therapy"),
    ("session", "monitoringdata"),
    ("observation", "condition"),
    ("observation", "therapy"),
    ("sensor", "condition"),
    ("sensor", "therapy"),
    ("parameter", "condition"),
    ("parameter", "therapy"),
    ("dataqualityassessment", "condition"),
    ("dataqualityassessment", "therapy"),
    ("dataqualityassessment", "monitoringdata"),
}


def _norm_type(t: str) -> str:
    return _n(t)


# Allowed relation labels for (domain_type, range_type). Strict benchmark: canonical forms only.
# If a type pair is NOT listed here, ALL relations are allowed (permissive default).
# Only add restrictions for type pairs where the valid set is well-defined.
ALLOWED_RELATIONS_BY_TYPE_PAIR: Dict[Tuple[str, str], Set[str]] = {
    # v1.0 constraints
    (_norm_type("Therapy"), _norm_type("Condition")): {
        _n("targetsCondition"),
        _n("targets condition"),
        _n("treats"),
    },
    (_norm_type("Patient"), _norm_type("MonitoringData")): {
        _n("hasMonitoringData"),
        _n("has monitoring data"),
    },
    (_norm_type("Patient"), _norm_type("Therapy")): {
        _n("receivesTherapy"),
        _n("receives therapy"),
    },
    (_norm_type("MonitoringData"), _norm_type("Condition")): {
        _n("indicatesCondition"),
        _n("monitoring indicates condition"),
        _n("monitoringIndicatesCondition"),
    },
    # v2.0 constraints
    (_norm_type("Patient"), _norm_type("Clinical Assessment")): {
        _n("has clinical assessment"),
        _n("hasclinicalassessment"),
    },
    (_norm_type("Patient"), _norm_type("Outcome")): {
        _n("has outcome"),
        _n("hasoutcome"),
    },
    (_norm_type("Patient"), _norm_type("Session")): {
        _n("has session"),
        _n("hassession"),
    },
    (_norm_type("Observation"), _norm_type("Parameter")): {
        _n("measures parameter"),
        _n("measuresparameter"),
    },
    (_norm_type("Observation"), _norm_type("Sensor")): {
        _n("produced by sensor"),
        _n("producedbysensor"),
    },
    (_norm_type("Observation"), _norm_type("Data Quality Assessment")): {
        _n("has quality assessment"),
        _n("hasqualityassessment"),
    },
    (_norm_type("Data Quality Assessment"), _norm_type("Condition")): {
        _n("associated with condition"),
        _n("associatedwithcondition"),
    },
    (_norm_type("Data Quality Assessment"), _norm_type("Therapy")): {
        _n("affected by treatment"),
        _n("affectedbytreatment"),
    },
}


def check_hierarchy_axiom(
    sub_type: str,
    super_type: str,
) -> bool:
    """True if (sub_type, super_type) is allowed; False if forbidden (physiologically illegal)."""
    return (_norm_type(sub_type), _norm_type(super_type)) not in FORBIDDEN_HIERARCHY_TYPE_PAIRS


def check_relation_axiom(
    domain_type: str,
    range_type: str,
    relation_label: str,
) -> bool:
    """
    True if relation with given domain/range types and label is allowed.
    If (domain_type, range_type) has an allowed set, relation_label must be in it; else allowed.
    """
    key = (_norm_type(domain_type), _norm_type(range_type))
    allowed = ALLOWED_RELATIONS_BY_TYPE_PAIR.get(key)
    if allowed is None:
        return True
    return _n(relation_label) in allowed
