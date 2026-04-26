"""Apply qualitative-judge verdicts to produce a cleaned ontology.

Given an ontology dict and a list of verdicts from
`qualitative_judge.run_qualitative_eval()`, this module:

  - drops every triple the judge labelled `reject`
  - applies every `revise` suggested_triple the judge returned (skipping
    revisions that would introduce classes not already in the ontology
    unless `allow_new_classes=True`)
  - keeps every `accept_direct` / `accept_indirect` triple unchanged

The verdict's `triple_index` is the position of the triple in the list
produced by `collect_triples()` (relations first in insertion order, then
hierarchy). We reconstruct that mapping here so edits line up even if the
verdict file is re-ordered by worker scheduling.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple


def _canonical_classes(ontology: Dict[str, Any]) -> set[str]:
    return {
        (c.get("label") or "").strip().lower()
        for c in (ontology.get("classes") or [])
        if (c.get("label") or "").strip()
    }


def apply_edits(
    ontology: Dict[str, Any],
    verdicts: List[Dict[str, Any]],
    *,
    allow_new_classes: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (cleaned_ontology, stats).

    cleaned_ontology: same shape as input, with triples edited/removed.
    stats: counts and per-category breakdown of what was done.
    """
    relations = list(ontology.get("relations") or [])
    hierarchy = list(ontology.get("hierarchy") or [])
    n_rel = len(relations)

    # triple_index → (kind, index_within_kind)
    def _idx_to_location(i: int) -> Tuple[str, int]:
        if i < n_rel:
            return ("relation", i)
        return ("hierarchy", i - n_rel)

    known_classes = _canonical_classes(ontology)

    # Defaults: keep all, then apply edits.
    kept_relations: List[Optional[Dict[str, Any]]] = [copy.deepcopy(r) for r in relations]
    kept_hierarchy: List[Optional[Dict[str, Any]]] = [copy.deepcopy(h) for h in hierarchy]

    stats = {
        "input_relations": n_rel,
        "input_hierarchy": len(hierarchy),
        "dropped_reject": 0,
        "revise_applied": 0,
        "revise_skipped_no_suggestion": 0,
        "revise_skipped_new_class": 0,
        "revise_changed_kind": 0,
        "kept_accept": 0,
        "unknown_verdict": 0,
        "by_subtype_applied": {},
        "by_subtype_skipped": {},
    }

    for v in verdicts:
        idx = int(v.get("triple_index", -1))
        if idx < 0 or idx >= n_rel + len(hierarchy):
            continue
        kind, j = _idx_to_location(idx)
        verdict = v.get("verdict")

        if verdict == "reject":
            if kind == "relation":
                kept_relations[j] = None
            else:
                kept_hierarchy[j] = None
            stats["dropped_reject"] += 1
            continue

        if verdict in ("accept_direct", "accept_indirect"):
            stats["kept_accept"] += 1
            continue

        if verdict != "revise":
            stats["unknown_verdict"] += 1
            continue

        # ── revise ──
        sug = v.get("suggested_triple") or None
        subtype = v.get("revise_subtype") or "4.10"
        if not sug or not isinstance(sug, dict):
            stats["revise_skipped_no_suggestion"] += 1
            stats["by_subtype_skipped"][subtype] = stats["by_subtype_skipped"].get(subtype, 0) + 1
            continue

        new_subj = (sug.get("subject") or "").strip()
        new_rel = (sug.get("relation") or "").strip()
        new_obj = (sug.get("object") or "").strip()
        if not (new_subj and new_rel and new_obj):
            stats["revise_skipped_no_suggestion"] += 1
            stats["by_subtype_skipped"][subtype] = stats["by_subtype_skipped"].get(subtype, 0) + 1
            continue

        # Only apply if both endpoints are known classes (unless user opts in to new classes).
        if not allow_new_classes:
            if new_subj.lower() not in known_classes or new_obj.lower() not in known_classes:
                stats["revise_skipped_new_class"] += 1
                stats["by_subtype_skipped"][subtype] = stats["by_subtype_skipped"].get(subtype, 0) + 1
                continue

        src = kept_relations[j] if kind == "relation" else kept_hierarchy[j]
        if src is None:
            # Already dropped somehow; skip.
            stats["revise_skipped_no_suggestion"] += 1
            continue

        became_hierarchy = new_rel.strip().lower() in ("subclassof", "subclass_of", "is_a", "isa")
        was_hierarchy = (kind == "hierarchy")

        if became_hierarchy == was_hierarchy:
            # In-place edit: preserve provenance/evidence, swap endpoints + relation.
            if kind == "relation":
                src["domain"] = new_subj
                src["range"] = new_obj
                src["label"] = new_rel
                kept_relations[j] = src
            else:
                src["subClass"] = new_subj
                src["superClass"] = new_obj
                kept_hierarchy[j] = src
        else:
            # Cross-kind move: drop from old list, append to new list with preserved provenance.
            ev = src.get("evidence", "")
            prov = list(src.get("provenance") or [])
            note = f"qualitative_judge:revise:{subtype}"
            if note not in prov:
                prov.append(note)
            if kind == "relation":
                kept_relations[j] = None  # drop original relation
                hierarchy.append({
                    "subClass": new_subj,
                    "superClass": new_obj,
                    "evidence": ev,
                    "provenance": prov,
                })
                kept_hierarchy.append(hierarchy[-1])
            else:
                kept_hierarchy[j] = None  # drop original hierarchy
                relations.append({
                    "domain": new_subj,
                    "range": new_obj,
                    "label": new_rel,
                    "evidence": ev,
                    "provenance": prov,
                })
                kept_relations.append(relations[-1])
            stats["revise_changed_kind"] += 1

        stats["revise_applied"] += 1
        stats["by_subtype_applied"][subtype] = stats["by_subtype_applied"].get(subtype, 0) + 1

    cleaned = copy.deepcopy(ontology)
    cleaned["relations"] = [r for r in kept_relations if r is not None]
    cleaned["hierarchy"] = [h for h in kept_hierarchy if h is not None]

    md = cleaned.setdefault("metadata", {})
    md_cleaning = md.setdefault("qualitative_cleaning", {})
    md_cleaning.update({
        "applied": True,
        "input_relations": stats["input_relations"],
        "input_hierarchy": stats["input_hierarchy"],
        "output_relations": len(cleaned["relations"]),
        "output_hierarchy": len(cleaned["hierarchy"]),
        "dropped_reject": stats["dropped_reject"],
        "revise_applied": stats["revise_applied"],
        "revise_skipped_no_suggestion": stats["revise_skipped_no_suggestion"],
        "revise_skipped_new_class": stats["revise_skipped_new_class"],
        "allow_new_classes": allow_new_classes,
    })

    stats["output_relations"] = len(cleaned["relations"])
    stats["output_hierarchy"] = len(cleaned["hierarchy"])
    return cleaned, stats


def fix_class_definitions(
    ontology: Dict[str, Any],
    fixes: Dict[str, str],
    *,
    only_if_broken: bool = True,
    broken_marker: Optional[Callable[[str], bool]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Overwrite class definitions for a small set of labels.

    When `only_if_broken=True`, only overwrite if `broken_marker(current)`
    returns True. Default marker treats empty/very-short definitions as
    broken. Returns (new_ontology, list_of_classes_fixed).
    """
    def _default_broken(defn: str) -> bool:
        return len((defn or "").strip()) < 40
    check = broken_marker or _default_broken

    out = copy.deepcopy(ontology)
    changed: List[str] = []
    for c in out.get("classes") or []:
        lab = (c.get("label") or "").strip()
        if lab in fixes:
            cur = c.get("definition") or ""
            if not only_if_broken or check(cur) or lab in fixes:
                # Always overwrite when user explicitly supplied a fix for this label.
                c["definition"] = fixes[lab]
                changed.append(lab)
    return out, changed
