"""Merge multiple ontology JSON files with canonical deduplication.

Used by the Ontology Engineering page to combine extractions from
multiple pipeline runs into a single deduplicated ontology.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .build import _title_case_label
from .canonical import canonical_key, resolve_to_canonical_label
from .export import ontology_from_dict, ontology_to_dict
from .model import ClassEntity, Ontology, RelationEntity


def merge_ontologies(ontology_dicts: List[Dict[str, Any]],
                     metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge multiple ontology JSON dicts into one, deduplicating classes/relations/hierarchy.

    Args:
        ontology_dicts: List of raw dicts (the content of each ``generated/ontology.json``).
        metadata: Optional metadata dict for the merged ontology.

    Returns:
        Merged ontology as a JSON-serialisable dict (same schema as ``ontology.json``).
    """
    merged = Ontology(metadata=dict(metadata or {}))

    # canonical_key -> (display_label, index in merged.classes)
    seen_classes: Dict[str, tuple] = {}
    # (rel_key, dom_key, rng_key) -> (display_label, index in merged.relations)
    seen_relations: Dict[tuple, tuple] = {}
    # (sub_key, sup_key) set
    seen_hierarchy: set = set()

    for onto_dict in ontology_dicts:
        onto = ontology_from_dict(onto_dict)

        # ── Merge classes ─────────────────────────────────────────
        for cls in onto.classes:
            label = (cls.label or "").strip()
            if not label:
                continue
            canonical_label, was_mapped = resolve_to_canonical_label(label)
            if not was_mapped:
                canonical_label = _title_case_label(canonical_label)
            key = canonical_key(canonical_label)
            if not key:
                continue

            aliases_to_add = list(cls.aliases or [])
            if was_mapped and label != canonical_label and label not in aliases_to_add:
                aliases_to_add.append(label)

            if key in seen_classes:
                stored_label, idx = seen_classes[key]
                existing = merged.classes[idx]
                existing.provenance = list(dict.fromkeys(
                    existing.provenance + (cls.provenance or [])
                ))
                existing.synonyms = list(dict.fromkeys(
                    existing.synonyms + (cls.synonyms or [])
                ))
                for a in aliases_to_add:
                    if a and a not in existing.aliases:
                        existing.aliases.append(a)
                if label != stored_label and label not in existing.aliases:
                    existing.aliases.append(label)
                # Keep longest evidence
                ev = (cls.evidence or "").strip()
                if ev and (not existing.evidence or len(ev) > len(existing.evidence or "")):
                    existing.evidence = ev
                # Keep longest definition
                defn = (cls.definition or "").strip()
                if defn and (not existing.definition or len(defn) > len(existing.definition or "")):
                    existing.definition = defn
                continue

            seen_classes[key] = (canonical_label, len(merged.classes))
            entity_aliases = list(aliases_to_add)
            if label != canonical_label and label not in entity_aliases:
                entity_aliases.append(label)
            merged.add_class(ClassEntity(
                label=canonical_label,
                definition=(cls.definition or "").strip() or None,
                synonyms=list(cls.synonyms or []),
                provenance=list(cls.provenance or []),
                stratum=cls.stratum,
                original_label=cls.original_label,
                evidence=(cls.evidence or "").strip() or None,
                aliases=entity_aliases,
            ))

        # ── Merge relations ───────────────────────────────────────
        for rel in onto.relations:
            label = (rel.label or "").strip()
            dom = (rel.domain or "").strip()
            rng = (rel.range or "").strip()
            if not label or not dom or not rng:
                continue

            rkey = canonical_key(label)
            if not rkey:
                continue

            dom_resolved = resolve_to_canonical_label(dom)[0]
            rng_resolved = resolve_to_canonical_label(rng)[0]
            dom_canon = seen_classes.get(canonical_key(dom_resolved), (dom_resolved,))[0]
            rng_canon = seen_classes.get(canonical_key(rng_resolved), (rng_resolved,))[0]

            rel_tuple = (rkey, canonical_key(dom_canon), canonical_key(rng_canon))
            if rel_tuple in seen_relations:
                stored_label, idx = seen_relations[rel_tuple]
                existing = merged.relations[idx]
                existing.provenance = list(dict.fromkeys(
                    existing.provenance + (rel.provenance or [])
                ))
                if label != stored_label and label not in existing.aliases:
                    existing.aliases.append(label)
                ev = (rel.evidence or "").strip()
                if ev and (not existing.evidence or len(ev) > len(existing.evidence or "")):
                    existing.evidence = ev
                defn = (rel.definition or "").strip()
                if defn and (not existing.definition or len(defn) > len(existing.definition or "")):
                    existing.definition = defn
                continue

            seen_relations[rel_tuple] = (label, len(merged.relations))
            merged.add_relation(RelationEntity(
                label=label,
                domain=dom_canon,
                range=rng_canon,
                definition=(rel.definition or "").strip() or None,
                provenance=list(rel.provenance or []),
                stratum=rel.stratum,
                evidence=(rel.evidence or "").strip() or None,
                aliases=list(rel.aliases or []),
            ))

        # ── Merge hierarchy ───────────────────────────────────────
        for edge in onto.hierarchy:
            sub = (edge.get("subClass") or "").strip()
            sup = (edge.get("superClass") or "").strip()
            if not sub or not sup:
                continue

            sub_resolved = resolve_to_canonical_label(sub)[0]
            sup_resolved = resolve_to_canonical_label(sup)[0]
            skey = canonical_key(sub_resolved)
            sukey = canonical_key(sup_resolved)
            if not skey or not sukey:
                continue
            if skey == sukey:
                continue  # skip self-referential

            sub_canon = seen_classes.get(skey, (sub_resolved,))[0]
            sup_canon = seen_classes.get(sukey, (sup_resolved,))[0]

            hkey = (canonical_key(sub_canon), canonical_key(sup_canon))
            if hkey in seen_hierarchy:
                continue
            seen_hierarchy.add(hkey)

            merged.hierarchy.append({
                "subClass": sub_canon,
                "superClass": sup_canon,
                "evidence": (edge.get("evidence") or "").strip() or None,
                "provenance": list(edge.get("provenance") or []),
                "stratum": edge.get("stratum"),
            })

    return ontology_to_dict(merged)
