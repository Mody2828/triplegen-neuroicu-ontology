"""Build ontology objects from parsed LLM outputs."""

from __future__ import annotations

import re
from typing import Dict, List

from .canonical import canonical_key, resolve_to_canonical_label
from .model import ClassEntity, Ontology, RelationEntity

# Tokens that should stay lowercase in Title Case (unless first word).
_TITLE_CASE_LOWER = {"of", "and", "the", "in", "for", "to", "by", "or", "a", "an", "at", "on", "with"}

# Known all-caps acronyms that should not be title-cased.
_ACRONYM_TOKENS = {
    "ICP", "MAP", "CPP", "GCS", "GOSe", "CVP", "ECG", "EtCO2", "SaO2",
    "SjO2", "TCD", "PbrO2", "PRx", "NIBP", "ARDS", "TBI", "BP", "HR",
    "DBP", "SBP", "TIL", "EUSIG", "EVD", "CSF",
}
# Case-insensitive lookup: upper() -> canonical-cased form. Needed because
# mixed-case acronyms (SaO2, PbrO2, GOSe, EtCO2) can't be matched by a
# simple `w.upper() in _ACRONYM_TOKENS` check.
_ACRONYM_LOOKUP = {t.upper(): t for t in _ACRONYM_TOKENS}


def _title_case_label(label: str) -> str:
    """Apply Title Case to a class label, preserving known acronyms and bracketed abbreviations."""
    if not label or not label.strip():
        return label
    # Don't touch labels that already come from canonical alias map (contain parens like "(MAP)")
    # but do fix fully lowercase labels
    words = label.split()
    result = []
    for i, w in enumerate(words):
        # Preserve bracketed abbreviations like (MAP), (ICP)
        if w.startswith("(") and w.endswith(")"):
            result.append(w.upper() if w[1:-1].isalpha() and len(w) <= 6 else w)
        elif w.upper() in _ACRONYM_LOOKUP:
            result.append(_ACRONYM_LOOKUP[w.upper()])
        elif i == 0:
            result.append(w.capitalize())
        elif w.lower() in _TITLE_CASE_LOWER:
            result.append(w.lower())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def _canonical_key(label: str) -> str:
    """Alias for canonical_key (backward compat)."""
    return canonical_key(label)


def _merge_provenance(chunk_id: str, item_provenance: List) -> List[str]:
    """Merge item-level provenance with chunk id; dedupe order (item first, then chunk)."""
    item = list(item_provenance or [])
    chunk_val = chunk_id or "unknown"
    return list(dict.fromkeys(item + [chunk_val]))


def build_ontology(parsed_chunks: List[Dict], metadata: Dict[str, str]) -> Ontology:
    ontology = Ontology(metadata=metadata)
    # canonical_key -> (canonical label, index in ontology.classes) for alias updates
    seen_class_keys: Dict[str, tuple] = {}
    # (relation_canonical_key, domain_canon, range_canon) -> (canonical label, index) so same predicate with different domain/range are kept
    seen_relation_keys: Dict[tuple, tuple] = {}
    seen_hierarchy = set()
    for chunk in parsed_chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        stratum = chunk.get("stratum") or "core"
        for cls in chunk.get("classes", []):
            label = (cls.get("label") or "").strip()
            evidence = (cls.get("evidence") or "").strip() or None
            if not label:
                continue
            if not evidence:
                continue
            canonical_label, was_mapped = resolve_to_canonical_label(label)
            # Phase 25.4: enforce Title Case on class labels (fixes lowercase labels)
            if not was_mapped:
                canonical_label = _title_case_label(canonical_label)
            key = canonical_key(canonical_label)
            if not key:
                continue
            provenance = _merge_provenance(chunk_id, cls.get("provenance"))
            aliases_to_add = []
            if was_mapped and label != canonical_label:
                aliases_to_add.append(label)
            if key in seen_class_keys:
                stored_label, class_idx = seen_class_keys[key]
                existing = ontology.classes[class_idx]
                existing.provenance = list(dict.fromkeys(existing.provenance + provenance))
                if existing.stratum is None:
                    existing.stratum = stratum
                existing.synonyms = list(dict.fromkeys(existing.synonyms + list(cls.get("synonyms") or [])))
                for a in aliases_to_add:
                    if a not in existing.aliases:
                        existing.aliases.append(a)
                if label != stored_label and label not in existing.aliases:
                    existing.aliases.append(label)
                if evidence and (not existing.evidence or len(evidence) > len(existing.evidence or "")):
                    existing.evidence = evidence
                # Merge definition: keep the longest non-empty definition
                new_defn = (cls.get("definition") or "").strip()
                if new_defn and (not existing.definition or len(new_defn) > len(existing.definition or "")):
                    existing.definition = new_defn
                continue
            seen_class_keys[key] = (canonical_label, len(ontology.classes))
            entity_aliases = list(aliases_to_add)
            if label != canonical_label and label not in entity_aliases:
                entity_aliases.append(label)
            ontology.add_class(
                ClassEntity(
                    label=canonical_label,
                    definition=(cls.get("definition") or "").strip() or None,
                    synonyms=cls.get("synonyms", []),
                    provenance=provenance,
                    stratum=stratum,
                    original_label=cls.get("original_label"),
                    evidence=evidence,
                    aliases=entity_aliases,
                )
            )
        for rel in chunk.get("relations", []):
            label = (rel.get("label") or "").strip()
            evidence = (rel.get("evidence") or "").strip() or None
            dom = (rel.get("domain") or "").strip()
            rng = (rel.get("range") or "").strip()
            if not label or not evidence:
                continue
            if not dom or not rng:
                continue
            rkey = canonical_key(label)
            if not rkey:
                continue
            # Resolve domain/range to canonical class label (from first-seen class label per key)
            dom_resolved = resolve_to_canonical_label(dom)[0] if dom else dom
            rng_resolved = resolve_to_canonical_label(rng)[0] if rng else rng
            dom_canon = seen_class_keys.get(canonical_key(dom_resolved), (dom_resolved,))[0] if dom else dom
            rng_canon = seen_class_keys.get(canonical_key(rng_resolved), (rng_resolved,))[0] if rng else rng
            provenance = _merge_provenance(chunk_id, rel.get("provenance"))
            # Dedupe by (label, domain, range) so includes(A,B) and includes(C,D) are both kept
            rel_key = (rkey, dom_canon, rng_canon)
            if rel_key in seen_relation_keys:
                canonical_rel_label, rel_idx = seen_relation_keys[rel_key]
                existing = ontology.relations[rel_idx]
                existing.provenance = list(dict.fromkeys(existing.provenance + provenance))
                if existing.stratum is None:
                    existing.stratum = stratum
                if label != canonical_rel_label and label not in existing.aliases:
                    existing.aliases.append(label)
                if evidence and (not existing.evidence or len(evidence) > len(existing.evidence or "")):
                    existing.evidence = evidence
                # Merge definition: keep the longest non-empty definition
                new_defn = (rel.get("definition") or "").strip()
                if new_defn and (not existing.definition or len(new_defn) > len(existing.definition or "")):
                    existing.definition = new_defn
                continue
            seen_relation_keys[rel_key] = (label, len(ontology.relations))
            ontology.add_relation(
                RelationEntity(
                    label=label,
                    domain=dom_canon,
                    range=rng_canon,
                    definition=(rel.get("definition") or "").strip() or None,
                    provenance=provenance,
                    stratum=stratum,
                    evidence=evidence,
                    aliases=[],
                )
            )
        for edge in chunk.get("hierarchy", []):
            sub = (edge.get("subClass") or "").strip()
            sup = (edge.get("superClass") or "").strip()
            evidence_h = (edge.get("evidence") or "").strip() or None
            if not sub or not sup:
                continue
            if not evidence_h:
                continue
            sub_resolved = resolve_to_canonical_label(sub)[0] if sub else sub
            sup_resolved = resolve_to_canonical_label(sup)[0] if sup else sup
            skey, sukey = canonical_key(sub_resolved), canonical_key(sup_resolved)
            if not skey or not sukey:
                continue
            # Resolve to canonical class label if we've seen a variant
            sub_canon = seen_class_keys.get(skey, (sub_resolved,))[0]
            sup_canon = seen_class_keys.get(sukey, (sup_resolved,))[0]
            hkey = (sub_canon, sup_canon)
            if hkey in seen_hierarchy:
                continue
            seen_hierarchy.add(hkey)
            provenance = _merge_provenance(chunk_id, edge.get("provenance"))
            ontology.hierarchy.append(
                {
                    "subClass": sub_canon,
                    "superClass": sup_canon,
                    "evidence": evidence_h,
                    "provenance": provenance,
                    "stratum": stratum,
                }
            )
    return ontology
