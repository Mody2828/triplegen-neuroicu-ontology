"""Ontology export utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

from .model import ClassEntity, Ontology, RelationEntity


def ontology_to_dict(ontology: Ontology) -> Dict:
    return {
        "classes": [c.__dict__ for c in ontology.classes],
        "relations": [r.__dict__ for r in ontology.relations],
        "hierarchy": ontology.hierarchy,
        "metadata": ontology.metadata,
    }


def write_ontology_json(path: str | Path, ontology: Ontology) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ontology_to_dict(ontology), indent=2, default=str), encoding="utf-8")


def ontology_from_dict(data: Dict) -> Ontology:
    ont = Ontology(metadata=dict(data.get("metadata") or {}))
    for c in data.get("classes") or []:
        ont.classes.append(
            ClassEntity(
                label=c.get("label", ""),
                definition=c.get("definition"),
                synonyms=list(c.get("synonyms") or []),
                provenance=list(c.get("provenance") or []),
                stratum=c.get("stratum"),
                original_label=c.get("original_label"),
                evidence=c.get("evidence"),
                aliases=list(c.get("aliases") or []),
            )
        )
    for r in data.get("relations") or []:
        ont.relations.append(
            RelationEntity(
                label=r.get("label", ""),
                domain=r.get("domain"),
                range=r.get("range"),
                definition=r.get("definition"),
                provenance=list(r.get("provenance") or []),
                stratum=r.get("stratum"),
                evidence=r.get("evidence"),
                aliases=list(r.get("aliases") or []),
            )
        )
    ont.hierarchy = list(data.get("hierarchy") or [])
    return ont


def _to_uri_local(label: str) -> str:
    """Convert a human label to a URI-safe local name (PascalCase)."""
    # Remove parenthetical content, strip
    label = re.sub(r"\([^)]*\)", "", label).strip()
    # Split on non-alphanumeric, capitalise each word, join
    parts = re.split(r"[^a-zA-Z0-9]+", label)
    return "".join(p.capitalize() for p in parts if p)


def ontology_to_ttl(data: Dict) -> str:
    """Convert an ontology dict to Turtle (TTL) format string."""
    ns = "http://example.org/triplegen#"
    lines = [
        f"@prefix : <{ns}> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"<{ns}> rdf:type owl:Ontology .",
        "",
    ]

    # Classes
    class_uris: Dict[str, str] = {}
    for c in data.get("classes", []):
        label = (c.get("label") or "").strip()
        if not label:
            continue
        local = _to_uri_local(label)
        if not local:
            continue
        class_uris[label] = local
        lines.append(f":{local} rdf:type owl:Class ;")
        lines.append(f'    rdfs:label "{label}" .')
        defn = c.get("definition")
        if defn:
            safe = str(defn).replace('"', '\\"').replace("\n", " ")
            # Re-emit with comment (rdfs:comment)
            lines[-1] = lines[-1][:-2] + " ;"
            lines.append(f'    rdfs:comment "{safe}" .')
        lines.append("")

    # Hierarchy (subClassOf)
    for h in data.get("hierarchy", []):
        sub = (h.get("subClass") or "").strip()
        sup = (h.get("superClass") or "").strip()
        sub_uri = class_uris.get(sub) or _to_uri_local(sub)
        sup_uri = class_uris.get(sup) or _to_uri_local(sup)
        if sub_uri and sup_uri:
            lines.append(f":{sub_uri} rdfs:subClassOf :{sup_uri} .")

    if data.get("hierarchy"):
        lines.append("")

    # Object properties — declare each property once, then emit
    # per-class existential restrictions (owl:someValuesFrom) rather than
    # flat rdfs:domain/rdfs:range.  This avoids the OWL semantics problem
    # where multiple rdfs:domain statements on one property are interpreted
    # as the intersection of all listed classes.
    seen_props: set = set()
    for r in data.get("relations", []):
        rlabel = (r.get("label") or "").strip()
        if not rlabel:
            continue
        prop_local = _to_uri_local(rlabel)
        if prop_local not in seen_props:
            seen_props.add(prop_local)
            lines.append(f":{prop_local} rdf:type owl:ObjectProperty ;")
            lines.append(f'    rdfs:label "{rlabel}" .')
            lines.append("")

    # Existential restrictions: for each (domain, property, range) triple,
    # emit "DomainClass rdfs:subClassOf [ owl:Restriction on property
    # someValuesFrom RangeClass ]".  This makes each relation scoped to
    # its owning class — e.g. SecondaryInsult ⊑ ∃Worsens.HeadTrauma.
    for r in data.get("relations", []):
        rlabel = (r.get("label") or "").strip()
        domain = (r.get("domain") or "").strip()
        range_ = (r.get("range") or "").strip()
        if not rlabel or not domain or not range_:
            continue
        prop_local = _to_uri_local(rlabel)
        d_uri = class_uris.get(domain) or _to_uri_local(domain)
        r_uri = class_uris.get(range_) or _to_uri_local(range_)
        if d_uri and r_uri:
            lines.append(f":{d_uri} rdfs:subClassOf [")
            lines.append(f"    rdf:type owl:Restriction ;")
            lines.append(f"    owl:onProperty :{prop_local} ;")
            lines.append(f"    owl:someValuesFrom :{r_uri}")
            lines.append(f"] .")

    lines.append("")
    return "\n".join(lines)


def write_summary(path: str | Path, ontology: Ontology) -> None:
    lines = [
        "=== CLASSES ===",
        f"Total: {len(ontology.classes)}",
        "",
    ]
    for cls in ontology.classes:
        parts = [f"- {cls.label}"]
        if getattr(cls, "definition", None) and str(cls.definition).strip():
            parts.append(f"  definition: {cls.definition}")
        if getattr(cls, "evidence", None) and str(cls.evidence).strip():
            parts.append(f"  evidence: {cls.evidence}")
        lines.append("\n".join(parts))
    lines.extend([
        "",
        "=== HIERARCHY (subclass / superclass) ===",
        f"Total: {len(ontology.hierarchy)}",
        "",
    ])
    for edge in ontology.hierarchy:
        sub = edge.get("subClass", "")
        sup = edge.get("superClass", "")
        ev = edge.get("evidence", "")
        lines.append(f"- {sub} -> {sup}")
        if ev and str(ev).strip():
            lines.append(f"  evidence: {ev}")
    lines.extend([
        "",
        "=== RELATIONS ===",
        f"Total: {len(ontology.relations)}",
        "",
    ])
    for rel in ontology.relations:
        label = getattr(rel, "label", "") or ""
        domain = getattr(rel, "domain", "") or ""
        range_ = getattr(rel, "range", "") or ""
        parts = [f"- {label}({domain} -> {range_})"]
        if getattr(rel, "definition", None) and str(rel.definition).strip():
            parts.append(f"  definition: {rel.definition}")
        if getattr(rel, "evidence", None) and str(rel.evidence).strip():
            parts.append(f"  evidence: {rel.evidence}")
        lines.append("\n".join(parts))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
