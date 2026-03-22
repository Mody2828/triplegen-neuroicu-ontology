"""Ontology export utilities."""

from __future__ import annotations

import json
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
        if getattr(rel, "evidence", None) and str(rel.evidence).strip():
            parts.append(f"  evidence: {rel.evidence}")
        lines.append("\n".join(parts))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
