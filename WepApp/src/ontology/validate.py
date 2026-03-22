"""Ontology validation helpers."""

from __future__ import annotations

from typing import List

from .model import Ontology


def validate_ontology(ontology: Ontology) -> List[str]:
    warnings: List[str] = []
    labels = [c.label for c in ontology.classes]
    if len(labels) != len(set(labels)):
        warnings.append("Duplicate class labels detected.")
    rel_labels = [r.label for r in ontology.relations]
    if len(rel_labels) != len(set(rel_labels)):
        warnings.append("Duplicate relation labels detected.")
    if not ontology.classes:
        warnings.append("No classes generated.")
    return warnings
