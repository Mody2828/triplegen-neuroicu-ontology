"""Internal ontology representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClassEntity:
    label: str
    definition: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)
    stratum: Optional[str] = None  # core | governance | provenance (from chunk routing)
    # Expansion strategy: set when class was produced by semantic mapping (original_label -> label).
    original_label: Optional[str] = None
    # Exact verbatim sentence fragment from the text (required for extraction contract).
    evidence: Optional[str] = None
    # Surface-form variants merged into this canonical class (e.g. "CVP", "Central Venous Pressure (CVP)").
    aliases: List[str] = field(default_factory=list)


@dataclass
class RelationEntity:
    label: str
    domain: Optional[str] = None
    range: Optional[str] = None
    definition: Optional[str] = None
    provenance: List[str] = field(default_factory=list)
    stratum: Optional[str] = None  # core | governance | provenance
    # Exact verbatim sentence fragment from the text (required for extraction contract).
    evidence: Optional[str] = None
    # Surface-form variants merged into this canonical relation.
    aliases: List[str] = field(default_factory=list)


@dataclass
class Ontology:
    classes: List[ClassEntity] = field(default_factory=list)
    relations: List[RelationEntity] = field(default_factory=list)
    hierarchy: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def add_class(self, entity: ClassEntity) -> None:
        self.classes.append(entity)

    def add_relation(self, entity: RelationEntity) -> None:
        self.relations.append(entity)
