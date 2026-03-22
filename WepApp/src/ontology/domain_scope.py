"""Loader for the domain scope file (resources/domain_scope.json).

The domain scope provides cleanup with domain knowledge about what is in-scope
for neuro-ICU papers — independent of the gold-standard ontology (which is used
for evaluation metrics only).

All sets are lazily loaded and cached on first access.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Set

_SCOPE_FILE = Path(__file__).resolve().parents[2] / "resources" / "domain_scope.json"


@lru_cache(maxsize=1)
def _load_scope() -> dict:
    """Load and cache the domain scope JSON."""
    with open(_SCOPE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm(label: str) -> str:
    """Lowercase, strip, collapse whitespace — must match reasoner._norm_label."""
    import re
    return re.sub(r"\s+", " ", label.strip().lower())


@lru_cache(maxsize=1)
def get_in_scope_classes_norm() -> FrozenSet[str]:
    """Normalized set of all in-scope class labels from domain scope."""
    return frozenset(_norm(c) for c in _load_scope().get("in_scope_classes", []))


@lru_cache(maxsize=1)
def get_in_scope_relations() -> FrozenSet[str]:
    """All in-scope relation labels (canonical form)."""
    return frozenset(_load_scope().get("in_scope_relations", []))


@lru_cache(maxsize=1)
def get_evidence_exemptions_norm() -> FrozenSet[str]:
    """Normalized set of classes exempt from evidence pruning."""
    exemptions = _load_scope().get("evidence_exemptions", {})
    labels = exemptions.get("labels", []) if isinstance(exemptions, dict) else exemptions
    return frozenset(_norm(c) for c in labels)


@lru_cache(maxsize=1)
def get_abstract_data_allowlist_norm() -> FrozenSet[str]:
    """Normalized set of classes allowed through abstract-data filter."""
    al = _load_scope().get("abstract_data_allowlist", {})
    labels = al.get("labels", []) if isinstance(al, dict) else al
    return frozenset(_norm(c) for c in labels)


@lru_cache(maxsize=1)
def get_broad_context_allowlist_norm() -> FrozenSet[str]:
    """Normalized set of classes allowed through broad-contextual filter."""
    al = _load_scope().get("broad_context_allowlist", {})
    labels = al.get("labels", []) if isinstance(al, dict) else al
    return frozenset(_norm(c) for c in labels)


@lru_cache(maxsize=1)
def get_concept_categories() -> dict:
    """Concept categories dict (category_name → list of class labels)."""
    return _load_scope().get("concept_categories", {})


@lru_cache(maxsize=1)
def get_out_of_scope_classes_norm() -> FrozenSet[str]:
    """Normalized set of classes that should always be removed (methodology/statistics terms)."""
    oos = _load_scope().get("out_of_scope_classes", {})
    labels = oos.get("labels", []) if isinstance(oos, dict) else oos
    return frozenset(_norm(c) for c in labels)


def is_in_scope(label: str) -> bool:
    """True if label is in the domain scope's in-scope classes."""
    return _norm(label) in get_in_scope_classes_norm()
