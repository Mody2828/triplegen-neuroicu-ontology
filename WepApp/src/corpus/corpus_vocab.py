"""Corpus-derived vocabulary loaded from resources/corpus_dictionary.json.

Provides descriptive (not prescriptive) vocabulary knowledge extracted from the
11 BrainIT papers. Used by Strict mode for evidence matching, deduplication,
and scope filtering without relying on gold standard knowledge.

The dictionary is built once by ``scripts/build_corpus_dict.py`` and checked
into the repo. This module loads it at import time for fast runtime access.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

_DICT_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "corpus_dictionary.json"

# Loaded once at import time
_DICT: Dict = {}
_ABBREVIATIONS: Dict[str, str] = {}
_REVERSE_ABBREVIATIONS: Dict[str, str] = {}
_CLINICAL_TERMS: Dict[str, Dict] = {}
_SYNONYM_GROUPS: List[Dict] = []
_SYNONYM_LOOKUP: Dict[str, List[str]] = {}
_RELATION_VERBS: List[str] = []


def _load() -> None:
    """Load the corpus dictionary from disk (called once at module import)."""
    global _DICT, _ABBREVIATIONS, _REVERSE_ABBREVIATIONS
    global _CLINICAL_TERMS, _SYNONYM_GROUPS, _SYNONYM_LOOKUP, _RELATION_VERBS

    if not _DICT_PATH.exists():
        return

    try:
        with open(_DICT_PATH, "r", encoding="utf-8") as f:
            _DICT = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        import warnings
        warnings.warn(f"corpus_vocab: failed to load {_DICT_PATH}: {exc}")
        return

    _ABBREVIATIONS = _DICT.get("abbreviations", {})
    _REVERSE_ABBREVIATIONS = _DICT.get("reverse_abbreviations", {})
    _CLINICAL_TERMS = _DICT.get("clinical_terms", {})
    _SYNONYM_GROUPS = _DICT.get("synonym_groups", [])
    _RELATION_VERBS = _DICT.get("relation_verbs", [])

    # Build synonym lookup: normalized term -> list of all variants in the group
    for group in _SYNONYM_GROUPS:
        canonical = group.get("canonical", "")
        variants = group.get("variants", [])
        all_forms = [canonical] + variants
        norm_key = canonical.lower().strip()
        _SYNONYM_LOOKUP[norm_key] = all_forms
        for v in variants:
            v_norm = v.lower().strip()
            if v_norm not in _SYNONYM_LOOKUP:
                _SYNONYM_LOOKUP[v_norm] = all_forms


_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_abbreviation_expansion(abbr: str) -> Optional[str]:
    """Look up a clinical abbreviation and return its full expansion, or None."""
    return _ABBREVIATIONS.get(abbr.upper().strip())


def get_reverse_abbreviation(term: str) -> Optional[str]:
    """Look up a clinical term and return its abbreviation, or None."""
    return _REVERSE_ABBREVIATIONS.get(term.lower().strip())


def get_synonym_group(term: str) -> List[str]:
    """Return all known variants for a term (including the term itself).

    Returns an empty list if the term is not in any synonym group.
    """
    return _SYNONYM_LOOKUP.get(term.lower().strip(), [])


def is_corpus_clinical_term(term: str, min_papers: int = 2) -> bool:
    """True if the term appears as a clinical term in at least *min_papers* papers."""
    info = _CLINICAL_TERMS.get(term.lower().strip())
    if info is None:
        return False
    return info.get("count", 0) >= min_papers


def get_term_paper_count(term: str) -> int:
    """Return the number of papers that mention this clinical term (0 if unknown)."""
    info = _CLINICAL_TERMS.get(term.lower().strip())
    return info.get("count", 0) if info else 0


def get_all_abbreviations() -> Dict[str, str]:
    """Return the full abbreviation → expansion map."""
    return dict(_ABBREVIATIONS)


def get_all_clinical_terms() -> Dict[str, Dict]:
    """Return the full clinical term index."""
    return dict(_CLINICAL_TERMS)


def get_all_synonym_groups() -> List[Dict]:
    """Return all synonym groups."""
    return list(_SYNONYM_GROUPS)


def get_relation_verbs() -> List[str]:
    """Return all known clinical relation verb phrases."""
    return list(_RELATION_VERBS)


def corpus_dict_loaded() -> bool:
    """True if the corpus dictionary was successfully loaded."""
    return bool(_ABBREVIATIONS)
