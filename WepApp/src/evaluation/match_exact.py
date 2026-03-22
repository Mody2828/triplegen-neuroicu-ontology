"""Exact matching rules with synonym awareness."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .normalize import normalize_label_for_match
from .synonyms import build_gold_terms_to_index


def match_exact(generated: List[Dict], gold: List[Dict]) -> Tuple[List[Dict], List[Dict], List[int]]:
    """Returns (matched_generated, unmatched_generated, matched_gold_indices). Synonym-aware.

    Enforces one-to-one matching: each gold class can only be claimed once.
    If multiple generated classes map to the same gold index, only the first
    is counted as matched; subsequent duplicates go to unmatched.
    """
    term_to_index, _ = build_gold_terms_to_index(gold)
    matched = []
    unmatched = []
    matched_gold_idx: List[int] = []
    claimed_gold: Set[int] = set()
    for item in generated:
        key = normalize_label_for_match(item.get("label") or "")
        if key in term_to_index:
            gold_idx = term_to_index[key]
            if gold_idx not in claimed_gold:
                matched.append(item)
                matched_gold_idx.append(gold_idx)
                claimed_gold.add(gold_idx)
            else:
                unmatched.append(item)
        else:
            unmatched.append(item)
    return matched, unmatched, matched_gold_idx
