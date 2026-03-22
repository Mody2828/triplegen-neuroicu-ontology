"""Alignment utilities."""

from __future__ import annotations

from typing import Dict, List, Set

from .match_exact import match_exact
from .match_semantic import match_semantic
from .normalize import normalize_label_for_match
from .synonyms import build_gold_terms_to_index


def align_entities(generated: List[Dict], gold: List[Dict], use_semantic: bool = True) -> Dict[str, List[Dict] | Set[str]]:
    matched_exact, unmatched, exact_gold_indices = match_exact(generated, gold)
    _, gold_normalized_labels = build_gold_terms_to_index(gold)
    gold_labels_matched: Set[str] = {gold_normalized_labels[i] for i in exact_gold_indices}
    if use_semantic and unmatched:
        matched_sem, still_unmatched, matched_gold_indices = match_semantic(
            unmatched, gold, exclude_gold_indices=set(exact_gold_indices),
        )
        for idx in matched_gold_indices:
            gold_labels_matched.add(gold_normalized_labels[idx])
    else:
        matched_sem, still_unmatched = [], unmatched
    return {
        "matched_exact": matched_exact,
        "matched_semantic": matched_sem,
        "unmatched": still_unmatched,
        "gold_labels_matched": gold_labels_matched,
    }
