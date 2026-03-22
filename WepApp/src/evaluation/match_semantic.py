"""Semantic matching using TF-IDF similarity; synonym-aware gold expansion."""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .normalize import normalize_label_for_match
from .synonyms import DOMAIN_SYNONYMS, _derive_synonyms_from_label, build_gold_terms_to_index


def _gold_doc_tokens(g: Dict, gold_normalized_labels: List[str], index: int) -> str:
    """One gold class as a space-joined doc: label + synonyms + derived + domain alts for TF-IDF."""
    label = g.get("label") or ""
    norm = gold_normalized_labels[index] if index < len(gold_normalized_labels) else normalize_label_for_match(label)
    parts = [norm]
    for s in g.get("synonyms") or []:
        if isinstance(s, str):
            parts.append(normalize_label_for_match(s))
    for d in _derive_synonyms_from_label(label):
        parts.append(normalize_label_for_match(d))
    for alt, canonical in DOMAIN_SYNONYMS:
        if normalize_label_for_match(canonical) == norm:
            parts.append(normalize_label_for_match(alt))
    return " ".join(p for p in parts if p)


def match_semantic(
    generated: List[Dict],
    gold: List[Dict],
    threshold: float | None = None,
    *,
    exclude_gold_indices: Set[int] | None = None,
) -> Tuple[List[Dict], List[Dict], List[int]]:
    """Returns (matched_generated, unmatched_generated, matched_gold_indices). Synonym-aware gold docs for TF-IDF.

    ``exclude_gold_indices`` — gold indices already claimed by exact matching.
    Semantic matching will not assign any generated class to these indices,
    preventing one-to-many gold matching that inflates recall.
    """
    if not generated or not gold:
        return [], generated, []
    if threshold is None:
        threshold = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.55"))
    _, gold_normalized_labels = build_gold_terms_to_index(gold)
    gen_labels = [normalize_label_for_match(g.get("label") or "") for g in generated]
    gold_docs = [_gold_doc_tokens(gold[i], gold_normalized_labels, i) for i in range(len(gold))]
    vectorizer = TfidfVectorizer().fit(gen_labels + gold_docs)
    gen_vec = vectorizer.transform(gen_labels)
    gold_vec = vectorizer.transform(gold_docs)
    sims = cosine_similarity(gen_vec, gold_vec)
    matched = []
    unmatched = []
    matched_gold_idx = []
    claimed_gold: set = set(exclude_gold_indices or ())
    scored = []
    for i, row in enumerate(sims):
        j = int(row.argmax())
        scored.append((i, j, float(row[j])))
    scored.sort(key=lambda x: -x[2])
    decided: set = set()
    for i, j, score in scored:
        if i in decided:
            continue
        decided.add(i)
        if score >= threshold and j not in claimed_gold:
            matched.append(generated[i])
            matched_gold_idx.append(j)
            claimed_gold.add(j)
        else:
            unmatched.append(generated[i])
    return matched, unmatched, matched_gold_idx
