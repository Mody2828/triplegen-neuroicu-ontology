"""Maximal Marginal Relevance (MMR) selection."""

from __future__ import annotations

from typing import List

import numpy as np


def mmr_select(sim_to_query: np.ndarray, sim_between_docs: np.ndarray, k: int = 3, lambda_param: float = 0.7) -> List[int]:
    if sim_to_query.ndim == 2:
        sim_to_query = sim_to_query.flatten()
    selected: List[int] = []
    candidates = list(range(sim_to_query.shape[0]))
    for _ in range(min(k, len(candidates))):
        if not selected:
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            candidates.remove(idx)
            continue
        scores = []
        for c in candidates:
            diversity = max(sim_between_docs[c, s] for s in selected)
            score = lambda_param * sim_to_query[c] - (1 - lambda_param) * diversity
            scores.append((score, c))
        scores.sort(reverse=True)
        chosen = scores[0][1]
        selected.append(chosen)
        candidates.remove(chosen)
    return selected
