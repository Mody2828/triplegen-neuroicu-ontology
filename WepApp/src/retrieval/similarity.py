"""Similarity helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim_matrix(query_matrix, doc_matrix) -> np.ndarray:
    return cosine_similarity(query_matrix, doc_matrix)
