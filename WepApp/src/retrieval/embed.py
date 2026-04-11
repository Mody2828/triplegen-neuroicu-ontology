"""Embedding interface using sentence-transformers for semantic retrieval.

Replaces TF-IDF with dense semantic embeddings (all-MiniLM-L6-v2, 384-dim)
so that synonym-rich clinical terms (e.g. "CPP" vs "Cerebral Perfusion Pressure")
map to nearby vectors and MMR retrieval quality improves.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Singleton model instance — loaded once, reused across calls
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@dataclass
class EmbeddingIndex:
    matrix: np.ndarray  # (n_docs, 384) dense float32

    def query(self, texts: List[str]) -> np.ndarray:
        """Encode query texts into the same embedding space."""
        model = _get_model()
        return model.encode(texts, show_progress_bar=False, batch_size=64)


def build_index(texts: List[str]) -> EmbeddingIndex:
    """Encode a list of texts into dense semantic vectors."""
    model = _get_model()
    matrix = model.encode(texts, show_progress_bar=False, batch_size=64)
    return EmbeddingIndex(matrix=matrix)


def save_index(path: str | Path, index: EmbeddingIndex) -> None:
    Path(path).write_bytes(pickle.dumps(index))


def load_index(path: str | Path) -> EmbeddingIndex:
    return pickle.loads(Path(path).read_bytes())
