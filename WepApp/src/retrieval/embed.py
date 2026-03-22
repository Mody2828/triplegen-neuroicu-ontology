"""Embedding interface using TF-IDF for lightweight runs.

Text is preprocessed before TF-IDF (lowercase, punctuation normalization,
stopwords, stemming) so variant forms (e.g. "Core Dataset", "core-dataset")
map consistently and MMR retrieval quality improves (CE4145 W5).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import preprocess_texts


@dataclass
class EmbeddingIndex:
    vectorizer: TfidfVectorizer
    matrix: object

    def query(self, texts: List[str]):
        preprocessed = preprocess_texts(texts)
        return self.vectorizer.transform(preprocessed)


def build_index(texts: List[str]) -> EmbeddingIndex:
    preprocessed = preprocess_texts(texts)
    # ngram_range=(1,2): unigrams + bigrams so phrases ("core dataset", "monitoring data", "traumatic brain injury") get semantic weight
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(preprocessed)
    return EmbeddingIndex(vectorizer=vectorizer, matrix=matrix)


def save_index(path: str | Path, index: EmbeddingIndex) -> None:
    Path(path).write_bytes(pickle.dumps(index))


def load_index(path: str | Path) -> EmbeddingIndex:
    return pickle.loads(Path(path).read_bytes())
