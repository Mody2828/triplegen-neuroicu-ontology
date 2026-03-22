"""Preprocess text before TF-IDF for retrieval (MMR).

Aligns with CE4145 W5: lowercasing, punctuation normalization, stopword removal,
and stemming so that "Core Dataset", "core dataset", "core-dataset", "core-datasets"
map to consistent tokens and similarity scoring improves.
"""

from __future__ import annotations

import re
from typing import List


def _ensure_nltk() -> None:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
    except Exception:
        pass


def preprocess_for_retrieval(text: str, use_stemming: bool = True) -> str:
    """Normalize text for TF-IDF retrieval: lowercase, punctuation, stopwords, optional stem.

    Applied to pool texts and query (chunk) before building/querying the index
    so that variant forms (e.g. "Core Dataset", "core-dataset", "core-datasets")
    contribute consistently to similarity and MMR selection improves.

    Args:
        text: Raw input string.
        use_stemming: If True, apply Porter stemmer (default). If False, only lower + normalize + stopwords.

    Returns:
        Preprocessed string (tokens joined by spaces) for downstream TF-IDF.
    """
    if not text or not text.strip():
        return ""
    text = str(text).strip()

    # Lowercase
    text = text.lower()

    # Punctuation normalization: replace hyphens with space so "core-dataset" -> "core dataset"
    # and remove other punctuation (keeps alphanumeric and spaces)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and filter
    try:
        _ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text)
        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop and len(t) > 1]
    except Exception:
        # Fallback: simple split and a minimal stopword set
        tokens = [t for t in text.split() if len(t) > 1]
        stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being"}
        tokens = [t for t in tokens if t not in stop]

    if use_stemming:
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        except Exception:
            pass

    return " ".join(tokens) if tokens else ""


def preprocess_texts(texts: List[str], use_stemming: bool = True) -> List[str]:
    """Preprocess a list of texts for retrieval (e.g. pool or corpus)."""
    return [preprocess_for_retrieval(t, use_stemming=use_stemming) for t in texts]
