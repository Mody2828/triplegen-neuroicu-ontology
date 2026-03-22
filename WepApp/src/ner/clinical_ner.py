"""Clinical NER: pre-identify entities in text to anchor LLM extraction (Medical Evidence Anchor).

Uses ScispaCy (or Med-spaCy if available) to extract biomedical/clinical entities from
each chunk before sending to the LLM. These are passed as "Suggested Concepts" so the
LLM focuses on relationships between verified concepts and reduces hallucinations.

Optional dependency: install with
  pip install scispacy
  pip install https://s2-public-api-prod.us-west-2.ons3.amazonaws.com/allenai/scibert_scivocab_uncased/a0b363f3c5c0a0c0a0c0a0c0a0c0a0c0a0c0a0c.tar.gz
  # Or the NER model (diseases/chemicals):
  pip install https://s2-public-api-prod.us-west-2.ons3.amazonaws.com/.../en_ner_bc5cdr_md-0.5.4.tar.gz

If scispacy is not installed, extract_suggested_concepts returns [] and extraction
proceeds without the anchor (no failure).
"""

from __future__ import annotations

import re
import warnings
from typing import List, Optional

# Module-level cache for the loaded pipeline (lazy load once per process)
_NLP = None
_NER_AVAILABLE = None
_NLP_MODEL_NAME: Optional[str] = None


def get_loaded_ner_model_name() -> Optional[str]:
    """Return the name of the NER model currently in use, or None if none loaded. Call after extract_suggested_concepts to see which model was used."""
    _load_nlp()
    return _NLP_MODEL_NAME


def _load_nlp():
    """Load clinical NER pipeline: try ScispaCy models, then Med-spaCy; set _NLP and _NER_AVAILABLE."""
    global _NLP, _NER_AVAILABLE, _NLP_MODEL_NAME
    if _NER_AVAILABLE is not None:
        return
    _NER_AVAILABLE = False
    _NLP = None
    _NLP_MODEL_NAME = None
    # 1) ScispaCy: NER models (diseases/chemicals) or generic sci pipelines
    for model_name in ("en_ner_bc5cdr_md", "en_ner_jnlpba_md", "en_core_sci_sm", "en_core_sci_md"):
        try:
            import spacy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                _NLP = spacy.load(model_name)
            _NER_AVAILABLE = True
            _NLP_MODEL_NAME = model_name
            return
        except (OSError, ImportError):
            continue
    # 2) Med-spaCy: clinical pipeline (if installed)
    for medspacy_model in ("en_core_med7_trf", "en_core_med7_lg", "en_core_med7_md"):
        try:
            import medspacy
            _NLP = medspacy.load(medspacy_model)
            if _NLP is not None:
                _NER_AVAILABLE = True
                _NLP_MODEL_NAME = medspacy_model
                return
        except Exception:
            continue


def extract_suggested_concepts(text: str, max_entities: int = 150) -> List[str]:
    """Pre-identify clinical/biomedical entities in the text for use as suggested concepts.

    Before extraction, the pipeline can call this on each chunk and inject the result
    into the prompt so the LLM prefers these verified terms and avoids inventing others.

    Args:
        text: Raw chunk text (e.g. one chunk of BrainIT literature).
        max_entities: Maximum number of unique concept labels to return (cap for long texts).

    Returns:
        List of unique, cleaned entity labels (e.g. "Intracranial Pressure", "CPP").
        Empty list if NER is unavailable or text is empty.
    """
    if not (text and isinstance(text, str) and text.strip()):
        return []
    text = text.strip()
    _load_nlp()
    if not _NER_AVAILABLE or _NLP is None:
        return []
    try:
        doc = _NLP(text[:500000])  # cap input length to avoid OOM
    except Exception:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for ent in doc.ents:
        label = (ent.text or "").strip()
        if not label or len(label) < 2:
            continue
        key = re.sub(r"\s+", " ", label).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= max_entities:
            break
    return out
