"""Corpus ingestion and provenance tracking.

Implements Pipeline Stage 1: Corpus Ingestion & Preprocessing
- Input: BrainIT publications and Neuro-ICU literature (PDF/text)
- Load → strip control chars → normalize → optional scope filter → chunk
- Preservation of provenance (source, section, paragraph; optional page boundaries, raw_text)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .clean_chars import strip_control_chars_for_prompt
from .normalize import normalize_text
from .scope_filter import apply_scope_filter

from ..common.logging import log_event


def _extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[str]]:
    """Extract text from PDF. Returns (full_text_with_formfeed_sep, list_of_page_texts) for provenance."""
    try:
        import PyPDF2
    except ImportError:
        raise RuntimeError(
            "PyPDF2 is not installed. Install it with: pip install PyPDF2"
        )
    text_content: List[str] = []
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_content.append(page_text)
        return "\f".join(text_content), text_content
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF {pdf_path}: {str(e)}")


def _read_text_file(file_path: Path) -> str:
    """Read text file with encoding fallback."""
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # If all encodings fail, try with error handling
    return file_path.read_text(encoding='utf-8', errors='replace')


def _build_doc(
    doc_id: str,
    source: str,
    raw: str,
    file_type: str,
    *,
    scope_filter: bool = False,
    keep_raw: bool = False,
    pages: Optional[List[str]] = None,
) -> Dict:
    """Normalize, strip control chars, optionally scope-filter; build doc dict with optional raw_text and pages."""
    text = normalize_text(raw or "")
    text = strip_control_chars_for_prompt(text)
    if scope_filter:
        text = apply_scope_filter(text, clinical_only=True)
    doc = {
        "id": doc_id,
        "source": source,
        "text": text,
        "file_type": file_type,
    }
    if keep_raw:
        doc["raw_text"] = raw
    if pages is not None:
        doc["pages"] = pages
    return doc


def load_corpus(
    path: str | Path,
    *,
    scope_filter: bool = False,
    keep_raw: bool = False,
) -> List[Dict]:
    """
    Load corpus from directory or single file.

    Pipeline: read raw → strip control chars → normalize → optional scope filter.

    Supports: .txt, .pdf.

    Returns documents with: id, source, text (normalized/cleaned), file_type.
    Optionally: raw_text (if keep_raw=True), pages (for PDFs, page-level segments for provenance).
    """
    base = Path(path)

    if base.is_file():
        ext = base.suffix.lower()
        if ext == ".pdf":
            full_text, pages = _extract_text_from_pdf(base)
            return [
                _build_doc(
                    base.stem,
                    str(base),
                    full_text,
                    "pdf",
                    scope_filter=scope_filter,
                    keep_raw=keep_raw,
                    pages=pages,
                )
            ]
        if ext == ".txt":
            raw = _read_text_file(base)
            return [
                _build_doc(
                    base.stem,
                    str(base),
                    raw,
                    "text",
                    scope_filter=scope_filter,
                    keep_raw=keep_raw,
                )
            ]
        raise ValueError(f"Unsupported file type: {ext}. Only .txt and .pdf are supported.")

    documents = []
    for file_path in sorted(base.glob("**/*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in (".txt", ".pdf"):
            continue
        try:
            if ext == ".pdf":
                full_text, pages = _extract_text_from_pdf(file_path)
                documents.append(
                    _build_doc(
                        file_path.stem,
                        str(file_path),
                        full_text,
                        "pdf",
                        scope_filter=scope_filter,
                        keep_raw=keep_raw,
                        pages=pages,
                    )
                )
            else:
                raw = _read_text_file(file_path)
                documents.append(
                    _build_doc(
                        file_path.stem,
                        str(file_path),
                        raw,
                        "text",
                        scope_filter=scope_filter,
                        keep_raw=keep_raw,
                    )
                )
        except Exception as e:
            log_event("ingest_file_error", {"file": str(file_path), "error": str(e)})
            continue
    if not documents and base.is_dir():
        log_event("ingest_empty_corpus", {"path": str(base)})
    return documents


def write_manifest(path: str | Path, docs: List[Dict[str, str]]) -> None:
    manifest = [{"id": d["id"], "source": d["source"]} for d in docs]
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
