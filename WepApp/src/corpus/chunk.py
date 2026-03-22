"""Corpus chunking utilities.

Semantic-first, size-second strategy:
- Segment by structure: headings → paragraphs → sentences (in that order).
- Merge adjacent segments until approaching max_chunk_tokens, without crossing
  semantic breaks (section headings, tables/figures).
- Guardrails: never split mid-sentence unless necessary; if a paragraph exceeds
  max, split by sentences with overlap.
- Overlap: 5–10% of chunk size, capped (e.g. 150–300 tokens), sentence-aligned.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from .clean_chars import strip_control_chars_for_prompt
from .candidates import extract_candidates


# Token-based policy (approximate: 1 word ≈ 1.35 tokens for English).
# Larger chunks give the LLM more context per call for richer extraction.
# Semantic breaks (headings, tables, domain section boundaries) are always
# respected regardless of size — chunks never cross section boundaries.
DEFAULT_TARGET_CHUNK_TOKENS = 2000
DEFAULT_HARD_MAX_TOKENS = 3200
DEFAULT_MIN_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 200  # sentence-aligned; ~10% of target

_WORDS_PER_TOKEN = 1.0 / 1.35


def _words_to_tokens(words: int) -> int:
    """Approximate word count to token count (English)."""
    return max(0, int(words * (1.0 / _WORDS_PER_TOKEN)))


def _tokens_to_words(tokens: int) -> int:
    """Approximate token count to word count."""
    return max(0, int(tokens * _WORDS_PER_TOKEN))


def _text_token_estimate(text: str) -> int:
    return _words_to_tokens(len(text.split()))


# Domain-specific section boundaries (BrainIT / Neuro-ICU paper family). Force chunk breaks on these
# so chunks align with meaningful ontology section boundaries (Part A/B, core dataset, outcome, etc.).
_DOMAIN_SECTION_BOUNDARIES = frozenset({
    "part a", "part b", "part c",
    "core dataset", "outcome", "minute by minute", "minute-by-minute",
    "intensive care management", "secondary insult treatment", "secondary insults",
    "demographic and clinical information",
})

# ── Section-type classification for extraction priority ──────────────────────
# Maps section headings to semantic types so downstream prompting can weight
# clinical-definition sections higher and skip methodology/statistics sections.
_SECTION_TYPE_HIGH = re.compile(
    r"(?:introduction|background|overview|definition|clinical|management|treatment|therapy|"
    r"monitoring|assessment|classification|pathophysiology|diagnosis|intensive\s+care|"
    r"secondary\s+insult|core\s+dataset|part\s+[a-c]|demographic|admission|outcome|"
    r"minute.by.minute|data\s+categor|data\s+item|parameter)",
    re.IGNORECASE,
)
_SECTION_TYPE_MEDIUM = re.compile(
    r"(?:results?|findings?|discussion|interpretation|analysis|summary|conclusions?)",
    re.IGNORECASE,
)
_SECTION_TYPE_LOW = re.compile(
    r"(?:method|statistical|regression|logistic|cohort|study\s+design|"
    r"data\s+collection|sample\s+size|inclusion\s+criteria|exclusion\s+criteria|"
    r"appendix|supplement|acknowledg|funding|conflict|abstract)",
    re.IGNORECASE,
)
_SECTION_TYPE_SKIP = re.compile(
    r"(?:reference|bibliograph|literature\s+cited|works\s+cited)",
    re.IGNORECASE,
)


def classify_section_type(section_heading: Optional[str]) -> str:
    """Classify a section heading into an extraction priority tier.

    Returns one of: 'high', 'medium', 'low', 'skip', or 'unknown'.
    - high: clinical definitions, management, monitoring — richest for ontology extraction
    - medium: results/discussion — may contain useful concepts but mixed with data
    - low: methodology, statistics, study design — mostly noise for ontology
    - skip: references/bibliography — should be excluded entirely
    - unknown: heading doesn't match any pattern
    """
    if not section_heading:
        return "unknown"
    h = section_heading.strip()
    if _SECTION_TYPE_SKIP.search(h):
        return "skip"
    if _SECTION_TYPE_HIGH.search(h):
        return "high"
    if _SECTION_TYPE_LOW.search(h):
        return "low"
    if _SECTION_TYPE_MEDIUM.search(h):
        return "medium"
    return "unknown"


def _detect_section_header(line: str) -> bool:
    """Detect if a line is likely a section header (semantic break)."""
    line = line.strip()
    if not line:
        return False
    line_lower = line.lower()
    if line_lower in _DOMAIN_SECTION_BOUNDARIES:
        return True
    if line.isupper() and len(line.split()) <= 8:
        return True
    if re.match(r"^\d{1,2}[\.\)]\s+[A-Z]", line) and len(line.split()) <= 8 and not re.search(r"\(\d{4}\)", line):
        return True
    if re.match(
        r"^(Introduction|Summary|Methods?|Results?|Discussion|Conclusions?|Abstract|Background|References?|Findings?|Interpretation)[:\.]?\s*$",
        line,
        re.IGNORECASE,
    ):
        return True
    return False


def _looks_like_table_or_figure(block: str) -> bool:
    """Treat as atomic unit: tables, figure captions, or dense numeric blocks."""
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if not lines:
        return False
    first = lines[0]
    if re.match(r"^(Table|Figure|Fig\.)\s+\d+", first, re.IGNORECASE):
        return True
    if re.match(r"^Appendix\s+[A-Z0-9]", first, re.IGNORECASE):
        return True
    # Many pipes or tabs suggest a table
    if len(lines) >= 2 and ("\t" in first or "|" in first):
        return True
    return False


_ABBREV_RE = re.compile(
    r"\b(?:e\.g|i\.e|et al|Fig|Figs|vs|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|No|Vol|Dept|Eq|Ref|Sect|approx|resp|sys|min|max|avg|temp|approx|incl|excl|ca)\."
)

def _sentence_split(text: str) -> List[str]:
    """Split on sentence boundaries, protecting common abbreviations from false splits."""
    _PLACEHOLDER = "\x00ABBR\x00"
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", _PLACEHOLDER), text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [s.replace(_PLACEHOLDER, ".").strip() for s in parts if s.strip()]


def _take_sentence_aligned_overlap(
    text: str, overlap_tokens: int
) -> Tuple[str, int]:
    """Return (overlap_text, token_estimate) from the end of text, aligned to sentence boundary.

    Caps overlap at 50% of the source text to avoid duplicating most of a chunk.
    """
    sentences = _sentence_split(text)
    if not sentences:
        return "", 0
    total_words = sum(len(s.split()) for s in sentences)
    max_overlap_words = max(1, total_words // 2)
    overlap_words_target = min(_tokens_to_words(overlap_tokens), max_overlap_words)
    collected = []
    word_count = 0
    for s in reversed(sentences):
        w = len(s.split())
        if word_count + w > overlap_words_target and collected:
            break
        collected.insert(0, s)
        word_count += w
    overlap_text = " ".join(collected)
    return overlap_text, _text_token_estimate(overlap_text)


def _split_paragraph_by_sentences_with_overlap(
    paragraph: str,
    hard_max_tokens: int,
    overlap_tokens: int,
    section_context: Optional[str],
) -> List[Dict]:
    """Split a single paragraph that exceeds hard_max by sentences, with overlap. Never split mid-sentence."""
    sentences = _sentence_split(paragraph)
    if not sentences:
        return []
    chunks = []
    current = []
    current_tokens = 0
    idx = 0
    for sent in sentences:
        st = _text_token_estimate(sent)
        if current_tokens + st > hard_max_tokens and current:
            chunk_text_str = " ".join(current)
            chunks.append({
                "chunk_id": f"sub-{idx}",
                "text": chunk_text_str,
                "word_count": len(chunk_text_str.split()),
                **({"section": section_context} if section_context else {}),
            })
            idx += 1
            overlap_text, _ = _take_sentence_aligned_overlap(chunk_text_str, overlap_tokens)
            current = [overlap_text] if overlap_text else []
            current_tokens = _text_token_estimate(overlap_text)
        current.append(sent)
        current_tokens += st
    if current:
        chunk_text_str = " ".join(current)
        chunks.append({
            "chunk_id": f"sub-{idx}",
            "text": chunk_text_str,
            "word_count": len(chunk_text_str.split()),
            **({"section": section_context} if section_context else {}),
        })
    return chunks


def _segment_by_structure(text: str) -> List[Dict]:
    """
    Segment text into semantic units: heading, paragraph, or table_figure.
    Each unit: {"type": str, "text": str, "tokens": int}.
    """
    blocks = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []
    section_context = None
    for block in blocks:
        lines = block.split("\n")
        if len(lines) == 1 and _detect_section_header(lines[0]):
            section_context = lines[0]
            units.append({
                "type": "heading",
                "text": lines[0],
                "tokens": _text_token_estimate(lines[0]),
                "section": section_context,
            })
            continue
        if len(lines) > 1 and _detect_section_header(lines[0].strip()):
            section_context = lines[0].strip()
            units.append({
                "type": "heading",
                "text": section_context,
                "tokens": _text_token_estimate(section_context),
                "section": section_context,
            })
            remainder = "\n".join(lines[1:]).strip()
            if remainder:
                if _looks_like_table_or_figure(remainder):
                    units.append({
                        "type": "table_figure",
                        "text": remainder,
                        "tokens": _text_token_estimate(remainder),
                        "section": section_context,
                    })
                else:
                    units.append({
                        "type": "paragraph",
                        "text": remainder,
                        "tokens": _text_token_estimate(remainder),
                        "section": section_context,
                    })
            continue
        if _looks_like_table_or_figure(block):
            units.append({
                "type": "table_figure",
                "text": block,
                "tokens": _text_token_estimate(block),
                "section": section_context,
            })
            continue
        units.append({
            "type": "paragraph",
            "text": block,
            "tokens": _text_token_estimate(block),
            "section": section_context,
        })
    return units


def chunk_text_semantic_first(
    text: str,
    target_chunk_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Semantic-first, size-second chunking.
    - Segment by structure (headings → paragraphs → sentences).
    - Merge adjacent segments until approaching target, without crossing semantic breaks.
    - Overlap is sentence-aligned (5–10% of chunk, capped).
    """
    units = _segment_by_structure(text)
    chunks = []
    current_parts = []
    current_tokens = 0
    current_section = None
    overlap_for_next = ""
    overlap_tokens_for_next = 0
    index = 0
    # Heading-only chunks are never emitted: heading is kept as metadata/context and merged into
    # the same chunk as the immediately following content (paragraph or table_figure). If the
    # document ends with a heading and no following content, that heading is dropped.
    pending_heading = None

    def flush_current():
        nonlocal current_parts, current_tokens, current_section, overlap_for_next, overlap_tokens_for_next, index
        if not current_parts:
            return
        chunk_text_str = "\n\n".join(p["text"] for p in current_parts)
        overlap_for_next, overlap_tokens_for_next = _take_sentence_aligned_overlap(
            chunk_text_str, min(overlap_tokens, int(0.1 * (current_tokens or 1)))
        )
        overlap_tokens_for_next = min(300, max(100, overlap_tokens_for_next))
        sec_type = classify_section_type(current_section)
        chunks.append({
            "chunk_id": f"chunk-{index}",
            "text": chunk_text_str,
            "word_count": len(chunk_text_str.split()),
            **({"section": current_section} if current_section else {}),
            "section_type": sec_type,
        })
        index += 1
        current_parts = []
        current_tokens = 0
        current_section = None

    def prepend_pending_heading():
        """If pending_heading is set, add it as first part and clear. Returns added tokens."""
        nonlocal pending_heading, current_parts, current_tokens
        if not pending_heading:
            return 0
        current_parts.insert(0, pending_heading)
        current_tokens += pending_heading["tokens"]
        added = pending_heading["tokens"]
        pending_heading = None
        return added

    i = 0
    while i < len(units):
        u = units[i]
        if u["type"] == "heading":
            flush_current()
            current_section = u.get("section")
            pending_heading = {"text": u["text"], "tokens": u["tokens"], "type": "heading"}
            i += 1
            continue
        if u["type"] == "table_figure":
            flush_current()
            text_for_chunk = u["text"]
            if pending_heading:
                text_for_chunk = pending_heading["text"] + "\n\n" + text_for_chunk
                pending_heading = None
            tf_sec = u.get("section")
            chunks.append({
                "chunk_id": f"chunk-{index}",
                "text": text_for_chunk,
                "word_count": len(text_for_chunk.split()),
                **({"section": tf_sec} if tf_sec else {}),
                "section_type": classify_section_type(tf_sec),
            })
            index += 1
            i += 1
            continue
        # paragraph
        if u["tokens"] > hard_max_tokens:
            flush_current()
            sub_chunks = _split_paragraph_by_sentences_with_overlap(
                u["text"], hard_max_tokens, overlap_tokens, u.get("section")
            )
            for si, sc in enumerate(sub_chunks):
                if pending_heading and si == 0:
                    sc["text"] = pending_heading["text"] + "\n\n" + sc["text"]
                    pending_heading = None
                chunks.append({**sc, "chunk_id": f"chunk-{index}"})
                index += 1
            i += 1
            continue
        # Merge paragraph into current if we don't cross semantic break
        candidate_tokens = overlap_tokens_for_next + current_tokens + u["tokens"]
        if pending_heading:
            candidate_tokens += pending_heading["tokens"]
        if overlap_for_next:
            current_parts.insert(0, {"text": overlap_for_next, "tokens": overlap_tokens_for_next})
            current_tokens += overlap_tokens_for_next
            overlap_for_next = ""
            overlap_tokens_for_next = 0
        # Would exceed hard max: flush and start fresh with u (and pending heading if any)
        if candidate_tokens > hard_max_tokens and current_parts:
            flush_current()
            overlap_for_next = ""
            overlap_tokens_for_next = 0
            if pending_heading:
                current_parts = [pending_heading, u]
                current_tokens = pending_heading["tokens"] + u["tokens"]
                pending_heading = None
            else:
                current_parts = [u]
                current_tokens = u["tokens"]
            current_section = u.get("section")
            i += 1
            continue
        # At or above target: flush so we don't grow past target; then start new chunk with u (and pending heading if any)
        if current_tokens >= target_chunk_tokens and current_parts:
            flush_current()
            overlap_for_next = ""
            overlap_tokens_for_next = 0
            if pending_heading:
                current_parts = [pending_heading, u]
                current_tokens = pending_heading["tokens"] + u["tokens"]
                pending_heading = None
            else:
                current_parts = [u]
                current_tokens = u["tokens"]
            current_section = u.get("section")
            i += 1
            continue
        # Below target: merge u into current (avoid tiny chunks: prefer merging until min)
        prepend_pending_heading()
        current_parts.append(u)
        current_tokens += u["tokens"]
        if u.get("section"):
            current_section = u.get("section")
        i += 1

    flush_current()

    if min_chunk_tokens > 0 and len(chunks) > 1:
        merged: List[Dict] = []
        for ch in chunks:
            if merged and _text_token_estimate(ch["text"]) < min_chunk_tokens:
                prev = merged[-1]
                prev["text"] = prev["text"] + "\n\n" + ch["text"]
                prev["word_count"] = len(prev["text"].split())
                if ch.get("section") and not prev.get("section"):
                    prev["section"] = ch["section"]
            elif merged and _text_token_estimate(merged[-1]["text"]) < min_chunk_tokens:
                prev = merged[-1]
                prev["text"] = prev["text"] + "\n\n" + ch["text"]
                prev["word_count"] = len(prev["text"].split())
                if ch.get("section"):
                    prev["section"] = ch["section"]
            else:
                merged.append(ch)
        if len(merged) > 1 and _text_token_estimate(merged[-1]["text"]) < min_chunk_tokens:
            tail = merged.pop()
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + tail["text"]
            merged[-1]["word_count"] = len(merged[-1]["text"].split())
        for i, ch in enumerate(merged):
            ch["chunk_id"] = f"chunk-{i}"
        chunks = merged

    return chunks


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> List[Dict[str, str]]:
    """
    Legacy: semantic chunking by word count (sentence boundaries preserved).
    Prefer chunk_text_semantic_first for new code.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk_words = []
    current_chunk_text = []
    word_count = 0
    index = 0
    section_context = None
    for para_idx, paragraph in enumerate(paragraphs):
        lines = paragraph.split("\n")
        if len(lines) == 1 and _detect_section_header(lines[0]):
            section_context = lines[0]
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            if word_count + sentence_word_count > chunk_size and current_chunk_words:
                chunk_text_str = " ".join(current_chunk_text)
                chunk_meta = {
                    "chunk_id": f"chunk-{index}",
                    "text": chunk_text_str,
                    "word_count": word_count,
                }
                if section_context:
                    chunk_meta["section"] = section_context
                chunks.append(chunk_meta)
                overlap_words = current_chunk_words[-overlap:] if len(current_chunk_words) > overlap else current_chunk_words
                current_chunk_words = overlap_words
                current_chunk_text = [" ".join(overlap_words)] if overlap_words else []
                word_count = len(overlap_words)
                index += 1
            current_chunk_words.extend(sentence_words)
            current_chunk_text.append(sentence)
            word_count += sentence_word_count
    if current_chunk_words:
        chunk_text_str = " ".join(current_chunk_text)
        chunk_meta = {
            "chunk_id": f"chunk-{index}",
            "text": chunk_text_str,
            "word_count": word_count,
        }
        if section_context:
            chunk_meta["section"] = section_context
        chunks.append(chunk_meta)
    return chunks


def chunk_documents(
    docs: Iterable[Dict[str, str]],
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    *,
    target_chunk_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[Dict[str, str]]:
    """
    Chunk documents with provenance metadata.

    Uses semantic-first, size-second strategy by default (token-based policy).
    If chunk_size and overlap are both passed (legacy word-based), uses legacy chunk_text.

    Each chunk includes: chunk_id, doc_id, source, text, candidates, section (if any), paragraph_index.
    """
    use_legacy = chunk_size is not None and overlap is not None
    all_chunks: List[Dict[str, str]] = []
    for doc in docs:
        if use_legacy:
            chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        else:
            chunks = chunk_text_semantic_first(
                doc["text"],
                target_chunk_tokens=target_chunk_tokens,
                hard_max_tokens=hard_max_tokens,
                min_chunk_tokens=min_chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = strip_control_chars_for_prompt(chunk["text"] or "")
            section = chunk.get("section")
            candidates = extract_candidates(chunk_text, section=section)
            chunk_meta = {
                "chunk_id": f"{doc['id']}::{chunk['chunk_id']}",
                "doc_id": doc["id"],
                "source": doc["source"],
                "text": chunk_text,
                "candidates": candidates,
                "paragraph_index": chunk_idx,
            }
            if section is not None:
                chunk_meta["section"] = section
            all_chunks.append(chunk_meta)
    return all_chunks
