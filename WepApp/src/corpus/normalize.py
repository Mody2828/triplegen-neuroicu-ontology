"""Text normalization and cleaning helpers for PDF-derived medical text.

Pipeline Stage 1:
- Fix broken hyphenation across line breaks (e.g. intra-\\ncranial)
- Normalize unicode (fractions, dashes, quotes, OCR artifacts)
- Remove form-feed and other control chars for safe downstream use
- Repeated header/footer removal using page/block boundaries
- Basic page-number and horizontal-rule cleanup
- Strip inline citations and reference markers
- Expand in-document abbreviations in-situ
- Preserves semantic structure for chunking
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

from .clean_chars import strip_control_chars_for_prompt

# Broken hyphenation: word-\nword -> wordword (join before chunking)
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)", re.MULTILINE)

# Unicode fractions -> ASCII slash equivalent (e.g. ¼ -> 1/4); fraction slash ⁄ (U+2044) -> /
_FRACTION_SLASH = "\u2044"  # ⁄
_VULGAR_FRACTIONS = {
    "\u00bc": "1/4", "\u00bd": "1/2", "\u00be": "3/4",
    "\u2153": "1/3", "\u2154": "2/3", "\u2155": "1/5", "\u2156": "2/5", "\u2157": "3/5", "\u2158": "4/5",
    "\u2159": "1/6", "\u215a": "5/6", "\u215b": "1/8", "\u215c": "3/8", "\u215d": "5/8", "\u215e": "7/8",
}

# Fancy dashes and minus -> ASCII hyphen-minus
_DASHES_AND_MINUS = re.compile(r"[\u2010-\u2015\u2212\ufe58\uff0d]")

# Curly/smart quotes -> straight
_QUOTE_LEFT_DOUBLE = "\u201c"
_QUOTE_RIGHT_DOUBLE = "\u201d"
_QUOTE_LEFT_SINGLE = "\u2018"
_QUOTE_RIGHT_SINGLE = "\u2019"

# Repeated OCR-style artifact: 4+ identical non-alphanumeric chars -> cap at 3
_OCR_REPEAT = re.compile(r"([^\w\s])\1{3,}")

# ── Citation / reference stripping patterns ──────────────────────────────────
# Bracketed numeric: [1], [1,2], [1-5], [1, 2, 3], [1,2,5-8]
_CITE_BRACKET_NUM = re.compile(r"\s*\[\d+(?:\s*[,\-–]\s*\d+)*\]")
# Superscript-style numeric citations (often appear after sentences): ¹²³ or similar
_CITE_SUPERSCRIPT = re.compile(r"[\u00b9\u00b2\u00b3\u2070-\u2079]+")
# Author-year inline: (Smith et al., 2020), (Smith and Jones, 2019), (Smith 2020)
_CITE_AUTHOR_YEAR = re.compile(
    r"\(\s*[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?"
    r"(?:\s*,?\s*\d{4}[a-z]?)\s*"
    r"(?:;\s*[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?(?:\s*,?\s*\d{4}[a-z]?)\s*)*\)"
)
# Full References / Bibliography section (from heading to end of text or next major heading)
_REFERENCES_SECTION = re.compile(
    r"\n\s*(?:References|Bibliography|Literature Cited|Works Cited)\s*\n"
    r"(.*?)(?=\n\s*(?:Appendix|Supplement|Acknowledgement|$))",
    re.IGNORECASE | re.DOTALL,
)


# ── In-document abbreviation expansion ────────────────────────────────────────
# Detect "Full Term (ABBREV)" definitions and build a per-document abbreviation table,
# then replace subsequent bare abbreviations with "Full Term (ABBREV)" for clarity.

# Matches patterns like: "intracranial pressure (ICP)", "Glasgow Coma Scale (GCS)"
# Captures: group(1) = full term, group(2) = abbreviation
_ABBREV_DEFINITION = re.compile(
    r"([A-Z][a-z]+(?:\s+(?:of|the|and|for|in|to|by|or|a|an)?\s*[A-Za-z]+){1,6})"  # full term (1-7 words, starts with capital)
    r"\s*\(([A-Z][A-Za-z0-9]{1,8})\)",  # (ABBREV) — 2-9 chars, starts with capital
)

# Known neuro-ICU abbreviations (fallback dictionary for abbreviations not defined in-document)
_KNOWN_ABBREVIATIONS: Dict[str, str] = {
    "ICP": "Intracranial Pressure",
    "CPP": "Cerebral Perfusion Pressure",
    "MAP": "Mean Arterial Pressure",
    "GCS": "Glasgow Coma Scale",
    "GOSe": "Glasgow Outcome Scale Extended",
    "GOS": "Glasgow Outcome Scale",
    "TBI": "Traumatic Brain Injury",
    "HR": "Heart Rate",
    "BP": "Blood Pressure",
    "CVP": "Central Venous Pressure",
    "ECG": "Electrocardiogram",
    "EtCO2": "End-Tidal CO2",
    "SaO2": "Arterial Oxygen Saturation",
    "SjO2": "Jugular Venous Oxygen Saturation",
    "PbrO2": "Brain Tissue Oxygen",
    "PbtO2": "Brain Tissue Oxygen",
    "TCD": "Transcranial Doppler",
    "PRx": "Pressure Reactivity Index",
    "EVD": "External Ventricular Drain",
    "CSF": "Cerebrospinal Fluid",
    "ARDS": "Acute Respiratory Distress Syndrome",
    "TIL": "Therapy Intensity Level",
    "EUSIG": "Edinburgh University Secondary Insult Grades",
    "ABP": "Arterial Blood Pressure",
    "ICU": "Intensive Care Unit",
    "CT": "Computed Tomography",
    "MRI": "Magnetic Resonance Imaging",
    "NIBP": "Non-Invasive Blood Pressure",
    "ANN": "Artificial Neural Network",
    "BTF": "Brain Trauma Foundation",
    "DC": "Decompressive Craniectomy",
    "SBP": "Systolic Blood Pressure",
    "DBP": "Diastolic Blood Pressure",
}


def _build_abbreviation_table(text: str) -> Dict[str, str]:
    """Scan text for 'Full Term (ABBREV)' patterns and build abbreviation → expansion map.

    Also includes known neuro-ICU abbreviations as fallback for undefined abbreviations.
    In-document definitions take priority over the dictionary.
    """
    table: Dict[str, str] = dict(_KNOWN_ABBREVIATIONS)
    for match in _ABBREV_DEFINITION.finditer(text):
        full_term = match.group(1).strip()
        abbrev = match.group(2).strip()
        if len(abbrev) >= 2 and len(full_term) > len(abbrev):
            table[abbrev] = full_term
    return table


def _expand_abbreviations(text: str) -> str:
    """Detect abbreviation definitions in text and expand subsequent bare uses.

    For each 'Full Term (ABBREV)' definition found, subsequent standalone uses of
    ABBREV are replaced with 'Full Term (ABBREV)' so the LLM sees both forms.

    The original definition sites are preserved as-is.
    """
    table = _build_abbreviation_table(text)
    if not table:
        return text

    # Sort by length descending to avoid partial matches (e.g. "ICP" before "IC")
    sorted_abbrevs = sorted(table.keys(), key=len, reverse=True)

    # Build a single regex matching any bare abbreviation (word-bounded)
    # Exclude matches that are already inside parentheses (definition sites)
    pattern = re.compile(
        r"(?<!\()\b(" + "|".join(re.escape(a) for a in sorted_abbrevs) + r")\b(?!\))",
    )

    # Track which abbreviations have been defined (first occurrence with parentheses)
    defined_abbrevs = set()
    for match in _ABBREV_DEFINITION.finditer(text):
        defined_abbrevs.add(match.group(2).strip())

    def _replace(m: re.Match) -> str:
        abbrev = m.group(1)
        full = table.get(abbrev)
        if not full:
            return abbrev
        # Don't expand if it would be redundant (full term already adjacent)
        return f"{full} ({abbrev})"

    # Only expand abbreviations that were defined in-document or are in the known dictionary
    result = pattern.sub(_replace, text)
    return result


def _strip_citations(text: str) -> str:
    """Remove inline citations and the References/Bibliography section.

    Handles: [1], [1,2], [1-5], (Author et al., 2020), superscript numbers,
    and the full References section at the end of the paper.
    """
    # Strip the full References section first (captures multi-line block)
    text = _REFERENCES_SECTION.sub("\n", text)
    # Also catch a simpler pattern: "References\n" followed by numbered entries to end
    text = re.sub(
        r"\n\s*References\s*\n(?:\s*\[?\d+\]?\.?\s+.+\n?)+$",
        "\n",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Strip inline citations
    text = _CITE_BRACKET_NUM.sub("", text)
    text = _CITE_SUPERSCRIPT.sub("", text)
    text = _CITE_AUTHOR_YEAR.sub("", text)
    return text


def _fix_hyphenation(text: str) -> str:
    """Join word-\\nword to wordword (PDF line-break hyphenation)."""
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def _normalize_symbols(text: str) -> str:
    """Unicode fractions to /, fancy dashes to -, curly quotes to straight, OCR repeats capped."""
    for char, replacement in _VULGAR_FRACTIONS.items():
        text = text.replace(char, replacement)
    text = text.replace(_FRACTION_SLASH, "/")
    text = _DASHES_AND_MINUS.sub("-", text)
    text = text.replace(_QUOTE_LEFT_DOUBLE, '"').replace(_QUOTE_RIGHT_DOUBLE, '"')
    text = text.replace(_QUOTE_LEFT_SINGLE, "'").replace(_QUOTE_RIGHT_SINGLE, "'")
    text = _OCR_REPEAT.sub(r"\1\1\1", text)
    return text


def _strip_control_chars(text: str) -> str:
    """Remove form-feed, vertical tab, and other control chars (keep \\t, \\n, \\r)."""
    return strip_control_chars_for_prompt(text)


def _repeated_header_footer_lines(text: str, page_sep: str = "\f", min_frequency_ratio: float = 0.3) -> Tuple[str, List[str]]:
    """
    Find lines that appear as first or last non-empty line in a high proportion of pages/blocks.
    Returns (text_with_seps_intact, set of line strings to remove).
    """
    if not text.strip():
        return text, []
    if page_sep not in text:
        # No form-feed: use double newline as block boundary so we can still drop repeated block headers
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    else:
        blocks = [b.strip() for b in text.split(page_sep) if b.strip()]
    if len(blocks) < 2:
        return text, []
    first_lines: List[str] = []
    last_lines: List[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])
    n = len(blocks)
    threshold = max(2, int(n * min_frequency_ratio))
    count_first = Counter(first_lines)
    count_last = Counter(last_lines)
    to_remove = set()
    for line, c in count_first.items():
        if c >= threshold and len(line) < 200:
            to_remove.add(line)
    for line, c in count_last.items():
        if c >= threshold and len(line) < 200:
            to_remove.add(line)
    return text, list(to_remove)


def _remove_repeated_header_footer(text: str, lines_to_remove: List[str], page_sep: str) -> str:
    """Remove lines that are repeated headers/footers from each block; preserve page_sep."""
    if not lines_to_remove:
        return text
    remove_set = set(lines_to_remove)
    if page_sep in text:
        blocks = text.split(page_sep)
    else:
        blocks = text.split("\n\n")
    out = []
    for block in blocks:
        lines = block.split("\n")
        kept = [ln for ln in lines if ln.strip() not in remove_set]
        out.append("\n".join(kept))
    joiner = page_sep if page_sep in text else "\n\n"
    return joiner.join(out)


def normalize_text(
    text: str,
    *,
    page_sep: str = "\f",
    repeated_header_footer_ratio: float = 0.3,
) -> str:
    """
    Clean and normalize PDF-derived medical text.

    Order:
    1. NFKD unicode normalization + zero-width char removal
    2. Normalize line breaks (\\r\\n, \\r -> \\n)
    3. Fix broken hyphenation (word-\\nword -> wordword)
    4. Normalize symbols (fractions, dashes, quotes, OCR repeats)
    5. Repeated header/footer: drop lines that appear as top/bottom in many pages/blocks
    6. Strip form-feed and other control chars
    7. Strip citations and references section
    8. Basic page-number and horizontal-rule removal
    9. Collapse whitespace, preserve paragraph structure
    10. Expand in-document abbreviations for LLM clarity
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)

    text = _fix_hyphenation(text)
    text = _normalize_symbols(text)

    # Repeated header/footer using page sep or \n\n blocks
    _, repeated_lines = _repeated_header_footer_lines(text, page_sep=page_sep, min_frequency_ratio=repeated_header_footer_ratio)
    text = _remove_repeated_header_footer(text, repeated_lines, page_sep)

    text = text.replace("\f", "\n\n")
    text = _strip_control_chars(text)

    # Strip citations and references (before whitespace collapse so gaps are cleaned up)
    text = _strip_citations(text)

    text = re.sub(r"\n\s*Page\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s+of\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"={3,}", "===", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r"[ \t]+", " ", line.strip())
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
        elif cleaned_lines and cleaned_lines[-1]:
            cleaned_lines.append("")
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Expand abbreviations last — after all cleanup so the text is stable
    text = _expand_abbreviations(text)

    return text.strip()
