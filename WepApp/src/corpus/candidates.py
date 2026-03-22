"""Candidate concept extraction utilities for ontology-style hints.

Pipeline Stage 2: Candidate Knowledge Identification
- Linguistics-based: spaCy noun_chunks + POS (NOUN, PROPN) when available (CE4145)
- Fallback: regex 1–3 gram phrases + section headers + capitalized sequences
- Aggressive filtering: no start/end stopwords, no verbs, no generic/admin terms, length cap
- Domain boosting: clinical patterns, measurements, acronyms (ICP, CPP, CVP, EtCO2, SjO2, TCD)
- Section context: suppress candidates harder for References, Membership, governance-heavy sections
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional

_STOPWORDS = {
    "the", "and", "of", "to", "in", "for", "a", "an", "on", "with", "by", "from", "at", "as",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "we", "our", "us",
    "their", "they", "or", "if", "then", "than", "into", "over", "under", "using",
    "per", "via", "also", "not", "but", "so", "yet", "nor", "both", "each", "all", "any",
    "can", "may", "will", "shall", "could", "would", "should", "must",
    "no", "yes", "such", "very", "more", "most", "less", "other", "some", "many", "much",
    "about", "between", "through", "during", "before", "after", "above", "below",
    "which", "who", "whom", "whose", "what", "where", "when", "how", "why",
    "along", "towards", "every", "further", "only", "often", "however",
    "whether", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "once", "twice", "still", "just", "even", "here", "there",
}

# Reject candidates that are purely or start/end with these (too generic for ontology).
_GENERIC_TERMS = frozenset({
    "data", "group", "technology", "model", "process", "system", "method", "approach",
    "information", "study", "results", "analysis", "level", "number", "part", "section",
    "concept", "definition", "dataset", "core", "type", "category", "class", "value",
    "purpose", "objective", "aim", "goal", "scope", "overview", "framework", "protocol",
    "criteria", "standard", "guideline", "recommendation", "practice", "procedure",
    "report", "paper", "document", "text", "paragraph", "page", "chapter",
    # Process/action nouns that are never ontology concepts
    "access", "accuracy", "benefit", "collection", "collaboration", "comments",
    "consensus", "creation", "effort", "efforts", "element", "elements", "exercise",
    "factor", "factors", "field", "fields", "form", "forms", "interface", "knowledge",
    "mechanism", "network", "rules", "set", "standards", "work",
    # Single-word generic terms that waste attention
    "additional", "appropriate", "assigned", "associated", "available", "based",
    "clear", "collect", "collected", "conducting", "context", "control", "created",
    "degree", "described", "designed", "detected", "determined", "developed",
    "expressed", "found", "given", "head", "high", "identified", "included",
    "individual", "made", "medical", "meeting", "new", "often", "only", "open",
    "outlined", "provided", "representing", "required", "used",
    # Research/academic terms
    "feasibility", "prospective", "collaborative", "collaborating", "multi-centre",
    "external", "internal", "linking", "raising", "conducting",
    "members", "member", "organisation", "organization", "participants",
    "research", "researchers", "investigate", "investigation",
    # Single-word verbs/nouns that appear in academic text but are never ontology concepts
    "collecting", "defining", "determining", "performing", "testing",
    "minute", "minutes", "hour", "hours", "test", "tests", "score", "scores",
})

# Admin/process phrases: do not include in hints (align with scope_filter upstream).
_CANDIDATE_ADMIN_SUPPRESS = frozenset({
    "group", "membership", "centres", "centers", "validation", "technology", "database",
    "quality control", "publication", "trials", "committee", "steering", "consortium",
    "references", "acknowledgements", "funding", "ethics", "author contributions",
    "copyright", "license", "licence", "disclosure", "editorial", "correspondence",
    "reprints", "supplement", "appendix", "table", "figure", "fig",
    "dataset", "core dataset", "dataset definition", "core dataset definition",
    # Organisation and institutional terms
    "brainit", "university", "hospital", "centre pi", "chairperson", "pi",
    "non profit", "non profit making", "profit", "grant", "pc software",
    "software", "website", "web site", "internet", "internet based",
    # Research workflow
    "data collection", "data element", "data elements", "data collected",
    "collection form", "data collection form", "feasibility exercise",
    "pilot collection", "pilot study",
})

# Publication / bibliographic metadata — never useful as ontology candidates.
_PUBLICATION_METADATA = frozenset({
    "published", "published online", "received", "accepted", "revised", "epub",
    "doi", "issn", "isbn", "pmid", "pubmed", "volume", "issue", "pages",
    "article", "clinical article", "original article", "research article",
    "review article", "case report", "letter", "editorial", "commentary",
    "open access", "springer", "elsevier", "wiley", "nature", "bmj", "lancet",
    "abstract", "keywords", "key words", "summary", "introduction", "conclusion",
    "discussion", "background", "methods", "materials", "supplementary",
})

# Journal name prefixes/patterns — phrases starting with these are likely journal names.
_JOURNAL_PREFIXES = (
    "acta", "annals", "archives", "british journal", "european journal",
    "international journal", "journal of", "proceedings", "transactions",
    "neurocritical care journal", "intensive care medicine journal",
    "jama", "new england journal", "brain injury journal",
)

# Months and temporal words that are never ontology concepts.
_TEMPORAL_NOISE = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "yesterday", "tomorrow", "year", "years", "month", "months",
    "week", "weeks", "day", "days",
})

# Section labels that trigger stronger candidate suppression (References, governance, etc.).
_SECTIONS_SUPPRESS_HARD = frozenset({
    "references", "reference", "acknowledgements", "acknowledgments", "membership",
    "ethics approval", "author contributions", "conflict of interest", "funding",
    "correspondence", "reprints", "database access", "publication criteria",
})

# Clinical-positive patterns: boost these when they appear (measurements, interventions, acronyms).
_CLINICAL_BOOST_PATTERNS = frozenset({
    "heart rate", "respiration rate", "blood pressure", "mean arterial pressure", "map",
    "intracranial pressure", "icp", "cpp", "cerebral perfusion pressure", "cvp", "central venous",
    "etco2", "sjo2", "sjv o2", "tcd", "transcranial doppler", "pupil", "gcs", "glasgow",
    "hypotension", "hypertension", "intracranial hypertension", "sedation", "ventilation",
    "vasopressors", "blood gases", "haematology", "biochemistry", "microdialysis",
    "secondary insult", "outcome", "monitoring", "therapy", "ct ", " ct", "ct scan",
})
_MAX_CANDIDATE_WORDS = 6
_MAX_CANDIDATE_CHARS = 55

def _is_metadata_noise(phrase: str) -> bool:
    """Catch publication metadata, author names, publisher names, and bibliographic noise."""
    p = phrase.strip()
    pl = p.lower()
    # Phrases containing publication keywords
    for kw in ("published", "online", "received", "accepted", "revised", "epub ahead",
               "doi:", "doi ", "vol.", "vol ", "pp.", "pp ", "no.", "suppl",
               "printed", "verlag", "springer", "elsevier", "wiley", "copyright"):
        if kw in pl:
            return True
    # Pure numbers, years, page ranges (e.g. "2003", "145-152", "15:30")
    if re.match(r"^[\d\s\-:;,./]+$", p):
        return True
    # Single proper-noun word that isn't a known clinical abbreviation — likely author name.
    if len(p.split()) == 1 and p[0].isupper() and not p.isupper():
        if not any(pat in pl for pat in _CLINICAL_BOOST_PATTERNS):
            if len(p) <= 15 and re.match(r"^[A-Z][a-z]+$", p):
                return True
    words = pl.split()
    if any(w in _TEMPORAL_NOISE for w in words):
        return True
    if any(w in _PUBLICATION_METADATA for w in words):
        return True
    # Multi-word Title Case with no clinical signal — likely author names or place names.
    # E.g. "Kiening Schoening Lanksch", "Charite Berlin"
    raw_words = p.split()
    if len(raw_words) >= 2 and all(w[0].isupper() and not w.isupper() for w in raw_words if len(w) > 1):
        if _clinical_boost_score(p) == 0:
            return True
    # Author name patterns: "Miller JD", "Dearden NM", "Piper IR" (Surname + initials)
    if re.match(r"^[A-Z][a-z]+(?: [A-Z]{1,3})+$", p):
        return True
    # Reverse: "JD Miller", "NM Dearden"
    if re.match(r"^[A-Z]{1,3} [A-Z][a-z]+$", p):
        return True
    # Multi-author: "NM Miller JD", "Dearden NM Miller"
    if re.match(r"^(?:[A-Z][a-z]+|[A-Z]{1,3})(?: (?:[A-Z][a-z]+|[A-Z]{1,3}))+$", p):
        has_initials = any(re.match(r"^[A-Z]{1,3}$", w) for w in raw_words)
        has_name = any(re.match(r"^[A-Z][a-z]+$", w) for w in raw_words)
        if has_initials and has_name and _clinical_boost_score(p) == 0:
            return True
    # Institutional names: "UK University Hospital", "Glasgow Royal Infirmary"
    # Only reject when the phrase looks like an institution name (contains
    # a proper-noun word alongside the institutional keyword). Clinical
    # compounds like "hospital mortality" or "hospital admission" are kept.
    _INST_KEYWORDS = ("university", "infirmary", "institute")
    for inst_kw in _INST_KEYWORDS:
        if inst_kw in pl:
            return True
    for inst_kw in ("hospital", "centre", "center"):
        if inst_kw in pl:
            raw_words = p.split()
            if any(w[0].isupper() and w.lower() not in ("hospital", "centre", "center") for w in raw_words if len(w) > 1):
                if _clinical_boost_score(p) == 0:
                    return True
    return False


_nlp = None


def _get_nlp():
    """Lazy-load spaCy English model. Returns None if spacy not installed or model missing.
    To enable: pip install spacy && python -m spacy download en_core_web_sm
    """
    global _nlp
    # _nlp can be the model (callable) or False (failed load). Only return the model.
    if _nlp is not None and _nlp is not False:
        return _nlp
    if _nlp is False:
        return None
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except (OSError, ImportError):
        _nlp = False
        return None
    return _nlp


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text)


def _candidate_rejected(phrase: str, *, section_suppress_hard: bool = False) -> bool:
    """True if candidate should be dropped: metadata, stopword, generic, admin, too long, etc."""
    p = (phrase or "").strip()
    if not p or len(p) <= 2:
        return True
    if len(p) > _MAX_CANDIDATE_CHARS or len(p.split()) > _MAX_CANDIDATE_WORDS:
        return True
    pl = p.lower()
    if pl in _GENERIC_TERMS or pl in _CANDIDATE_ADMIN_SUPPRESS:
        return True
    if pl in _PUBLICATION_METADATA or pl in _TEMPORAL_NOISE:
        return True
    if any(pl.startswith(jp) for jp in _JOURNAL_PREFIXES):
        return True
    if _is_metadata_noise(p):
        return True
    if section_suppress_hard and any(term in pl for term in _CANDIDATE_ADMIN_SUPPRESS if len(term) > 4):
        return True
    tokens = pl.split()
    if tokens[0] in _STOPWORDS or tokens[-1] in _STOPWORDS:
        return True
    if tokens[0] in _GENERIC_TERMS or tokens[-1] in _GENERIC_TERMS:
        return True
    return False


def _candidate_is_admin(phrase: str) -> bool:
    """True if phrase is clearly admin/process/metadata (suppress from hints)."""
    pl = (phrase or "").strip().lower()
    if pl in _CANDIDATE_ADMIN_SUPPRESS or pl in _PUBLICATION_METADATA or pl in _TEMPORAL_NOISE:
        return True
    for term in _CANDIDATE_ADMIN_SUPPRESS:
        if term in pl and len(pl) < 80:
            return True
    if _is_metadata_noise(phrase):
        return True
    return False


def _clinical_boost_score(phrase: str) -> int:
    """Return boost count for domain-aware ranking (higher = more clinical)."""
    pl = (phrase or "").lower()
    return sum(1 for pat in _CLINICAL_BOOST_PATTERNS if pat in pl or pl in pat)


def _phrase_candidates(tokens: Iterable[str]) -> List[str]:
    phrases: List[str] = []
    token_list = list(tokens)
    for i in range(len(token_list)):
        for size in (1, 2, 3):
            if i + size > len(token_list):
                continue
            window = token_list[i : i + size]
            if any(t.lower() in _STOPWORDS for t in window):
                continue
            phrase = " ".join(window)
            if len(phrase) < 3:
                continue
            phrases.append(phrase)
    return phrases


def extract_section_headers(text: str) -> List[str]:
    """Extract section headers from text (for provenance and context)."""
    headers = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.isupper() and len(line.split()) <= 8:
            headers.append(line)
        elif re.match(r"^\d+[\.\)]\s+[A-Z]", line):
            headers.append(line)
        elif re.match(r"^(Introduction|Methods?|Results?|Discussion|Conclusion|Abstract|Background|References?)[:\.]?\s*$", line, re.IGNORECASE):
            headers.append(line)

    return headers


def _contains_verb(phrase: str, doc) -> bool:
    """True if any word in phrase is tagged as verb (reject predicate-like fragments)."""
    if not doc:
        return False
    phrase_words = set((phrase or "").lower().split())
    for token in doc:
        if token.pos_ == "VERB" and token.text.lower() in phrase_words:
            return True
    return False


def _extract_candidates_spacy(
    text: str,
    max_terms: int,
    *,
    section_suppress_hard: bool = False,
) -> Optional[List[str]]:
    """Extract candidate terms using spaCy noun_chunks and NOUN/PROPN. Returns None if spaCy unavailable."""
    nlp = _get_nlp()
    if nlp is None:
        return None
    if not text or not text.strip():
        return []
    doc = nlp(text[:50000])
    candidates: List[tuple] = []  # (phrase, boost_score)
    seen_lower = set()

    def add(phrase: str) -> None:
        p = phrase.strip()
        if _candidate_rejected(p, section_suppress_hard=section_suppress_hard) or _candidate_is_admin(p):
            return
        if _contains_verb(p, doc):
            return
        pl = p.lower()
        if pl in seen_lower:
            return
        seen_lower.add(pl)
        boost = _clinical_boost_score(p)
        candidates.append((p, boost))

    for chunk in doc.noun_chunks:
        add(chunk.text)
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.text.strip():
            add(token.text)

    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [c[0] for c in candidates[:max_terms]]


def _extract_candidates_regex(
    text: str,
    max_terms: int,
    *,
    section_suppress_hard: bool = False,
) -> List[str]:
    """Fallback: regex-based phrase extraction + section headers + capitalized sequences; filtered and boosted."""
    candidates: List[tuple] = []

    if not section_suppress_hard:
        headers = extract_section_headers(text)
        for h in headers[:5]:
            if not _candidate_rejected(h, section_suppress_hard=False) and not _candidate_is_admin(h):
                candidates.append((h, _clinical_boost_score(h)))

    tokens = _tokenize(text)
    if tokens:
        phrases = _phrase_candidates(tokens)
        counts = Counter(phrases)
        for phrase, _ in counts.most_common(max_terms * 2):
            if _candidate_rejected(phrase, section_suppress_hard=section_suppress_hard) or _candidate_is_admin(phrase):
                continue
            candidates.append((phrase, _clinical_boost_score(phrase)))

    for term in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)[:5]:
        if not _candidate_rejected(term, section_suppress_hard=section_suppress_hard) and not _candidate_is_admin(term):
            candidates.append((term, _clinical_boost_score(term)))

    text_lower = text.lower()
    for pat in _CLINICAL_BOOST_PATTERNS:
        if pat in text_lower and pat.strip() not in {c[0].lower() for c in candidates}:
            if not _candidate_rejected(pat.strip(), section_suppress_hard=section_suppress_hard):
                candidates.append((pat.strip(), 2))

    seen = set()
    unique: List[tuple] = []
    for c, boost in candidates:
        cl = c.lower().strip()
        if cl not in seen and len(c) > 2:
            seen.add(cl)
            unique.append((c, boost))
    unique.sort(key=lambda x: (-x[1], x[0]))
    return [u[0] for u in unique[:max_terms]]


def _section_suppress_hard(section: Optional[str]) -> bool:
    """True if section is References, Membership, governance, etc. (suppress candidates harder)."""
    if not (section or "").strip():
        return False
    norm = " ".join((section or "").lower().split())
    return norm in _SECTIONS_SUPPRESS_HARD or any(s in norm for s in ("references", "membership", "governance", "acknowledg"))


def extract_candidates(
    text: str,
    max_terms: int = 12,
    *,
    section: Optional[str] = None,
) -> List[str]:
    """
    Extract candidate concepts from text for ontology-style hints.

    Prefers spaCy noun_chunks + NOUN/PROPN; falls back to regex phrases + section headers.
    Filters: start/end stopwords, verbs, generic/admin terms, too long; boosts clinical patterns.
    When section is References/Membership/governance, suppresses candidates harder.
    """
    section_suppress = _section_suppress_hard(section)
    spacy_candidates = _extract_candidates_spacy(text, max_terms, section_suppress_hard=section_suppress)
    if spacy_candidates is not None:
        if not section_suppress:
            headers = extract_section_headers(text)[:5]
            seen = {c.lower() for c in spacy_candidates}
            for h in reversed(headers):
                if h.lower() not in seen and not _candidate_rejected(h) and not _candidate_is_admin(h):
                    spacy_candidates.insert(0, h)
        return spacy_candidates[:max_terms]
    return _extract_candidates_regex(text, max_terms, section_suppress_hard=section_suppress)
