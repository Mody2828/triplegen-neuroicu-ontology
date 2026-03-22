"""Parse LLM outputs into structured data."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any

# Minimum evidence length (chars) for classes/relations; shorter evidence is rejected (stricter checks).
# Kept in sync with schema.MIN_EVIDENCE_LENGTH_STRICT to avoid circular import at module load.
MIN_EVIDENCE_LENGTH = 12

from ..corpus.clean_chars import strip_control_chars_for_prompt


def _sanitize_json_text(s: str) -> str:
    """Replace illegal JSON control characters with space; keep newline/tab/CR (pipeline-wide clean_chars)."""
    if not s:
        return s
    return strip_control_chars_for_prompt(s)


def _sanitize_json_string(s: str) -> str:
    """Alias for backward compatibility; same as _sanitize_json_text."""
    return _sanitize_json_text(s)


def safe_json_loads(text: str) -> Any:
    """Parse JSON after replacing control characters. Use for LLM output or any file that may contain bad chars."""
    if not text or not text.strip():
        raise ValueError("Empty input")
    return json.loads(_sanitize_json_text(text))


def parse_output(raw: str, *, min_evidence_length: int | None = None) -> Dict:
    """Parse LLM JSON output. Sanitizes control chars before parsing (safety net).
    Falls back to truncated-JSON recovery when the response was cut off mid-output.
    Caller may save the original raw string when parsed is empty for debugging.

    Args:
        min_evidence_length: Override the default MIN_EVIDENCE_LENGTH for evidence
            filtering. Pass 1 to accept any non-empty evidence (e.g. for SGC where
            short labels like "EtCO2" or "TCD" are legitimate corpus mentions).
    """
    raw = _sanitize_json_text(raw or "")
    parsed = _try_parse_json(raw)
    if parsed is None:
        parsed = _try_parse_embedded_json(raw)
    if parsed is None:
        parsed = _try_recover_truncated_json(raw)
    if parsed is None:
        parsed = _try_parse_triples(raw)
    if parsed is None:
        return {"classes": [], "relations": [], "hierarchy": []}
    return normalize_parsed(parsed, min_evidence_length=min_evidence_length)


def _try_parse_json(raw: str) -> Any | None:
    """Parse full string as JSON; sanitize immediately before json.loads so no control chars reach the decoder."""
    try:
        return json.loads(_sanitize_json_text(raw or ""))
    except json.JSONDecodeError:
        return None


def _try_parse_embedded_json(raw: str) -> Any | None:
    """Extract {...} snippet and parse; sanitize immediately before json.loads."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        return json.loads(_sanitize_json_text(snippet))
    except json.JSONDecodeError:
        return None


def _try_recover_truncated_json(raw: str) -> Any | None:
    """Attempt to recover a valid ontology dict from truncated JSON.

    When the LLM response is cut off mid-output (e.g. max_tokens reached), the
    JSON will be incomplete. This function tries to close open arrays/objects
    so that at least the classes, relations, and hierarchy parsed so far are
    recovered instead of returning nothing.
    """
    start = raw.find("{")
    if start == -1:
        return None
    fragment = _sanitize_json_text(raw[start:])

    # Strip trailing whitespace and commas that would invalidate closings
    fragment_stripped = fragment.rstrip()
    while fragment_stripped.endswith(","):
        fragment_stripped = fragment_stripped[:-1].rstrip()

    for closing in (
        "]}",        # truncated after a complete array element
        "}]}",       # truncated inside the last hierarchy/relation/class object
        "\"}]}",     # truncated inside a string value
        "}]]}",      # truncated inside an array element with one extra nesting
        "]]}",       # truncated after a complete array element (nested)
    ):
        # Try the pre-stripped fragment first (most common case)
        for frag in (fragment_stripped, fragment):
            for trim in range(min(120, len(frag))):
                candidate = frag[: len(frag) - trim] + closing
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and ("classes" in obj or "relations" in obj or "hierarchy" in obj):
                        return obj
                except json.JSONDecodeError:
                    continue
    return None


def _try_parse_triples(raw: str) -> Any | None:
    triples = re.findall(r"\[([^\[\]]+)\]", raw)
    if not triples:
        return None
    triple_list = []
    for t in triples:
        parts = [p.strip().strip("'\"") for p in t.split(",")]
        if len(parts) >= 3:
            triple_list.append(parts[:3])
    if not triple_list:
        return None
    return {"Triples": triple_list}


def _clear_definition_when_no_evidence(item: Dict) -> None:
    """Clear definition when evidence is missing (avoid storing external/encyclopedic definitions)."""
    ev = (item.get("evidence") or "").strip()
    if not ev and item.get("definition"):
        item["definition"] = ""


def _has_evidence(item: Dict, evidence_key: str = "evidence") -> bool:
    """Return True if the item has a non-empty evidence field (evidence-first rule)."""
    ev = (item.get(evidence_key) or "").strip()
    return bool(ev)


def _has_sufficient_evidence(item: Dict, min_length: int = MIN_EVIDENCE_LENGTH, evidence_key: str = "evidence") -> bool:
    """Return True if the item has evidence with at least min_length characters (reject trivial/single-word evidence)."""
    ev = (item.get(evidence_key) or "").strip()
    return len(ev) >= min_length


def _has_domain_and_range(relation: Dict) -> bool:
    """Return True if the relation has non-empty domain and range (relation minimalism: no vague edges)."""
    domain = (relation.get("domain") or "").strip()
    range_val = (relation.get("range") or "").strip()
    return bool(domain) and bool(range_val)


def normalize_parsed(parsed: Any, *, min_evidence_length: int | None = None) -> Dict:
    """
    Normalize parsed LLM output to standard ontology dict.
    Preserves optional fields on each item (e.g. confidence, evidence).
    Clears definition when evidence is empty so external definitions are not persisted.
    Evidence-first: classes and relations without a non-empty evidence field are dropped
    (reject option; omission over hallucination). Hierarchy is kept as-is.
    Confidence: optional float 0.0-1.0 per class/relation for extraction confidence scoring.

    Args:
        min_evidence_length: Override default MIN_EVIDENCE_LENGTH. Pass 1 to accept
            any non-empty evidence (useful for SGC where short labels are valid).
    """
    eff_min = min_evidence_length if min_evidence_length is not None else MIN_EVIDENCE_LENGTH
    if isinstance(parsed, dict):
        if "Triples" in parsed:
            return triples_to_ontology(parsed.get("Triples", []))
        if "classes" in parsed or "relations" in parsed:
            classes = list(parsed.get("classes", []))
            relations = list(parsed.get("relations", []))
            hierarchy = list(parsed.get("hierarchy", []))
            for c in classes:
                _clear_definition_when_no_evidence(c)
            for r in relations:
                _clear_definition_when_no_evidence(r)
            classes = [c for c in classes if _has_evidence(c) and _has_sufficient_evidence(c, min_length=eff_min)]
            relations = [
                r for r in relations
                if _has_evidence(r) and _has_sufficient_evidence(r, min_length=eff_min) and _has_domain_and_range(r)
            ]
            return {"classes": classes, "relations": relations, "hierarchy": hierarchy}
    return {"classes": [], "relations": [], "hierarchy": []}


def triples_to_ontology(triples: List[List[str]]) -> Dict:
    classes = {}
    relations = {}
    for subj, pred, obj in triples:
        classes[subj] = {"label": subj, "definition": None, "synonyms": []}
        classes[obj] = {"label": obj, "definition": None, "synonyms": []}
        relations[pred] = {"label": pred, "domain": subj, "range": obj, "definition": None}
    return {"classes": list(classes.values()), "relations": list(relations.values()), "hierarchy": []}
