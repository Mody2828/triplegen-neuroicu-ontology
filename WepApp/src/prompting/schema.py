"""Schema validator: whitelist classes/relations and optionally require evidence."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from .vocabulary import ALLOWED_CLASSES_CORE, CLASS_SYNONYM_MAP

# Lexical triggers that must appear in evidence for a hierarchy edge to be accepted (strict source fidelity).
# "include:" and "includes:" capture the BrainIT paper's dominant enumeration pattern:
#   "monitoring data required...include: Heart Rate, Respiration Rate, MAP..."
#   "Types of data include: GCS scores, ventilation settings, sedation levels..."
HIERARCHY_LEXICAL_TRIGGERS = (
    "such as", "is a", "type of", "kind of", "include:", "includes:",
    "consists of", "comprises", "comprised of", "composed of",
    "categorized as", "classified as", "grouped into", "subdivided into",
    "subtypes", "subtype of", "forms of", "categories of",
    "types include", "types of", "subclasses of", "subclass of",
    "is a subclass of", "is a form of", "is a type of",
)

# Minimum character length for evidence when stricter checks are enabled (reject trivial/single-word "evidence").
MIN_EVIDENCE_LENGTH_STRICT = 12

# Known acronyms/short forms allowed as class labels when require_label_in_evidence is True (label need not be literal substring of evidence).
ALLOWED_LABEL_ACRONYMS: frozenset = frozenset({
    "icp", "tbi", "cvp", "tcd", "etco2", "cpp", "gcs", "bp", "sbp", "dbp",
    "map", "eeg", "ecg", "ct", "mri", "dv", "pc", "rdf", "sql",
    "pbro2", "pbto2", "ptio2", "sao2", "spo2", "sjo2", "fio2",
    "prx", "lax", "rap", "amp", "ccp", "ptd",
    "gose", "gos", "gms", "gcsv", "til",
    "abg", "wbc", "hb", "na", "hr", "hrt",
    "poe", "pre", "dai", "sah", "ich",
    "abp", "abpm", "abps", "bpm", "bps", "nibp",
    "evd", "nirs", "cppopt", "ards",
})


def _normalize_label(label: str) -> str:
    raw = (label or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "", raw)
    return raw


def _normalize_whitespace(s: str) -> str:
    """Collapse whitespace to single space and strip, for substring matching."""
    return " ".join((s or "").split())


def _label_in_evidence_or_allowed_acronym(
    label: str, evidence: str, *, allowed_acronyms: Iterable[str] = ()
) -> bool:
    """True if normalized label is a substring of normalized evidence, or label is an allowed acronym/short form."""
    if not label or not evidence:
        return False
    ev_norm = _normalize_whitespace(evidence).lower()
    label_clean = (label or "").strip()
    # Strip parenthetical suffix e.g. "CVP (Central Venous Pressure)" -> use "Central Venous Pressure" and "CVP"
    open_idx = label_clean.find("(")
    close_idx = label_clean.find(")", open_idx + 1 if open_idx >= 0 else 0)
    if open_idx >= 0 and close_idx > open_idx:
        main = label_clean[:open_idx].strip()
        inside = label_clean[open_idx + 1 : close_idx].strip()
        candidates = [_normalize_whitespace(main).lower(), _normalize_whitespace(inside).lower()]
    else:
        candidates = [_normalize_whitespace(label_clean).lower()]
    for c in candidates:
        if c and c in ev_norm:
            return True
    # Allow known acronyms (e.g. ICP, TBI) even if not literal substring
    acronym_set = set(allowed_acronyms) if allowed_acronyms else set(ALLOWED_LABEL_ACRONYMS)
    for c in candidates:
        if c and c in acronym_set:
            return True
    return False


def filter_parsed_to_vocabulary(
    parsed: Dict,
    allowed_classes: Iterable[str],
    allowed_relations: Iterable[str],
    *,
    require_evidence: bool = False,
    chunk_text: str | None = None,
    min_evidence_length: int = 0,
    relation_domains: Mapping[str, Tuple[str | None, str | None]] | None = None,
    relation_aliases: Mapping[str, str] | None = None,
    gold_hierarchy: Iterable[Dict] | None = None,
    require_hierarchy_lexical_cues: bool = False,
    require_label_in_evidence: bool = False,
) -> Dict:
    """Restrict parsed output to allowed labels; optionally drop items without evidence.

    Returns a new dict with keys classes, relations, hierarchy. Only items whose label is in
    the whitelist are kept. If require_evidence is True, classes/relations without a
    non-empty "evidence" field are dropped. If chunk_text is provided and require_evidence
    is True, evidence must be a substring of chunk_text (after normalizing whitespace).
    If min_evidence_length > 0 and require_evidence is True, evidence must have at least
    that many characters (rejects trivial/single-word evidence).     If require_hierarchy_lexical_cues
    is True, hierarchy edges are filtered to those whose evidence contains at least one of:
    "such as", "is a", "type of", "kind of". relation_aliases maps LLM variants to canonical labels.
    If require_label_in_evidence is True, classes are kept only when the label (or parenthetical form)
    is a substring of the evidence, or the label is in ALLOWED_LABEL_ACRONYMS (reject over-abstracted labels).
    """
    class_map = {_normalize_label(c): c for c in allowed_classes}
    rel_map = {_normalize_label(r): r for r in allowed_relations}
    if relation_aliases:
        for alias, canonical in relation_aliases.items():
            rel_map[_normalize_label(alias)] = canonical

    # Build a lightweight "is subclass of" checker from the gold hierarchy (if provided).
    parent_map: Dict[str, Set[str]] = {}
    if gold_hierarchy:
        for e in gold_hierarchy:
            sub = _normalize_label(e.get("subClass") or "")
            sup = _normalize_label(e.get("superClass") or "")
            if not sub or not sup:
                continue
            parent_map.setdefault(sub, set()).add(sup)

    def is_subclass(sub_label: str, sup_label: str) -> bool:
        sub = _normalize_label(sub_label or "")
        sup = _normalize_label(sup_label or "")
        if not sub or not sup:
            return False
        if sub == sup:
            return True
        # BFS up the parent chain; sizes are small (restricted gold).
        seen: Set[str] = set()
        stack = list(parent_map.get(sub, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur == sup:
                return True
            stack.extend(parent_map.get(cur, set()))
        return False

    chunk_normalized = _normalize_whitespace(chunk_text) if chunk_text else ""

    def keep_class(item: Dict) -> bool:
        raw_label = (item.get("label") or "").strip()
        label = _normalize_label(raw_label)
        if label not in class_map:
            return False
        if require_evidence:
            ev = (item.get("evidence") or "").strip()
            if not ev:
                return False
            if min_evidence_length > 0 and len(ev) < min_evidence_length:
                return False
            if chunk_normalized and _normalize_whitespace(ev) not in chunk_normalized:
                return False
        if require_label_in_evidence:
            ev = (item.get("evidence") or "").strip()
            if not _label_in_evidence_or_allowed_acronym(raw_label, ev):
                return False
        return True

    def keep_relation(item: Dict) -> bool:
        label = _normalize_label(item.get("label") or "")
        if label not in rel_map:
            return False
        # Require domain and range for every relation (relation minimalism: no vague edges).
        if not (item.get("domain") or "").strip() or not (item.get("range") or "").strip():
            return False
        if require_evidence:
            ev = (item.get("evidence") or "").strip()
            if not ev:
                return False
            if min_evidence_length > 0 and len(ev) < min_evidence_length:
                return False
            if chunk_normalized and _normalize_whitespace(ev) not in chunk_normalized:
                return False
        return True

    def keep_hierarchy(item: Dict) -> bool:
        sub = _normalize_label(item.get("subClass") or "")
        sup = _normalize_label(item.get("superClass") or "")
        if sub not in class_map or sup not in class_map:
            return False
        if require_evidence:
            ev = (item.get("evidence") or "").strip()
            if not ev:
                return False
            if min_evidence_length > 0 and len(ev) < min_evidence_length:
                return False
            if chunk_normalized and _normalize_whitespace(ev) not in chunk_normalized:
                return False
        if require_hierarchy_lexical_cues:
            ev = (item.get("evidence") or "").strip()
            ev_lower = ev.lower()
            if not any(cue in ev_lower for cue in HIERARCHY_LEXICAL_TRIGGERS):
                return False
        return True

    def _strip_definition_if_no_evidence(item: Dict, key: str = "evidence") -> None:
        """Clear definition when evidence is missing (safety net against hallucinated definitions on evidenceless items).
        Definitions are now LLM-generated scope notes, so we no longer require them to be substrings of evidence."""
        ev = (item.get(key) or "").strip()
        defn = (item.get("definition") or "").strip()
        if not defn:
            return
        if not ev:
            item["definition"] = ""

    classes = []
    for c in parsed.get("classes") or []:
        if not keep_class(c):
            continue
        _strip_definition_if_no_evidence(c)
        canonical = class_map[_normalize_label(c.get("label") or "")]
        c["label"] = canonical
        classes.append(c)

    relations = []
    for r in parsed.get("relations") or []:
        if not keep_relation(r):
            continue
        _strip_definition_if_no_evidence(r)
        canonical = rel_map[_normalize_label(r.get("label") or "")]
        r["label"] = canonical
        # Normalize domain/range to canonical class labels when possible.
        if r.get("domain"):
            domain_key = _normalize_label(r.get("domain") or "")
            if domain_key in class_map:
                r["domain"] = class_map[domain_key]
        if r.get("range"):
            range_key = _normalize_label(r.get("range") or "")
            if range_key in class_map:
                r["range"] = class_map[range_key]
        if relation_domains and canonical in relation_domains:
            domain_req, range_req = relation_domains[canonical]
            # Validate (do NOT overwrite): allow subclasses of the required domain/range.
            if domain_req:
                if not r.get("domain"):
                    r["domain"] = domain_req
                elif not is_subclass(r.get("domain") or "", domain_req):
                    continue
            if range_req:
                if not r.get("range"):
                    r["range"] = range_req
                elif not is_subclass(r.get("range") or "", range_req):
                    continue
        relations.append(r)

    hierarchy = []
    for h in parsed.get("hierarchy") or []:
        if not keep_hierarchy(h):
            continue
        h["subClass"] = class_map[_normalize_label(h.get("subClass") or "")]
        h["superClass"] = class_map[_normalize_label(h.get("superClass") or "")]
        hierarchy.append(h)

    if require_hierarchy_lexical_cues and hierarchy:
        hierarchy = filter_hierarchy_to_lexical_cues(hierarchy, HIERARCHY_LEXICAL_TRIGGERS)

    return {"classes": classes, "relations": relations, "hierarchy": hierarchy}


def filter_hierarchy_to_lexical_cues(
    hierarchy: Iterable[Dict],
    triggers: Tuple[str, ...] = HIERARCHY_LEXICAL_TRIGGERS,
) -> List[Dict]:
    """Keep only hierarchy edges whose evidence contains at least one of the given lexical triggers."""
    result = []
    for edge in hierarchy:
        ev = (edge.get("evidence") or "").strip().lower()
        if not ev:
            continue
        if any(t in ev for t in triggers):
            result.append(edge)
    return result


def extract_hierarchy_from_text(chunk_text: str) -> List[Dict]:
    """
    Dedicated hierarchy pass: scan chunk text with regex for lexical triggers
    ("such as", "is a type of", "is a kind of", "include:", "includes:") and return
    hierarchy edges with evidence. The "include:" pattern is the dominant enumeration
    style in the BrainIT paper (e.g. "monitoring data...include: Heart Rate, Respiration
    Rate, MAP..."; "Types of data include: ventilation settings, sedation levels...").
    Guarantees trigger-based extraction even when the LLM underperforms (Improvement #8).
    """
    if not chunk_text or not chunk_text.strip():
        return []
    text = " ".join(chunk_text.split())
    edges: List[Dict] = []
    seen: Set[Tuple[str, str]] = set()

    def _add_edge(sub: str, super_: str, evidence: str) -> None:
        sub_c = (sub or "").strip()
        super_c = (super_ or "").strip()
        if not sub_c or not super_c or sub_c.lower() == super_c.lower():
            return
        if len(evidence.strip()) < 12:
            return
        key = (sub_c, super_c)
        if key in seen:
            return
        seen.add(key)
        edges.append({
            "subClass": sub_c,
            "superClass": super_c,
            "evidence": evidence.strip(),
        })

    # Split into sentences (and short phrases) so we don't match across sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 15:
            continue
        sent_lower = sent.lower()

        # "X such as Y [, Z] [and W]" -> superClass=X, subClass=Y (and Z, W)
        _such_as = " such as "
        if _such_as in sent_lower:
            idx = sent_lower.find(_such_as)
            super_phrase = sent[:idx].strip()
            rest = sent[idx + len(_such_as):].strip()
            # Trim trailing punctuation/phrase so we don't pull in next sentence
            rest = re.sub(r"[.!?].*$", "", rest).strip()
            if not super_phrase or len(super_phrase) < 2:
                continue
            # Take last noun phrase before "such as" (e.g. "secondary insults" from "e.g. secondary insults such as")
            super_candidate = re.split(r"\s*,\s*|\s+and\s+", super_phrase)[-1].strip()
            if len(super_candidate) < 2:
                super_candidate = super_phrase
            # Split rest by comma, " and ", " or "
            for part in re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", rest):
                sub_candidate = part.strip()
                if len(sub_candidate) >= 2 and sub_candidate != super_candidate:
                    _add_edge(sub_candidate, super_candidate, sent)

        # "X include: Y [, Z] [and W]" / "X includes: Y [, Z]" -> superClass=X, subClass=Y (and Z, W)
        # This is the dominant enumeration pattern in the BrainIT paper:
        #   "monitoring data required...include: Heart Rate, Respiration Rate, MAP, ICP, SaO2 and Temperature"
        #   "Types of data include: GCS scores, ventilation settings, sedation levels..."
        for inc_trigger in (" include:", " includes:"):
            idx = sent_lower.find(inc_trigger)
            if idx >= 0:
                super_phrase = sent[:idx].strip()
                rest = sent[idx + len(inc_trigger):].strip()
                rest = re.sub(r"[.!?].*$", "", rest).strip()
                if not super_phrase or len(super_phrase) < 2:
                    continue
                # Strip trailing sentence-glue connectors ("... Database and include:")
                super_phrase_clean = re.sub(r"\s+(?:and|or|which|that|whose)\s*$", "", super_phrase, flags=re.IGNORECASE).strip()
                # Take the last meaningful noun phrase: split by comma, "and", "or", then
                # strip trailing verb/preposition clauses and cap to a clean noun phrase.
                super_parts = re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", super_phrase_clean)
                super_candidate = super_parts[-1].strip()
                if not super_candidate or len(super_candidate) < 2:
                    super_candidate = super_phrase_clean
                # Strip leading articles
                super_candidate = re.sub(r"^(?:the|a|an)\s+", "", super_candidate, flags=re.IGNORECASE).strip()
                # Strip trailing verb/preposition clauses (e.g. "required for ...", "collected during ...")
                super_candidate = re.sub(
                    r"\s+(?:required|collected|measured|used|needed|obtained|recorded|given|administered|available)\s+.*$",
                    "", super_candidate, flags=re.IGNORECASE,
                ).strip()
                # Cap super_candidate to 5 words (take leading words — the noun phrase head)
                super_words = super_candidate.split()
                if len(super_words) > 5:
                    super_candidate = " ".join(super_words[:5])

                # Split sub-class list by comma first; then handle the trailing "X and Y"
                # only when BOTH sides are plausible standalone class names (≥1 word, right
                # side starts uppercase OR is a known terminal).  This avoids splitting the
                # compound "fluid input and output" into ["fluid input", "output"].
                comma_parts = re.split(r"\s*,\s*", rest)
                sub_candidates: List[str] = []
                for cp in comma_parts:
                    cp = cp.strip()
                    and_match = re.search(r"\s+and\s+", cp, re.IGNORECASE)
                    if and_match:
                        left = cp[:and_match.start()].strip()
                        right = cp[and_match.end():].strip()
                        # Split on "and" only when right side looks like a standalone class:
                        # starts uppercase, or is ≥ 2 words, or is a short well-known acronym.
                        right_is_standalone = (
                            (right and right[0].isupper())
                            or len(right.split()) >= 2
                            or len(right) <= 5  # short acronym like "SaO2", "ICP"
                        )
                        left_makes_sense = left and len(left) >= 2
                        if right_is_standalone and left_makes_sense:
                            sub_candidates.extend([left, right])
                        else:
                            sub_candidates.append(cp)  # keep compound ("fluid input and output")
                    else:
                        sub_candidates.append(cp)

                for sub_candidate in sub_candidates:
                    # Strip trailing parenthetical qualifiers: "Heart Rate (ECG source)" -> "Heart Rate"
                    sub_candidate = re.sub(r"\s*\([^)]*\)\s*$", "", sub_candidate).strip()
                    # Strip leading "use of " / "use of the " (paper uses "use of vasopressors")
                    sub_candidate = re.sub(r"^use\s+of\s+(?:the\s+)?", "", sub_candidate, flags=re.IGNORECASE).strip()
                    if len(sub_candidate) >= 2 and sub_candidate.lower() != super_candidate.lower():
                        _add_edge(sub_candidate, super_candidate, sent)
                break  # only process first trigger per sentence

        # "Y is a type of X" / "Y is a kind of X"
        for trigger in (" is a type of ", " is a kind of "):
            idx = sent_lower.find(trigger)
            if idx >= 0:
                sub_phrase = sent[:idx].strip()
                super_phrase = sent[idx + len(trigger):].strip()
                super_phrase = re.sub(r"[.!?].*$", "", super_phrase).strip()
                if sub_phrase and super_phrase and len(sub_phrase) >= 2 and len(super_phrase) >= 2:
                    _add_edge(sub_phrase, super_phrase, sent)

        # "Y is a X" / "Y is an X" (short superClass to avoid full clauses, e.g. "ICP is a monitoring parameter")
        for trigger in (" is a ", " is an "):
            idx = sent_lower.find(trigger)
            if idx >= 0:
                sub_phrase = sent[:idx].strip()
                rest = sent[idx + len(trigger):].strip()
                rest = re.sub(r"[.!?].*$", "", rest).strip()
                super_words = rest.split()
                if len(super_words) <= 6 and len(super_words) >= 1:
                    super_phrase = " ".join(super_words)
                    if sub_phrase and len(sub_phrase) >= 2 and len(super_phrase) >= 2:
                        _add_edge(sub_phrase, super_phrase, sent)

    return _normalize_hierarchy_to_gold(edges)


def _normalize_hierarchy_to_gold(edges: List[Dict]) -> List[Dict]:
    """Map raw regex-extracted noun phrases to gold class labels via synonym map and fuzzy match."""
    gold_lower = {g.lower(): g for g in ALLOWED_CLASSES_CORE}
    syn_lower = {k.lower(): v for k, v in CLASS_SYNONYM_MAP.items() if v is not None}

    def _resolve(raw: str) -> str | None:
        s = raw.strip()
        if s.lower() in gold_lower:
            return gold_lower[s.lower()]
        if s in CLASS_SYNONYM_MAP:
            return CLASS_SYNONYM_MAP[s]
        if s.lower() in syn_lower:
            return syn_lower[s.lower()]
        raw_lower = s.lower()
        best_match = None
        best_len = 0
        for g_lower, g_label in gold_lower.items():
            if g_lower in raw_lower or raw_lower in g_lower:
                if len(g_lower) > best_len:
                    best_len = len(g_lower)
                    best_match = g_label
        return best_match

    out: List[Dict] = []
    seen: Set[tuple] = set()
    for e in edges:
        sub_resolved = _resolve(e["subClass"])
        sup_resolved = _resolve(e["superClass"])
        if not sub_resolved or not sup_resolved:
            continue
        if sub_resolved.lower() == sup_resolved.lower():
            continue
        key = (sub_resolved, sup_resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "subClass": sub_resolved,
            "superClass": sup_resolved,
            "evidence": e.get("evidence", ""),
        })
    return out
