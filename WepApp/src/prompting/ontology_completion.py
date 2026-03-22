"""Ontology completion: schema-driven expansion (per-chunk and whole-ontology).

Schema-guided completion (whole-ontology, optional post-merge step) logic:
1. Run extraction per chunk (phased or not) → merge to one ontology.
2. Normalise labels, dedupe; compare against schema/proxy gold:
   - missing_classes = gold_classes - predicted_classes (by normalised label)
   - missing_relations = gold_relations - predicted_relations (by normalised label, domain, range)
   - missing_hierarchy = gold_hierarchy - predicted_hierarchy (by normalised sub/super pair)
3. Ask LLM once: "From the corpus evidence, which missing items are supported?"
   Prompt includes the list of missing items and the full corpus text.
4. Add only items with: evidence span(s), schema membership, valid domain/range (for relations).

Whole-ontology completion: build_whole_ontology_completion_prompt, run_schema_guided_completion
(corpus-grounded; add only items with evidence + schema).

LLM Reasoning Layer (patch-based PROPOSE → VERIFY): run_llm_reasoning_layer_patch.
Two LLM calls to propose and verify schema-licensed hierarchy edges. Runs after cleanup
so the LLM operates on a clean ontology. No corpus evidence required; constrained to gold
schema hierarchy and existing class endpoints for safety.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable, Dict, List, Optional

from ..ontology.canonical import canonical_key, resolve_to_canonical_label


SGC_MAX_TOKENS = 8192
LLM_REASONING_MAX_TOKENS = 4096


def _generate_with_cancel_check(
    llm,
    prompt: str,
    progress_callback: Optional[Callable[[int, int, str], None]],
    progress_current: int,
    progress_total: int,
    message: str,
    *,
    max_tokens: int | None = None,
) -> str:
    """Run llm.generate(prompt) in a worker thread; in the caller thread poll progress_callback
    every second so cancel can be checked (e.g. RunCancelledError). Returns the generated text.
    Use for long-running LLM calls (e.g. Me-LLaMA, Llama-3-Meditron) so Cancel works instead of appearing stuck.
    """
    result: List[Optional[str]] = [None]
    exc: List[Optional[BaseException]] = [None]

    def worker() -> None:
        try:
            result[0] = llm.generate(prompt, max_tokens=max_tokens)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        if progress_callback is not None:
            progress_callback(progress_current, progress_total, message)
        time.sleep(1)
    if exc[0] is not None:
        raise exc[0]
    assert result[0] is not None
    return result[0]


def _canonical_norm(label: str) -> str:
    """Resolve label through canonical alias map, then compute canonical_key.
    This handles abbreviation-to-full-label matching (e.g., "ICP" -> "Intracranial Pressure (ICP)")."""
    resolved, _ = resolve_to_canonical_label(label)
    return canonical_key(resolved)


def _resolve_to_gold_norm(label: str, gold_class_norms: set) -> str | None:
    """Try _norm_label first, then canonical resolution, to match label against gold norms.
    gold_class_norms should include both _norm_label and _canonical_norm forms of each gold class."""
    norm = _norm_label(label)
    if norm in gold_class_norms:
        return norm
    resolved, _ = resolve_to_canonical_label(label)
    rnorm = _norm_label(resolved)
    if rnorm in gold_class_norms:
        return rnorm
    cn = _canonical_norm(label)
    if cn in gold_class_norms:
        return cn
    return None


def filter_completion_to_schema(
    completion_results: Dict,
    gold_schema: Dict
) -> Dict:
    """
    Filter completion results to only include classes and relations that exist in gold schema.

    Class matching is canonical-aware: abbreviations like "ICP" are resolved through
    the canonical alias map before matching against gold class labels.
    Relation label matching is alias-aware via RELATION_ALIASES_CORE.
    """
    from ..prompting.vocabulary import RELATION_ALIASES_CORE

    gold_class_norms = set()
    for c in gold_schema.get("classes", []):
        lbl = c.get("label", "")
        if lbl:
            gold_class_norms.add(_norm_label(lbl))
            gold_class_norms.add(_canonical_norm(lbl))

    gold_relation_labels_norm = {_norm_label(r.get("label", "")) for r in gold_schema.get("relations", []) if r.get("label")}
    alias_norms = {_norm_label(alias) for alias in RELATION_ALIASES_CORE}
    allowed_relation_norms = gold_relation_labels_norm | alias_norms

    gold_hierarchy_norms = set()
    for e in gold_schema.get("hierarchy", []):
        sub, sup = e.get("subClass", ""), e.get("superClass", "")
        if sub and sup:
            gold_hierarchy_norms.add((_norm_label(sub), _norm_label(sup)))
            gold_hierarchy_norms.add((_canonical_norm(sub), _canonical_norm(sup)))

    filtered_classes = []
    for cls in completion_results.get("classes", []):
        label = cls.get("label", "")
        if _resolve_to_gold_norm(label, gold_class_norms) is not None:
            filtered_classes.append(cls)

    filtered_relations = []
    for rel in completion_results.get("relations", []):
        label_norm = _norm_label(rel.get("label") or "")
        domain = rel.get("domain") or ""
        range_ = rel.get("range") or ""
        domain_ok = not domain or _resolve_to_gold_norm(domain, gold_class_norms) is not None
        range_ok = not range_ or _resolve_to_gold_norm(range_, gold_class_norms) is not None
        if label_norm in allowed_relation_norms and domain_ok and range_ok:
            filtered_relations.append(rel)

    filtered_hierarchy = []
    for edge in completion_results.get("hierarchy", []):
        sub = edge.get("subClass") or ""
        sup = edge.get("superClass") or ""
        if not sub or not sup:
            continue
        sub_norm = _resolve_to_gold_norm(sub, gold_class_norms)
        sup_norm = _resolve_to_gold_norm(sup, gold_class_norms)
        if sub_norm is None or sup_norm is None:
            continue
        if (sub_norm, sup_norm) in gold_hierarchy_norms:
            filtered_hierarchy.append(edge)
            continue
        sub_cn = _canonical_norm(sub)
        sup_cn = _canonical_norm(sup)
        if (sub_cn, sup_cn) in gold_hierarchy_norms:
            filtered_hierarchy.append(edge)

    return {
        "classes": filtered_classes,
        "relations": filtered_relations,
        "hierarchy": filtered_hierarchy,
    }


def _norm_label(s: str) -> str:
    """Normalise label for set comparison: strip, lower, collapse non-alphanumeric."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def build_whole_ontology_completion_prompt(
    ontology_dict: Dict,
    gold_schema: Dict,
    corpus_text: str,
) -> str:
    """
    Build prompt for schema-guided completion over the full merged ontology.
    Logic: missing = gold - predicted (normalised); ask LLM once which missing items
    are supported by corpus evidence; require evidence span(s) in the output.
    """
    current_classes = ontology_dict.get("classes", [])
    current_relations = ontology_dict.get("relations", [])

    # Normalise for comparison so we dedupe / map synonyms consistently
    predicted_class_norm = {_norm_label(c.get("label", "")) for c in current_classes if c.get("label")}
    predicted_relation_norm = {
        (_norm_label(r.get("label")), _norm_label(r.get("domain")), _norm_label(r.get("range")))
        for r in current_relations
        if r.get("label")
    }

    gold_classes = gold_schema.get("classes", [])
    gold_relations = gold_schema.get("relations", [])
    gold_hierarchy = gold_schema.get("hierarchy", [])

    missing_classes = [
        c for c in gold_classes
        if c.get("label") and _norm_label(c.get("label", "")) not in predicted_class_norm
    ]
    missing_relations = [
        r for r in gold_relations
        if r.get("label")
        and (
            _norm_label(r.get("label")),
            _norm_label(r.get("domain")),
            _norm_label(r.get("range")),
        ) not in predicted_relation_norm
    ]

    current_hierarchy = ontology_dict.get("hierarchy", [])
    predicted_hierarchy_norm = {
        (_norm_label(e.get("subClass", "")), _norm_label(e.get("superClass", "")))
        for e in current_hierarchy
        if e.get("subClass") and e.get("superClass")
    }
    missing_hierarchy = [
        e for e in gold_hierarchy
        if e.get("subClass") and e.get("superClass")
        and (_norm_label(e.get("subClass", "")), _norm_label(e.get("superClass", "")))
        not in predicted_hierarchy_norm
    ]

    if not missing_classes and not missing_relations and not missing_hierarchy:
        return ""

    missing_section = "## Missing from current ontology (in gold schema but not yet predicted):\n\n"
    if missing_classes:
        missing_section += "### Missing classes:\n"
        for c in missing_classes:
            missing_section += f"- {c.get('label', '')}" + (f" ({c.get('definition', '')})" if c.get("definition") else "") + "\n"
        missing_section += "\n"
    if missing_relations:
        missing_section += "### Missing relations:\n"
        for r in missing_relations:
            missing_section += f"- {r.get('label', '')}({r.get('domain', '')} -> {r.get('range', '')})\n"
        missing_section += "\n"
    if missing_hierarchy:
        missing_section += "### Missing hierarchy edges (subClass -> superClass):\n"
        for e in missing_hierarchy:
            missing_section += f"- {e.get('subClass', '')} -> {e.get('superClass', '')}\n"
        missing_section += "\n"

    corpus_section = "## Corpus (evidence source):\n\n" + (corpus_text.strip() or "(no text)") + "\n\n"

    # Concrete few-shot example anchors the expected output format and demonstrates
    # that classes like "Fluids", "Nutrition", "Sedation" ARE worth extracting from brief mentions.
    few_shot_example = (
        "## EXAMPLE (do not copy — for format guidance only):\n"
        "Missing classes: Fluids, Nutrition, Sedation\n"
        "Corpus snippet: \"...ventilation settings, sedation levels, fluid input and output, nutrition...\"\n"
        "Correct output:\n"
        "{\n"
        "  \"classes\": [\n"
        "    {\"label\": \"Fluids\", \"evidence\": \"fluid input and output\"},\n"
        "    {\"label\": \"Nutrition\", \"evidence\": \"nutrition\"},\n"
        "    {\"label\": \"Sedation\", \"evidence\": \"sedation levels\"}\n"
        "  ],\n"
        "  \"relations\": [],\n"
        "  \"hierarchy\": []\n"
        "}\n"
        "Note: even a brief mention (1-3 words) is sufficient evidence — do NOT require full sentences.\n\n"
    )

    instruction = (
        "You are performing schema-guided completion. The current ontology was merged from extraction; "
        "the missing items below are gold schema items NOT yet in the ontology.\n\n"
        "Task: From the corpus evidence below, which of these missing items are SUPPORTED by the text?\n"
        "1. Consider ONLY the missing classes, relations, and hierarchy edges listed.\n"
        "2. For each item, provide an evidence span: an EXACT quote or short phrase copied from the corpus text below.\n"
        "   IMPORTANT: Evidence must be actual text from the corpus, NOT ontology URIs (e.g. 'pd:Session'), "
        "NOT the label repeated, and NOT definitions. Copy a real fragment from the corpus.\n"
        "3. A brief mention (even 1-3 words) in the corpus IS sufficient evidence. Do not require a full sentence.\n"
        "4. You MUST output as many matching items as the corpus supports — do not leave known classes out.\n"
        "5. Use exact labels from the gold schema (same spelling and capitalisation).\n"
        "6. For relations, domain and range must be gold class labels or already in the ontology.\n"
        "7. For hierarchy edges, output only edges where the text implies an is-a relationship "
        "(e.g. 'X such as Y', 'types of X include Y', 'Y is a form of X'). "
        "Both subClass and superClass must be classes in the current ontology or in your classes output.\n\n"
        "Output valid JSON only with keys: classes, relations, hierarchy.\n\n"
    )

    return instruction + few_shot_example + missing_section + corpus_section + """Output JSON format:
{
  "classes": [{"label": "...", "evidence": "exact span from corpus"}],
  "relations": [{"label": "...", "domain": "...", "range": "...", "evidence": "exact span from corpus"}],
  "hierarchy": [{"subClass": "...", "superClass": "...", "evidence": "span or phrase from corpus that implies the is-a relationship"}]
}

Include every missing item that has ANY support in the corpus. Your answer:"""


def run_schema_guided_completion(
    ontology,
    gold_schema: Dict,
    llm,
    corpus_text: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    run_dir: Optional[str] = None,
) -> Dict:
    """
    Run schema-guided completion on the full merged ontology and merge results in place.
    Logic: normalise & compare (missing = gold - predicted); ask LLM once which missing
    items are supported by corpus evidence; add only items with evidence + schema membership + valid domain/range.
    Returns dict with classes_added, relations_added, hierarchy_added for ablation reporting.
    """
    from ..ontology.export import ontology_to_dict
    from ..ontology.model import ClassEntity, RelationEntity
    from .parse import parse_output
    import os

    ontology_dict = ontology_to_dict(ontology)
    prompt = build_whole_ontology_completion_prompt(ontology_dict, gold_schema, corpus_text or "")
    if not prompt:
        return {"classes_added": 0, "relations_added": 0, "hierarchy_added": 0}

    raw = _generate_with_cancel_check(
        llm, prompt, progress_callback, 0, 1, "Schema-guided completion: calling LLM…",
        max_tokens=SGC_MAX_TOKENS,
    )

    # Save SGC prompt and raw response for debugging (optional, when run_dir is provided)
    if run_dir:
        try:
            prompts_dir = os.path.join(run_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            with open(os.path.join(prompts_dir, "sgc_prompt.txt"), "w", encoding="utf-8") as fh:
                fh.write(prompt)
            with open(os.path.join(prompts_dir, "sgc_response.txt"), "w", encoding="utf-8") as fh:
                fh.write(raw or "")
        except Exception:
            pass

    # SGC evidence is often short but legitimate (e.g. "EtCO2", "TCD") — use lenient threshold
    parsed = parse_output(raw, min_evidence_length=1)
    filtered = filter_completion_to_schema(parsed, gold_schema)

    # Diagnostic: log parsing pipeline counts for debugging SGC yield
    _sgc_diag = {
        "raw_length": len(raw or ""),
        "parsed_classes": len(parsed.get("classes", [])),
        "parsed_relations": len(parsed.get("relations", [])),
        "parsed_hierarchy": len(parsed.get("hierarchy", [])),
        "filtered_classes": len(filtered.get("classes", [])),
        "filtered_relations": len(filtered.get("relations", [])),
        "filtered_hierarchy": len(filtered.get("hierarchy", [])),
    }
    if run_dir:
        try:
            import json as _json
            with open(os.path.join(run_dir, "prompts", "sgc_diagnostic.json"), "w", encoding="utf-8") as fh:
                _json.dump(_sgc_diag, fh, indent=2)
        except Exception:
            pass

    gold_class_labels_norm = {_norm_label(c.get("label", "")): (c.get("label", "").strip()) for c in gold_schema.get("classes", []) if c.get("label")}
    existing_classes_norm = {_norm_label(c.label) for c in ontology.classes}
    existing_relations_norm = {
        (_norm_label(r.label or ""), _norm_label((r.domain or "")), _norm_label((r.range or "")))
        for r in ontology.relations
    }
    existing_hierarchy_norm = {
        (_norm_label(str(e.get("subClass", ""))), _norm_label(str(e.get("superClass", ""))))
        for e in ontology.hierarchy
    }

    valid_class_labels = set(gold_class_labels_norm.values()) | {c.label for c in ontology.classes}
    valid_class_norm = {_norm_label(l) for l in valid_class_labels}

    provenance = ["schema_guided_completion"]
    classes_added = 0
    relations_added = 0
    hierarchy_added = 0

    for cls in filtered.get("classes", []):
        label = (cls.get("label") or "").strip()
        evidence = (cls.get("evidence") or "").strip()
        if not label or not evidence:
            continue
        key = _norm_label(label)
        if key in existing_classes_norm:
            continue
        if key not in gold_class_labels_norm:
            resolved, _ = resolve_to_canonical_label(label)
            rkey = _norm_label(resolved)
            if rkey in gold_class_labels_norm and rkey not in existing_classes_norm:
                key = rkey
            else:
                continue
        canonical = gold_class_labels_norm.get(key, label)
        existing_classes_norm.add(key)
        valid_class_norm.add(key)
        ontology.add_class(
            ClassEntity(
                label=canonical,
                definition=cls.get("definition"),
                synonyms=cls.get("synonyms", []),
                provenance=provenance,
                stratum="schema_guided",
                evidence=cls.get("evidence"),
                aliases=list(cls.get("aliases") or []),
            )
        )
        classes_added += 1

    for rel in filtered.get("relations", []):
        rlabel = (rel.get("label") or "").strip()
        domain = (rel.get("domain") or "").strip()
        range_ = (rel.get("range") or "").strip()
        evidence = (rel.get("evidence") or "").strip()
        if not rlabel or not evidence:
            continue
        if not domain or not range_:
            continue
        if _norm_label(domain) not in valid_class_norm or _norm_label(range_) not in valid_class_norm:
            continue
        norm_key = (_norm_label(rlabel), _norm_label(domain), _norm_label(range_))
        if norm_key in existing_relations_norm:
            continue
        existing_relations_norm.add(norm_key)
        ontology.add_relation(
            RelationEntity(
                label=rlabel,
                domain=domain,
                range=range_,
                definition=rel.get("definition"),
                provenance=provenance,
                stratum="schema_guided",
                evidence=evidence,
                aliases=list(rel.get("aliases") or []),
            )
        )
        relations_added += 1

    for edge in filtered.get("hierarchy", []):
        sub = (edge.get("subClass") or "").strip()
        sup = (edge.get("superClass") or "").strip()
        if not sub or not sup:
            continue
        norm_pair = (_norm_label(sub), _norm_label(sup))
        if norm_pair in existing_hierarchy_norm:
            continue
        if _norm_label(sub) not in valid_class_norm or _norm_label(sup) not in valid_class_norm:
            continue
        existing_hierarchy_norm.add(norm_pair)
        llm_evidence = (edge.get("evidence") or "").strip()
        evidence = llm_evidence if llm_evidence else f"Schema-inferred: {sub} is a subclass of {sup}."
        ontology.hierarchy.append({
            "subClass": sub,
            "superClass": sup,
            "evidence": evidence,
            "provenance": provenance,
            "stratum": "schema_guided",
        })
        hierarchy_added += 1

    return {"classes_added": classes_added, "relations_added": relations_added, "hierarchy_added": hierarchy_added}


# --- Integrated LLM Reasoning Phase (Option B) ---


def build_llm_reasoning_propose_prompt(
    current_class_labels: List[str],
    candidate_edges: List[Dict],
) -> str:
    """
    Propose step: suggest hierarchy edges using indices only.
    candidate_edges items: {sub_idx, super_idx, sub_label, super_label}.
    """
    cls_lines = "\n".join([f"{i}: {lbl}" for i, lbl in enumerate(current_class_labels)])
    edge_lines = "\n".join(
        [f"- ({e['sub_idx']} -> {e['super_idx']}) {e['sub_label']} ⊑ {e['super_label']}" for e in candidate_edges]
    )
    return (
        "You are the LLM Reasoning Layer (PROPOSE step).\n"
        "Closed world: you may ONLY refer to classes by index from CURRENT_CLASSES.\n"
        "You may ONLY propose edges that appear in CANDIDATE_EDGES.\n"
        "Do NOT add classes or relations. Precision > recall.\n"
        "Return valid JSON only.\n\n"
        "CURRENT_CLASSES (index: label):\n"
        f"{cls_lines}\n\n"
        "CANDIDATE_EDGES (schema-licensed, endpoints exist):\n"
        f"{edge_lines}\n\n"
        "Task:\n"
        "- Choose up to 30 edges that are missing and improve hierarchy structure.\n\n"
        "Output JSON format:\n"
        "{\n"
        '  "edits": {\n'
        '    "add_hierarchy": [{"sub_idx": 0, "super_idx": 1, "justification": "brief", "confidence": 0.0}],\n'
        '    "remove_hierarchy": [],\n'
        '    "rename_classes": [],\n'
        '    "merge_classes": [],\n'
        '    "add_classes": [],\n'
        '    "remove_classes": [],\n'
        '    "add_relations": [],\n'
        '    "remove_relations": []\n'
        "  },\n"
        '  "flags": [],\n'
        '  "notes": []\n'
        "}\n"
    )


def build_llm_reasoning_verify_prompt(
    current_class_labels: List[str],
    candidate_edges: List[Dict],
    proposed: Dict,
    existing_edges: List[Dict],
) -> str:
    """
    Verify step: filter the proposed patch strictly to candidate edges, keep only valid index pairs.
    """
    cls_lines = "\n".join([f"{i}: {lbl}" for i, lbl in enumerate(current_class_labels)])
    edge_set_lines = "\n".join([f"- ({e['sub_idx']} -> {e['super_idx']})" for e in candidate_edges])
    existing_lines = "\n".join([f"- {e.get('subClass','')} ⊑ {e.get('superClass','')}" for e in existing_edges]) or "(none)"
    proposed_json = json.dumps(proposed, indent=2)

    return (
        "You are the LLM Reasoning Layer (VERIFY step / critic).\n"
        "Your job is to output a FINAL patch that is strictly valid.\n\n"
        "Hard rules:\n"
        "1) Keep ONLY add_hierarchy edges where (sub_idx, super_idx) appears in CANDIDATE_EDGE_INDEX_PAIRS.\n"
        "2) Remove any duplicates.\n"
        "3) Do NOT include edges already present in EXISTING_HIERARCHY.\n"
        "4) Do NOT add classes/relations. Leave those arrays empty.\n"
        "5) If confidence is missing, set it to 0.7 (minimum).\n"
        "6) Return valid JSON only.\n\n"
        "CURRENT_CLASSES:\n"
        f"{cls_lines}\n\n"
        "CANDIDATE_EDGE_INDEX_PAIRS:\n"
        f"{edge_set_lines}\n\n"
        "EXISTING_HIERARCHY:\n"
        f"{existing_lines}\n\n"
        "PROPOSED_PATCH:\n"
        f"{proposed_json}\n\n"
        "Output JSON format (same as PROPOSE, using sub_idx/super_idx):\n"
        "{\n"
        '  "edits": {\n'
        '    "add_hierarchy": [{"sub_idx": 0, "super_idx": 1, "justification": "brief", "confidence": 0.0}],\n'
        '    "remove_hierarchy": [],\n'
        '    "rename_classes": [],\n'
        '    "merge_classes": [],\n'
        '    "add_classes": [],\n'
        '    "remove_classes": [],\n'
        '    "add_relations": [],\n'
        '    "remove_relations": []\n'
        "  },\n"
        '  "flags": [],\n'
        '  "notes": []\n'
        "}\n"
    )


def _try_parse_json_any(raw: str):
    try:
        from .parse import safe_json_loads
        return safe_json_loads(raw)
    except Exception:
        return None


def _try_parse_embedded_json_any(raw: str):
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        from .parse import safe_json_loads
        return safe_json_loads(snippet)
    except Exception:
        return None


def parse_patch_output(raw: str) -> Dict:
    parsed = _try_parse_json_any(raw)
    if parsed is None:
        parsed = _try_parse_embedded_json_any(raw)
    if parsed is None:
        from .parse import _try_recover_truncated_json
        parsed = _try_recover_truncated_json(raw or "")
    if not isinstance(parsed, dict):
        parsed = {}
    edits = parsed.get("edits") if isinstance(parsed.get("edits"), dict) else {}
    return {
        "edits": edits,
        "flags": parsed.get("flags") if isinstance(parsed.get("flags"), list) else [],
        "notes": parsed.get("notes") if isinstance(parsed.get("notes"), list) else [],
    }


def _extract_add_hierarchy_idx(patch: Dict) -> List[Dict]:
    edits = patch.get("edits") if isinstance(patch.get("edits"), dict) else {}
    items = edits.get("add_hierarchy") if isinstance(edits.get("add_hierarchy"), list) else []
    out: List[Dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sub_idx = it.get("sub_idx")
        super_idx = it.get("super_idx")
        if not isinstance(sub_idx, int) or not isinstance(super_idx, int):
            continue
        out.append(
            {
                "sub_idx": sub_idx,
                "super_idx": super_idx,
                "justification": (it.get("justification") or "").strip(),
                "confidence": it.get("confidence"),
            }
        )
    return out


def run_llm_reasoning_layer_patch(
    ontology,
    gold_schema: Dict,
    llm,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    run_dir: Optional[str] = None,
) -> tuple[Dict, Dict, Dict, Dict]:
    """
    LLM Reasoning Layer (patch-based):
    - two LLM calls: PROPOSE -> VERIFY
    - patch-only, schema-licensed hierarchy completion + optional flags
    Returns (counts, final_patch, proposed_patch, verified_patch).
    """
    from ..ontology.export import ontology_to_dict
    import os

    ontology_dict = ontology_to_dict(ontology)
    current_labels: List[str] = [c.get("label", "").strip() for c in ontology_dict.get("classes", []) if c.get("label")]

    # Use canonical normalization (alias-aware) so gold abbreviations like "ICP" match ontology
    # full labels like "Intracranial Pressure (ICP)". _norm_label alone is insufficient here.
    def _ckey(s: str) -> str:
        return canonical_key(resolve_to_canonical_label(s)[0]) if s else ""

    canon_to_idx: Dict[str, int] = {}
    for i, lbl in enumerate(current_labels):
        k = _ckey(lbl)
        if k and k not in canon_to_idx:
            canon_to_idx[k] = i
    # Also populate a _norm_label-based fallback for labels without a canonical alias
    norm_to_idx_fallback = {_norm_label(lbl): i for i, lbl in enumerate(current_labels)}

    # Candidate edges: only schema hierarchy edges whose endpoints exist in current classes
    gold_hierarchy = gold_schema.get("hierarchy", []) or []
    candidates: List[Dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    for e in gold_hierarchy:
        if not isinstance(e, dict):
            continue
        sub = (e.get("subClass") or "").strip()
        sup = (e.get("superClass") or "").strip()
        if not sub or not sup:
            continue
        sub_i = canon_to_idx.get(_ckey(sub)) if _ckey(sub) else None
        if sub_i is None:
            sub_i = norm_to_idx_fallback.get(_norm_label(sub))
        sup_i = canon_to_idx.get(_ckey(sup)) if _ckey(sup) else None
        if sup_i is None:
            sup_i = norm_to_idx_fallback.get(_norm_label(sup))
        if sub_i is None or sup_i is None:
            continue
        key = (sub_i, sup_i)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        candidates.append(
            {"sub_idx": sub_i, "super_idx": sup_i, "sub_label": current_labels[sub_i], "super_label": current_labels[sup_i]}
        )

    # If there are no candidates, short-circuit but still return empty patches for artefacts.
    if not candidates:
        n_gold = len([x for x in gold_hierarchy if isinstance(x, dict) and (x.get("subClass") or "").strip() and (x.get("superClass") or "").strip()])
        notes = [
            "No schema-licensed candidate edges with existing endpoints.",
            f"Gold schema has {n_gold} hierarchy edge(s); current ontology has {len(current_labels)} class(es). "
            "Hierarchy is only added when both subclass and superclass (after label normalization) exist in current classes. "
            "If extraction uses different class names than the gold TTL/JSON, no candidates are produced and the reasoning layer adds nothing.",
        ]
        empty = {"edits": {"add_hierarchy": [], "remove_hierarchy": [], "rename_classes": [], "merge_classes": [], "add_classes": [], "remove_classes": [], "add_relations": [], "remove_relations": []}, "flags": [], "notes": notes}
        counts = {"classes_inferred": 0, "relations_inferred": 0, "hierarchy_inferred": 0}
        return counts, empty, empty, empty

    # PROPOSE
    propose_prompt = build_llm_reasoning_propose_prompt(current_labels, candidates)
    proposed_raw = _generate_with_cancel_check(
        llm, propose_prompt, progress_callback, 0, 2, "LLM Reasoning: PROPOSE step…",
        max_tokens=LLM_REASONING_MAX_TOKENS,
    )
    proposed_patch = parse_patch_output(proposed_raw)

    # VERIFY
    existing_edges = list(ontology_dict.get("hierarchy", []) or [])
    verify_prompt = build_llm_reasoning_verify_prompt(current_labels, candidates, proposed_patch, existing_edges)
    verified_raw = _generate_with_cancel_check(
        llm, verify_prompt, progress_callback, 1, 2, "LLM Reasoning: VERIFY step…",
        max_tokens=LLM_REASONING_MAX_TOKENS,
    )
    verified_patch = parse_patch_output(verified_raw)

    if run_dir:
        try:
            prompts_dir = os.path.join(run_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            with open(os.path.join(prompts_dir, "llm_reasoning_propose_prompt.txt"), "w", encoding="utf-8") as fh:
                fh.write(propose_prompt)
            with open(os.path.join(prompts_dir, "llm_reasoning_propose_response.txt"), "w", encoding="utf-8") as fh:
                fh.write(proposed_raw or "")
            with open(os.path.join(prompts_dir, "llm_reasoning_verify_prompt.txt"), "w", encoding="utf-8") as fh:
                fh.write(verify_prompt)
            with open(os.path.join(prompts_dir, "llm_reasoning_verify_response.txt"), "w", encoding="utf-8") as fh:
                fh.write(verified_raw or "")
        except Exception:
            pass

    # Apply safe verified edits
    candidate_pair_set = {(c["sub_idx"], c["super_idx"]) for c in candidates}
    existing_norm = {
        (_norm_label(e.get("subClass", "")), _norm_label(e.get("superClass", "")))
        for e in ontology.hierarchy
        if e.get("subClass") and e.get("superClass")
    }

    hierarchy_inferred = 0
    max_add = 30
    min_conf = 0.7
    add_edges_idx = _extract_add_hierarchy_idx(verified_patch)[:max_add]
    for edge in add_edges_idx:
        pair = (edge["sub_idx"], edge["super_idx"])
        if pair not in candidate_pair_set:
            continue
        conf = edge.get("confidence")
        if conf is None:
            conf = min_conf
        if isinstance(conf, (int, float)) and conf < min_conf:
            continue
        sub_label = current_labels[edge["sub_idx"]]
        sup_label = current_labels[edge["super_idx"]]
        sub_n = _norm_label(sub_label)
        sup_n = _norm_label(sup_label)
        if (sub_n, sup_n) in existing_norm:
            continue
        ontology.hierarchy.append(
            {
                "subClass": sub_label,
                "superClass": sup_label,
                # Synthetic evidence satisfies require_evidence=True in filter_parsed_to_vocabulary
                # (called during eval_restrict_to_gold) and contains "is a" — a HIERARCHY_LEXICAL_TRIGGER.
                "evidence": f"Schema-inferred: {sub_label} is a subclass of {sup_label}.",
                "provenance": ["llm_reasoning_layer_patch"],
                "stratum": "llm_reasoning",
                "justification": edge.get("justification") or None,
                "confidence": conf,
            }
        )
        existing_norm.add((sub_n, sup_n))
        hierarchy_inferred += 1

    final_patch = {
        "edits": {
            "add_hierarchy": [
                {
                    "subClass": current_labels[e["sub_idx"]],
                    "superClass": current_labels[e["super_idx"]],
                    "justification": e.get("justification") or "",
                    "confidence": e.get("confidence") if isinstance(e.get("confidence"), (int, float)) else min_conf,
                }
                for e in add_edges_idx
                if (e["sub_idx"], e["super_idx"]) in candidate_pair_set
            ],
            "remove_hierarchy": [],
            "rename_classes": [],
            "merge_classes": [],
            "add_classes": [],
            "remove_classes": [],
            "add_relations": [],
            "remove_relations": [],
        },
        "flags": (verified_patch.get("flags") or []) if isinstance(verified_patch.get("flags"), list) else [],
        "notes": (verified_patch.get("notes") or []) if isinstance(verified_patch.get("notes"), list) else [],
    }

    counts = {"classes_inferred": 0, "relations_inferred": 0, "hierarchy_inferred": hierarchy_inferred}
    return counts, final_patch, proposed_patch, verified_patch


# ─────────────────────────────────────────────────────────────────────────────
# TEXT-GROUNDED COMPLETION (built-in, always-on)
# Two-pass architecture: Pass 1 = classes + relations, Pass 2 = hierarchy
# ─────────────────────────────────────────────────────────────────────────────

TEXT_GROUNDED_MAX_TOKENS = 8192

# Max characters of corpus text to include in TGC / CoL prompts.
# GPT-4o free-tier has a 30k TPM limit — ~4 chars/token means we need
# to keep the entire prompt (instructions + ontology listing + text)
# well under ~120k chars.  20k chars of text ≈ 5k tokens leaves room
# for the rest of the prompt and the response.
_TGC_MAX_TEXT_CHARS = int(os.environ.get("TGC_MAX_TEXT_CHARS", "20000"))

_EVIDENCE_DUPLICATE_THRESHOLD = 3
_EVIDENCE_MIN_MATCH_RATIO = 0.55


def _evidence_appears_in_text(evidence: str, corpus_text: str) -> bool:
    """Verify that an evidence string genuinely appears in the corpus.

    Uses progressive matching:
    1. Exact substring match (after whitespace normalisation).
    2. Word-overlap ratio — the fraction of evidence words found in the corpus
       must meet _EVIDENCE_MIN_MATCH_RATIO.
    """
    if not evidence or not corpus_text:
        return False
    norm_ev = re.sub(r"\s+", " ", evidence.strip().lower())
    norm_ct = re.sub(r"\s+", " ", corpus_text.lower())
    if norm_ev in norm_ct:
        return True
    ev_words = set(re.findall(r"[a-z0-9]+", norm_ev))
    if not ev_words:
        return False
    ct_words = set(re.findall(r"[a-z0-9]+", norm_ct))
    overlap = len(ev_words & ct_words) / len(ev_words)
    return overlap >= _EVIDENCE_MIN_MATCH_RATIO


def _filter_bulk_fabricated(items: List[Dict], evidence_key: str = "evidence") -> List[Dict]:
    """Remove items that share an identical evidence string with too many others
    (a strong signal the LLM copy-pasted one phrase for everything)."""
    from collections import Counter
    ev_counts: Counter = Counter()
    for item in items:
        ev = (item.get(evidence_key) or "").strip().lower()
        if ev:
            ev_counts[ev] += 1
    return [
        item for item in items
        if ev_counts.get((item.get(evidence_key) or "").strip().lower(), 0) < _EVIDENCE_DUPLICATE_THRESHOLD
    ]


_HIERARCHY_EDGE_CHECKLIST: List[tuple[str, str]] = [
    ("Core Monitoring Parameter", "Monitoring Data"),
    ("Optional Monitoring Parameter", "Monitoring Data"),
    ("Derived Parameter", "Monitoring Data"),
    ("Mean Arterial Pressure (MAP)", "Core Monitoring Parameter"),
    ("Intracranial Pressure (ICP)", "Core Monitoring Parameter"),
    ("Cerebral Perfusion Pressure (CPP)", "Core Monitoring Parameter"),
    ("Heart Rate", "Core Monitoring Parameter"),
    ("SaO2", "Core Monitoring Parameter"),
    ("Temperature", "Core Monitoring Parameter"),
    ("Respiration Rate", "Core Monitoring Parameter"),
    ("CVP", "Optional Monitoring Parameter"),
    ("EtCO2", "Optional Monitoring Parameter"),
    ("NIBP", "Optional Monitoring Parameter"),
    ("Peripheral Temperature", "Optional Monitoring Parameter"),
    ("PbrO2", "Optional Monitoring Parameter"),
    ("SjO2", "Optional Monitoring Parameter"),
    ("Cardiac Output", "Optional Monitoring Parameter"),
    ("Brain Temperature", "Optional Monitoring Parameter"),
    ("TCD", "Optional Monitoring Parameter"),
    ("Microdialysis", "Optional Monitoring Parameter"),
    ("Cerebral Perfusion Pressure (CPP)", "Derived Parameter"),
    ("Pressure Reactivity Index (PRx)", "Derived Parameter"),
    ("Baseline Therapy", "Therapy"),
    ("Secondary Insult Therapy", "Therapy"),
    ("Ventilation", "Baseline Therapy"),
    ("Sedation", "Baseline Therapy"),
    ("Analgesia", "Baseline Therapy"),
    ("Paralysis", "Baseline Therapy"),
    ("Fluids", "Baseline Therapy"),
    ("Nutrition", "Baseline Therapy"),
    ("Vasopressors", "Baseline Therapy"),
    ("Antibiotics", "Baseline Therapy"),
    ("Anti-hypertensives", "Baseline Therapy"),
    ("Anti-pyretics", "Baseline Therapy"),
    ("Hypothermia Therapy", "Baseline Therapy"),
    ("Noradrenaline", "Baseline Therapy"),
    ("Adrenaline", "Baseline Therapy"),
    ("Arterial Pressors", "Baseline Therapy"),
    ("Osmotics", "Secondary Insult Therapy"),
    ("Barbiturates", "Secondary Insult Therapy"),
    ("Steroids", "Secondary Insult Therapy"),
    ("ICP Sensor Placement", "Surgical Procedure"),
    ("Evacuation of Mass Lesion", "Surgical Procedure"),
    ("Skull Fracture Elevation", "Surgical Procedure"),
    ("Extra Ventricular Drain Placement", "Surgical Procedure"),
    ("Decompressive Craniectomy", "Surgical Procedure"),
    ("Removal of Foreign Body", "Surgical Procedure"),
    ("Anterior Fossa Repair", "Surgical Procedure"),
    ("GCS Assessment", "Clinical Assessment"),
    ("Pupil Assessment", "Clinical Assessment"),
    ("CT Scan Assessment", "Clinical Assessment"),
    ("Secondary Insult", "Condition"),
    ("Arterial Hypotension", "Secondary Insult"),
    ("Intracranial Hypertension", "Secondary Insult"),
    ("Systemic Hypotension", "Secondary Insult"),
    ("Jugular Venous Desaturation", "Secondary Insult"),
    ("GOSe Outcome", "Outcome"),
    ("Blood Gases", "Laboratory Values"),
    ("Glucose", "Laboratory Values"),
    ("Haematocrit", "Laboratory Values"),
    ("Biochemistry", "Laboratory Values"),
    ("Haematology", "Laboratory Values"),
    ("Sodium", "Laboratory Values"),
    ("Potassium", "Laboratory Values"),
    ("Haemoglobin", "Laboratory Values"),
    ("White Blood Cell Count", "Laboratory Values"),
    ("Routine Nursing Care", "Nursing Intervention"),
    ("Physiotherapy", "Nursing Intervention"),
    ("Bedside Intervention", "Nursing Intervention"),
    ("Patient Transport", "Nursing Intervention"),
    # v3.1 — papers 3-11
    ("Systolic Blood Pressure (SBP)", "Core Monitoring Parameter"),
    ("Diastolic Blood Pressure (DBP)", "Core Monitoring Parameter"),
    ("Electrocardiogram (ECG)", "Core Monitoring Parameter"),
    ("Pulse Pressure", "Derived Parameter"),
    ("Pressure-Time Dose", "Derived Parameter"),
    ("Hemicraniectomy", "Decompressive Craniectomy"),
    ("Bifrontal Craniectomy", "Decompressive Craniectomy"),
    ("Cranioplasty", "Surgical Procedure"),
    ("Head Elevation", "Baseline Therapy"),
    ("Hypotensive Event", "Secondary Insult"),
    ("Artefact", "Data Quality Assessment"),
    ("Mortality", "Outcome"),
    ("EUSIG Grade", "Clinical Assessment"),
    ("Cerebrovascular Autoregulation", "Condition"),
    ("Therapy Intensity Level (TIL)", "Clinical Assessment"),
    ("Guideline Adherence", "Clinical Assessment"),
    ("Injury Mechanism", "Demographic Data"),
]


def _build_classes_relations_prompt(
    current_classes: List[str],
    current_relations: List[Dict],
    corpus_text: str,
) -> str:
    """Pass 1 prompt: extract missing classes and relations from text (no vocabulary whitelist).

    Includes a 'dangling classes' section that identifies classes with no relations,
    directing the LLM to search for their missing connections specifically.
    """
    cls_list = "\n".join(f"  - {c}" for c in sorted(current_classes)) if current_classes else "  (none)"
    rel_list = "\n".join(
        f"  - {r.get('label','?')}: {r.get('domain','?')} -> {r.get('range','?')}"
        for r in current_relations
    ) if current_relations else "  (none)"

    connected: set[str] = set()
    for r in current_relations:
        dom = (r.get("domain") or "").strip()
        rng = (r.get("range") or "").strip()
        if dom:
            connected.add(dom)
        if rng:
            connected.add(rng)
    dangling = sorted(c for c in current_classes if c not in connected)

    dangling_section = ""
    if dangling:
        dangling_list = ", ".join(dangling)
        dangling_section = (
            "### DANGLING CLASSES (priority for relation search)\n"
            "The following classes have NO relations yet — they are isolated nodes. "
            "Search the source text specifically for relationships involving these classes. "
            "If the text describes how they relate to other clinical concepts, extract those relations.\n"
            f"Dangling: {dangling_list}\n\n"
        )

    text_truncated = corpus_text[:_TGC_MAX_TEXT_CHARS] if len(corpus_text) > _TGC_MAX_TEXT_CHARS else corpus_text

    return (
        "### TASK\n"
        "You are a text-grounded ontology completion engine for biomedical and "
        "clinical literature.\n"
        "Review the SOURCE TEXT below and find **clinical classes** and **relations** "
        "that are explicitly mentioned but MISSING from the current ontology.\n"
        "The per-chunk extraction may miss concepts — be thorough. "
        "Extract every clinical concept and relation the text explicitly names.\n\n"
        "### CURRENT ONTOLOGY\n"
        f"Classes already extracted:\n{cls_list}\n\n"
        f"Relations already extracted:\n{rel_list}\n\n"
        + dangling_section +
        "### WHAT TO EXTRACT\n"
        "Extract any clinical concept from the text that is NOT yet in the ontology above:\n"
        "- Clinical conditions, diseases, injuries, syndromes, secondary insults\n"
        "- Physiological parameters and monitoring variables (e.g. ICP, MAP, HR, SpO2, EtCO2)\n"
        "- Treatments, therapies, interventions, drugs, medications, surgical procedures\n"
        "- Clinical assessments, scores, and outcome scales (e.g. GCS, GOSe, pupil assessment)\n"
        "- Outcomes, patient outcomes, outcome measures\n"
        "- Laboratory values and test categories (blood gases, biochemistry, haematology)\n"
        "- Nursing interventions and bedside care (suctioning, sedation, nutrition)\n"
        "- Medical devices and sensors\n"
        "- Data categories and abstract groupings (Monitoring Data, Demographic Data, "
        "Admission Data, Treatment Data, Outcome Data, Core Monitoring Parameter, "
        "Optional Monitoring Parameter, Baseline Therapy, etc.)\n"
        "- Clinical guidelines and management protocols\n\n"
        "DO NOT extract: authors, institutions, study design, databases, software, "
        "governance, ethics, or administrative concepts.\n\n"
        "### WHAT IS AN ONTOLOGY RELATION?\n"
        "An ontology relation (object property) defines a typed semantic link between "
        "two class types. The DOMAIN is the subject class; the RANGE is the object "
        "class. Think: 'Every instance of [domain] can be linked via [relation] to an "
        "instance of [range].' Model the SCHEMA, not individual facts.\n\n"
        "### RELATION EXTRACTION METHOD (Chain-of-Thought)\n"
        "For each pair of classes in the ontology, reason step by step:\n"
        "1. Do these classes co-occur or interact in the text?\n"
        "2. What is the semantic nature of their link?\n"
        "3. Which class is domain (subject) and which is range (object)?\n"
        "4. Assign a relation label from the patterns below, or create one.\n"
        "5. Find an exact text quote as evidence.\n\n"
        "### ONTOLOGY DESIGN PATTERNS (use when the text supports them)\n"
        "**Pattern 1 — Clinical Monitoring:**\n"
        "  Patient --[has monitoring data]--> Monitoring Data\n"
        "  Monitoring Data --[includes]--> Parameter\n"
        "  Observation --[measures parameter]--> Parameter\n"
        "  Observation --[produced by sensor]--> Sensor\n"
        "  Session --[has timepoint]--> Timepoint\n"
        "  Timepoint --[has observation]--> Observation\n\n"
        "**Pattern 2 — Clinical Intervention:**\n"
        "  Patient --[receives therapy]--> Therapy\n"
        "  Therapy --[targets condition]--> Condition\n"
        "  Condition --[triggers intervention]--> Therapy\n"
        "  Patient --[has surgical procedure]--> Surgical Procedure\n"
        "  Patient --[has nursing intervention]--> Nursing Intervention\n\n"
        "**Pattern 3 — Assessment and Outcome:**\n"
        "  Patient --[has clinical assessment]--> Clinical Assessment\n"
        "  Patient --[has outcome]--> Outcome\n"
        "  Patient --[has laboratory value]--> Laboratory Values\n"
        "  Monitoring Data --[monitoring indicates condition]--> Condition\n\n"
        "**Pattern 4 — Patient Context:**\n"
        "  Patient --[has session]--> Session\n"
        "  Patient --[has demographic data]--> Demographic Data\n"
        "  Patient --[has injury mechanism]--> Injury Mechanism\n\n"
        "**Pattern 5 — Data Quality:**\n"
        "  Data Quality Assessment --[affected by treatment]--> Therapy\n"
        "  Data Quality Assessment --[associated with condition]--> Condition\n"
        "  Observation --[has quality assessment]--> Data Quality Assessment\n\n"
        "**Pattern 6 — Composition (model as HIERARCHY, not relation):**\n"
        "  If X includes/comprises Y → extract as hierarchy: Y subClassOf X\n"
        "  Derived Parameter --[derived from]--> Source Parameter\n\n"
        "CRITICAL: Do NOT use 'includes', 'consists of', 'comprises', or "
        "'is a type of' as relation labels. These are HIERARCHY (subClassOf) "
        "relationships. Model them as hierarchy edges instead.\n\n"
        "You SHOULD also extract other relations the text supports beyond these patterns.\n"
        "Use the abstract superclass as the range (e.g., 'Monitoring Data' not 'ICP').\n\n"
        "### EVIDENCE RULES (CRITICAL)\n"
        "- The 'evidence' field MUST be an **exact quote** copied verbatim from the "
        "SOURCE TEXT.\n"
        "- DO NOT reuse the same evidence quote for multiple items.\n"
        "- DO NOT invent or paraphrase — copy the text exactly.\n"
        "- If you cannot find a real quote for a concept, do NOT include it.\n\n"
        "### LABEL GUIDANCE\n"
        "- Use short, clean noun-phrase labels.\n"
        "- Expand common abbreviations: HR → Heart Rate, ICP → Intracranial Pressure, "
        "CPP → Cerebral Perfusion Pressure, GCS → GCS Assessment, GOSe → GOSe Outcome, "
        "PbtO2/PtiO2 → PbrO2, PRx → Pressure Reactivity Index.\n"
        "- For other abbreviations, expand to full clinical name.\n\n"
        "### HIERARCHY GUIDANCE\n"
        "Extract subClassOf (is-a) edges when the text shows that one concept is a type, "
        "subtype, category member, or specific instance of another.\n"
        "Look for cues such as: 'such as', 'is a', 'type of', 'include:', 'consists of', "
        "'comprises', 'categorized as', 'grouped into', 'subdivided into', enumeration lists.\n"
        "For enumeration patterns (e.g. 'parameters include: X, Y, Z'), extract EACH listed "
        "item as a subclass of the parent category.\n\n"
        "### SELF-CHECK (before finalising output)\n"
        "- Is every label a clinical TYPE (not an instance or a study name)?\n"
        "- Are there any duplicate or synonym labels? Merge them.\n"
        "- Do all relation domain/range labels match extracted class labels?\n"
        "- Does every evidence field contain a verbatim quote from the text?\n\n"
        "### OUTPUT FORMAT (JSON only, no commentary)\n"
        "{\n"
        '  "classes": [{"label": "ClassName", "evidence": "exact quote from text"}],\n'
        '  "relations": [{"label": "rel_name", "domain": "DomainClass", '
        '"range": "RangeClass", "evidence": "exact quote from text"}],\n'
        '  "hierarchy": [{"subClass": "ChildClass", "superClass": "ParentClass", '
        '"evidence": "exact quote from text"}]\n'
        "}\n\n"
        "### SOURCE TEXT\n"
        f"{text_truncated}\n"
    )


def valid_class_norms_to_labels(ontology) -> List[str]:
    """Return sorted list of unique class labels from the ontology."""
    return sorted({(c.label or "").strip() for c in ontology.classes if c.label})


# ---------------------------------------------------------------------------
# Dedicated Relation Completion Pass
# After the main TGC pass extracts classes, this focused pass specifically
# targets relation discovery using class-pair analysis and Chain-of-Thought.
# ---------------------------------------------------------------------------

def _build_relation_completion_prompt(
    all_class_labels: List[str],
    existing_relations: List[Dict],
    corpus_text: str,
) -> str:
    """Build a focused relation extraction prompt that analyses class pairs
    and asks the LLM to identify semantic links between them."""
    if len(all_class_labels) < 2:
        return ""

    cls_list = "\n".join(f"  - {c}" for c in sorted(all_class_labels))

    existing_rel_list = "\n".join(
        f"  - {r.get('label', '?')}({r.get('domain', '?')} → {r.get('range', '?')})"
        for r in existing_relations
    ) if existing_relations else "  (none yet)"

    # Identify classes that have NO relations (dangling)
    rel_participants = set()
    for r in existing_relations:
        rel_participants.add((r.get("domain") or "").strip().lower())
        rel_participants.add((r.get("range") or "").strip().lower())
    dangling = [c for c in all_class_labels if c.strip().lower() not in rel_participants]
    dangling_section = ""
    if dangling:
        dangling_list = "\n".join(f"  - {c}" for c in sorted(dangling))
        dangling_section = (
            "### CLASSES WITH NO RELATIONS (high priority — find connections!)\n"
            f"{dangling_list}\n\n"
        )

    text_truncated = corpus_text[:_TGC_MAX_TEXT_CHARS] if len(corpus_text) > _TGC_MAX_TEXT_CHARS else corpus_text

    return (
        "### TASK: Relation Extraction\n"
        "You are a clinical ontology engineer. Your task is to find semantic "
        "RELATIONS between the classes listed below, using the source text as "
        "evidence. Extract every relation the text explicitly supports.\n\n"
        "### WHAT IS AN ONTOLOGY RELATION?\n"
        "An ontology relation (object property) defines a typed semantic link between "
        "two classes. The DOMAIN is the subject; the RANGE is the object.\n"
        "Model the SCHEMA of the domain — not individual facts. Think:\n"
        "'Every instance of [domain] can be linked via [relation] to an instance of [range].'\n\n"
        "### RELATION EXTRACTION METHOD\n"
        "Systematically examine class pairs and reason step by step:\n"
        "1. Scan the text for sentences where two or more classes co-occur.\n"
        "2. For each co-occurrence, determine: what is the semantic nature of the link?\n"
        "   (measurement, treatment, assessment, outcome, composition, causation, etc.)\n"
        "3. Decide which class is domain (subject) and which is range (object).\n"
        "4. Assign a clear, descriptive relation label.\n"
        "5. Provide an exact verbatim quote from the text as evidence.\n\n"
        "### RELATION PATTERNS\n"
        "Patient-centric:\n"
        "  - has monitoring data: Patient → Monitoring Data\n"
        "  - receives therapy: Patient → Therapy\n"
        "  - has outcome: Patient → Outcome\n"
        "  - has clinical assessment: Patient → Clinical Assessment\n"
        "  - has laboratory value: Patient → Laboratory Values\n"
        "  - has nursing intervention: Patient → Nursing Intervention\n"
        "  - has surgical procedure: Patient → Surgical Procedure\n"
        "  - has session: Patient → Session\n"
        "  - has demographic data: Patient → Demographic Data\n"
        "Clinical process:\n"
        "  - monitoring indicates condition: Monitoring Data → Condition\n"
        "  - targets condition: Therapy → Condition\n"
        "  - triggers intervention: Condition → Therapy\n"
        "  - affected by treatment: Data Quality Assessment → Therapy\n"
        "  - associated with condition: Data Quality Assessment → Condition\n"
        "Structural:\n"
        "  - has observation: Timepoint → Observation\n"
        "  - has timepoint: Session → Timepoint\n"
        "  - measures parameter: Observation → Parameter\n"
        "  - produced by sensor: Observation → Sensor\n"
        "  - has quality assessment: Observation → Data Quality Assessment\n"
        "  - includes: Category → Member\n"
        "  - derived from: DerivedParameter → SourceParameter\n"
        "You SHOULD also discover relations not in this list if the text supports them.\n\n"
        "### CLASSES IN THE ONTOLOGY\n"
        f"{cls_list}\n\n"
        + dangling_section +
        "### EXISTING RELATIONS (do NOT duplicate these)\n"
        f"{existing_rel_list}\n\n"
        "### EVIDENCE RULES (CRITICAL)\n"
        "- The 'evidence' field MUST be an exact verbatim quote from the SOURCE TEXT.\n"
        "- Evidence should mention or imply at least one endpoint class.\n"
        "- Do NOT reuse the same evidence for multiple relations.\n"
        "- Do NOT invent or paraphrase — if no supporting text exists, skip.\n"
        "- Use the abstract superclass as the range (e.g., 'Monitoring Data' not 'ICP').\n\n"
        "### OUTPUT FORMAT (JSON only, no commentary)\n"
        "{\n"
        '  "relations": [{"label": "rel_name", "domain": "DomainClass", '
        '"range": "RangeClass", "evidence": "exact quote from text"}]\n'
        "}\n\n"
        "Extract as many valid relations as the text supports. Be thorough — "
        "especially for the classes that currently have NO relations.\n\n"
        "### SOURCE TEXT\n"
        f"{text_truncated}\n"
    )


# ---------------------------------------------------------------------------
# Chain-of-Layer (CoL) hierarchy extraction
# Inspired by Zeng et al. (CIKM 2024): build taxonomy layer-by-layer using
# hierarchical numbering so the LLM sees the global tree structure.
# ---------------------------------------------------------------------------

def _build_col_prompt(
    all_class_labels: List[str],
    current_hierarchy: List[Dict],
    corpus_text: str,
) -> str:
    """Build a Chain-of-Layer prompt that asks the LLM to organize classes
    into a numbered taxonomy tree, layer by layer from the root."""
    if not all_class_labels:
        return ""

    class_list = "\n".join(f"  - {c}" for c in sorted(all_class_labels))

    existing_hier = "\n".join(
        f"  - {h.get('subClass', '?')} ⊑ {h.get('superClass', '?')}"
        for h in current_hierarchy
    ) if current_hierarchy else "  (none yet)"

    text_truncated = corpus_text[:_TGC_MAX_TEXT_CHARS] if len(corpus_text) > _TGC_MAX_TEXT_CHARS else corpus_text

    return (
        "### TASK: Chain-of-Layer Taxonomy Construction\n"
        "You are a clinical ontology engineer. Organize the classes below into a "
        "hierarchical taxonomy tree using numbered indentation.\n\n"
        "### METHOD (Chain-of-Layer)\n"
        "Build the tree TOP-DOWN, layer by layer:\n"
        "1. First, identify the top-level root categories (e.g. Monitoring Data, "
        "Therapy, Condition, Outcome, etc.)\n"
        "2. Then place intermediate grouping categories under them (e.g. Core Monitoring "
        "Parameter under Monitoring Data)\n"
        "3. Finally place the specific leaf concepts under their parent categories\n\n"
        "Use this numbered format:\n"
        "```\n"
        "1. Monitoring Data\n"
        "  1.1 Core Monitoring Parameter\n"
        "    1.1.1 Heart Rate\n"
        "    1.1.2 Mean Arterial Pressure (MAP)\n"
        "  1.2 Optional Monitoring Parameter\n"
        "    1.2.1 CVP\n"
        "2. Therapy\n"
        "  2.1 Baseline Therapy\n"
        "    2.1.1 Ventilation\n"
        "    2.1.2 Sedation\n"
        "```\n\n"
        "### RULES\n"
        "1. Use ONLY the class labels from the list below — do not invent new labels.\n"
        "2. You MAY create intermediate grouping categories if the text clearly implies "
        "them (e.g. 'Core Monitoring Parameter' as a group for ICP, MAP, Heart Rate).\n"
        "   New intermediate categories must be supported by the source text.\n"
        "3. Every class should appear EXACTLY ONCE in the tree.\n"
        "4. MINIMISE standalone roots — try to place every class under a parent. "
        "Only leave a class as a standalone root if it truly has no parent category. "
        "Prefer placing classes in the tree rather than leaving them unattached, "
        "but only when the text supports the parent-child relationship.\n"
        "5. Use the SOURCE TEXT to decide parent-child relationships — look for "
        "enumeration patterns, category headings, 'such as', 'includes', 'consists of', "
        "'type of', 'classified as'.\n"
        "6. Place Patient, Session, Timepoint, and Observation as standalone roots "
        "(they are entity types, not part of a category hierarchy).\n"
        "7. Maximum depth is 4 levels.\n\n"
        "### ALREADY KNOWN HIERARCHY (for reference)\n"
        f"{existing_hier}\n\n"
        "### CLASSES TO ORGANIZE\n"
        f"{class_list}\n\n"
        "### SOURCE TEXT (use for evidence of parent-child relationships)\n"
        f"{text_truncated}\n\n"
        "### OUTPUT\n"
        "Output ONLY the numbered tree. No JSON, no commentary, no explanation.\n"
        "Every line must start with a number (e.g. '1.', '1.1', '1.1.1').\n"
    )


def _parse_col_tree(tree_text: str) -> List[Dict]:
    """Parse a Chain-of-Layer numbered tree into subClassOf triples.

    Input format:
        1. Monitoring Data
          1.1 Core Monitoring Parameter
            1.1.1 Heart Rate
            1.1.2 Mean Arterial Pressure (MAP)
          1.2 Optional Monitoring Parameter

    Output: [{"subClass": "Heart Rate", "superClass": "Core Monitoring Parameter"}, ...]
    """
    import re

    lines = tree_text.strip().split("\n")
    # Parse each line into (numbering, label)
    entries: List[tuple] = []
    number_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*\.?\s+(.+)$")

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        m = number_pattern.match(line)
        if m:
            numbering = m.group(1)
            label = m.group(2).strip().rstrip(".")
            entries.append((numbering, label))

    if not entries:
        return []

    # Build parent map: for numbering "1.2.3", parent is "1.2"
    number_to_label: Dict[str, str] = {}
    for numbering, label in entries:
        number_to_label[numbering] = label

    hierarchy: List[Dict] = []
    seen: set = set()

    for numbering, label in entries:
        parts = numbering.split(".")
        if len(parts) <= 1:
            continue
        parent_numbering = ".".join(parts[:-1])
        parent_label = number_to_label.get(parent_numbering)
        if parent_label and parent_label != label:
            key = (_norm_label(label), _norm_label(parent_label))
            if key not in seen:
                seen.add(key)
                hierarchy.append({
                    "subClass": label,
                    "superClass": parent_label,
                    "evidence": f"Chain-of-Layer: {label} is organized under {parent_label}",
                })

    return hierarchy


def _build_hierarchy_prompt(
    current_classes: List[str],
    current_hierarchy: List[Dict],
    corpus_text: str,
    new_classes_from_pass1: List[str],
) -> str:
    """Pass 2 prompt: dedicated hierarchy extraction with explicit edge checklist."""
    all_classes = sorted(set(current_classes) | set(new_classes_from_pass1))
    all_class_norms = {_norm_label(c) for c in all_classes}
    existing_hier_norms = {
        (_norm_label(h.get("subClass", "")), _norm_label(h.get("superClass", "")))
        for h in current_hierarchy
    }

    checklist_lines = []
    for sub, sup in _HIERARCHY_EDGE_CHECKLIST:
        sub_norm = _norm_label(sub)
        sup_norm = _norm_label(sup)
        if sub_norm not in all_class_norms or sup_norm not in all_class_norms:
            continue
        if (sub_norm, sup_norm) in existing_hier_norms:
            continue
        checklist_lines.append(f"  {len(checklist_lines)+1}. {sub} ⊑ {sup}")

    if not checklist_lines:
        return ""

    hier_list = "\n".join(
        f"  - {h.get('subClass','?')} ⊑ {h.get('superClass','?')}"
        for h in current_hierarchy
    ) if current_hierarchy else "  (none)"

    text_truncated = corpus_text[:_TGC_MAX_TEXT_CHARS] if len(corpus_text) > _TGC_MAX_TEXT_CHARS else corpus_text

    return (
        "### TASK\n"
        "You are a text-grounded ontology hierarchy completion engine.\n"
        "Below is a checklist of subClassOf edges that are MISSING from the ontology.\n"
        "For each edge, decide: does the source text support this parent-child relationship?\n\n"
        "### ALREADY EXTRACTED HIERARCHY\n"
        f"{hier_list}\n\n"
        "### CANDIDATE EDGES TO CHECK\n"
        "For each numbered edge below, output it in your JSON if the source text mentions, "
        "implies, or lists the child under/alongside the parent:\n"
        f"\n{''.join(line + chr(10) for line in checklist_lines)}\n"
        "### EVIDENCE RULES (CRITICAL)\n"
        "- 'evidence' MUST be an **exact verbatim quote** from the SOURCE TEXT below.\n"
        "- Good evidence includes enumeration patterns (e.g., 'core parameters: MAP, ICP, CPP'), "
        "category headers, or sentences that link the child to the parent.\n"
        "- A brief mention (even 2-3 words like 'CPP monitoring' or 'sedation levels') is "
        "sufficient IF the text really contains those words.\n"
        "- ⚠ DO NOT reuse the same quote for every edge — find the specific text for each.\n"
        "- ⚠ DO NOT invent quotes. If the text does not mention a concept, skip that edge.\n\n"
        "### EXAMPLE\n"
        "Checklist: Sedation ⊑ Baseline Therapy\n"
        'Text: "...baseline treatment included ventilation, sedation, and fluid management..."\n'
        '→ {"subClass": "Sedation", "superClass": "Baseline Therapy", '
        '"evidence": "baseline treatment included ventilation, sedation, and fluid management"}\n\n'
        "Checklist: Cerebral Perfusion Pressure (CPP) ⊑ Derived Parameter\n"
        'Text: "...CPP (= BP - ICP) is derived from blood pressure and ICP..."\n'
        '→ {"subClass": "Cerebral Perfusion Pressure (CPP)", "superClass": "Derived Parameter", '
        '"evidence": "CPP (= BP - ICP) is derived from blood pressure and ICP"}\n\n'
        "### OUTPUT FORMAT (JSON only, no commentary)\n"
        "{\n"
        '  "hierarchy": [{"subClass": "ChildClass", "superClass": "ParentClass", '
        '"evidence": "exact quote from text"}]\n'
        "}\n\n"
        "Output all edges from the checklist that the text explicitly supports. "
        "Each edge must have verbatim evidence from the text.\n\n"
        "### SOURCE TEXT\n"
        f"{text_truncated}\n"
    )


def _add_deterministic_hierarchy(
    ontology,
    existing_hier_keys: set,
    all_class_norms: set,
) -> int:
    """Add hierarchy edges from _HIERARCHY_EDGE_CHECKLIST where both endpoints
    already exist in the ontology. Returns count of edges added."""
    added = 0
    for sub, sup in _HIERARCHY_EDGE_CHECKLIST:
        sub_n = _norm_label(sub)
        sup_n = _norm_label(sup)
        if sub_n == sup_n:
            continue
        if sub_n not in all_class_norms or sup_n not in all_class_norms:
            continue
        key = (sub_n, sup_n)
        if key in existing_hier_keys:
            continue
        ontology.hierarchy.append({
            "subClass": sub,
            "superClass": sup,
            "evidence": f"Structural: {sub} is a subclass of {sup}.",
            "provenance": ["text_grounded_completion"],
            "stratum": "deterministic_hierarchy",
        })
        existing_hier_keys.add(key)
        added += 1
    return added


def run_text_grounded_completion(
    ontology,
    llm,
    corpus_text: str,
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    run_dir: Optional[str] = None,
) -> Dict:
    """
    Text-grounded completion: LLM-driven second-pass extraction with soft hierarchy enrichment.

    1. LLM pass: Extract missing classes and relations from the full document text.
       - Classes are accepted if the LLM provides non-empty evidence (no vocabulary whitelist).
       - Relations require evidence that appears in the source text + bulk-fabrication filtering.
    2. Deterministic hierarchy: Add edges from _HIERARCHY_EDGE_CHECKLIST where both
       endpoints already exist in the ontology. Soft enrichment only — no new classes are forced.
    """
    from .vocabulary import (
        resolve_class_synonyms,
    )
    from .parse import parse_output
    import os

    current_class_labels = sorted({(c.label or "").strip() for c in ontology.classes if c.label})
    current_relations = [
        {"label": r.label, "domain": r.domain, "range": r.range}
        for r in ontology.relations
    ]

    existing_class_norms = {(c.label or "").strip().lower() for c in ontology.classes}
    existing_rel_keys = {
        (_norm_label(r.label or ""), _norm_label(r.domain or ""), _norm_label(r.range or ""))
        for r in ontology.relations
    }
    existing_hier_keys = {
        (_norm_label(h.get("subClass", "")), _norm_label(h.get("superClass", "")))
        for h in ontology.hierarchy
    }

    classes_added = 0
    relations_added = 0
    hierarchy_added = 0

    # ── Step 1: Deterministic hierarchy (pre-LLM) ───────────────────────
    all_class_norms = {_norm_label(c) for c in current_class_labels}
    hierarchy_added += _add_deterministic_hierarchy(ontology, existing_hier_keys, all_class_norms)

    if progress_callback is not None:
        progress_callback(0, 1, f"Deterministic hierarchy: +{hierarchy_added} edges. Running LLM pass…")

    # ── Step 2: LLM pass for classes + relations ─────────────────────────
    pass1_prompt = _build_classes_relations_prompt(
        current_class_labels,
        current_relations,
        corpus_text,
    )

    raw1 = _generate_with_cancel_check(
        llm, pass1_prompt, progress_callback, 0, 1,
        "Text-grounded completion: waiting for LLM…",
        max_tokens=TEXT_GROUNDED_MAX_TOKENS,
    )

    if run_dir:
        prompts_dir = os.path.join(run_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        try:
            with open(os.path.join(prompts_dir, "tgc_pass1_prompt.txt"), "w", encoding="utf-8") as fh:
                fh.write(pass1_prompt)
            with open(os.path.join(prompts_dir, "tgc_pass1_response.txt"), "w", encoding="utf-8") as fh:
                fh.write(raw1 or "")
        except Exception:
            pass

    parsed1 = parse_output(raw1 or "")
    parsed1 = resolve_class_synonyms(parsed1)

    # Relations: strict evidence filtering (bulk-fabrication + text verification)
    parsed1["relations"] = _filter_bulk_fabricated(parsed1.get("relations") or [])
    parsed1["relations"] = [
        r for r in parsed1["relations"]
        if _evidence_appears_in_text(r.get("evidence", ""), corpus_text)
    ]

    new_class_labels: List[str] = []
    from ..ontology.model import ClassEntity, RelationEntity

    # Classes: accept any LLM-suggested class with non-empty evidence.
    for c in parsed1.get("classes") or []:
        label = (c.get("label") or "").strip()
        if not label or label.lower() in existing_class_norms:
            continue
        evidence = (c.get("evidence") or "").strip()
        if not evidence:
            continue
        ontology.classes.append(ClassEntity(label=label, evidence=evidence))
        existing_class_norms.add(label.lower())
        new_class_labels.append(label)
        classes_added += 1

    # Rebuild valid class map for relation domain/range checking and CoL
    valid_class_norms: Dict[str, str] = {}
    for c in current_class_labels:
        valid_class_norms[_norm_label(c)] = c
    for c in new_class_labels:
        valid_class_norms[_norm_label(c)] = c

    for r in parsed1.get("relations") or []:
        label = (r.get("label") or "").strip()
        domain = (r.get("domain") or "").strip()
        range_ = (r.get("range") or "").strip()
        evidence = (r.get("evidence") or "").strip()
        if not label or not domain or not range_ or not evidence:
            continue
        if _norm_label(domain) not in valid_class_norms or _norm_label(range_) not in valid_class_norms:
            continue
        key = (_norm_label(label), _norm_label(domain), _norm_label(range_))
        if key in existing_rel_keys:
            continue
        ontology.relations.append(RelationEntity(
            label=label, domain=domain, range=range_, evidence=evidence,
        ))
        existing_rel_keys.add(key)
        relations_added += 1

    # ── Step 2b: Hierarchy from LLM pass 1 ───────────────────────────────
    # The pass 1 prompt now includes hierarchy in its output format
    for edge in parsed1.get("hierarchy") or []:
        sub = (edge.get("subClass") or "").strip()
        sup = (edge.get("superClass") or "").strip()
        evidence = (edge.get("evidence") or "").strip()
        if not sub or not sup or not evidence:
            continue
        key = (_norm_label(sub), _norm_label(sup))
        if key in existing_hier_keys:
            continue
        if _norm_label(sub) not in valid_class_norms or _norm_label(sup) not in valid_class_norms:
            continue
        ontology.hierarchy.append({
            "subClass": sub, "superClass": sup, "evidence": evidence,
            "provenance": ["text_grounded_completion"],
            "stratum": "tgc_llm_hierarchy",
        })
        existing_hier_keys.add(key)
        hierarchy_added += 1

    # ── Step 2c: Dedicated Relation Completion pass ─────────────────────
    # Focused LLM pass targeting relation discovery, especially for classes
    # that have no relations yet (dangling classes / class-pair analysis).
    rel_pass_relations_added = 0
    all_labels_for_rel = sorted(valid_class_norms.values())
    current_rels_for_prompt = [
        {"label": r.label, "domain": r.domain, "range": r.range}
        for r in ontology.relations
    ]
    rel_prompt = _build_relation_completion_prompt(
        all_labels_for_rel, current_rels_for_prompt, corpus_text,
    )
    if rel_prompt:
        if progress_callback is not None:
            progress_callback(0, 1, "Dedicated relation extraction pass…")
        raw_rel = _generate_with_cancel_check(
            llm, rel_prompt, progress_callback, 0, 1,
            "Text-grounded completion: relation extraction…",
            max_tokens=TEXT_GROUNDED_MAX_TOKENS,
        )
        if run_dir:
            try:
                with open(os.path.join(prompts_dir, "tgc_relation_pass_prompt.txt"), "w", encoding="utf-8") as fh:
                    fh.write(rel_prompt)
                with open(os.path.join(prompts_dir, "tgc_relation_pass_response.txt"), "w", encoding="utf-8") as fh:
                    fh.write(raw_rel or "")
            except Exception:
                pass

        parsed_rel = parse_output(raw_rel or "")
        parsed_rel = resolve_class_synonyms(parsed_rel)
        parsed_rel["relations"] = _filter_bulk_fabricated(parsed_rel.get("relations") or [])
        parsed_rel["relations"] = [
            r for r in parsed_rel["relations"]
            if _evidence_appears_in_text(r.get("evidence", ""), corpus_text)
        ]

        for r in parsed_rel.get("relations") or []:
            label = (r.get("label") or "").strip()
            domain = (r.get("domain") or "").strip()
            range_ = (r.get("range") or "").strip()
            evidence = (r.get("evidence") or "").strip()
            if not label or not domain or not range_ or not evidence:
                continue
            if _norm_label(domain) not in valid_class_norms and _norm_label(range_) not in valid_class_norms:
                continue
            key = (_norm_label(label), _norm_label(domain), _norm_label(range_))
            if key in existing_rel_keys:
                continue
            ontology.relations.append(RelationEntity(
                label=label, domain=domain, range=range_, evidence=evidence,
                provenance=["relation_completion_pass"],
            ))
            existing_rel_keys.add(key)
            relations_added += 1
            rel_pass_relations_added += 1

    # ── Step 2d: Chain-of-Layer (CoL) hierarchy pass ────────────────────
    # Ask the LLM to organize all classes into a numbered taxonomy tree,
    # then parse the tree into subClassOf triples (Zeng et al., CIKM 2024).
    col_display_labels = valid_class_norms_to_labels(ontology)
    current_hierarchy_snapshot = list(ontology.hierarchy)
    col_prompt = _build_col_prompt(
        col_display_labels,
        current_hierarchy_snapshot,
        corpus_text,
    )
    col_hierarchy_count = 0
    if col_prompt:
        raw_col = _generate_with_cancel_check(
            llm, col_prompt, progress_callback, 0, 1,
            "Text-grounded completion: Chain-of-Layer taxonomy…",
            max_tokens=TEXT_GROUNDED_MAX_TOKENS,
        )
        if run_dir:
            try:
                with open(os.path.join(prompts_dir, "tgc_col_prompt.txt"), "w", encoding="utf-8") as fh:
                    fh.write(col_prompt)
                with open(os.path.join(prompts_dir, "tgc_col_response.txt"), "w", encoding="utf-8") as fh:
                    fh.write(raw_col or "")
            except Exception:
                pass
        col_edges = _parse_col_tree(raw_col or "")
        for edge in col_edges:
            sub = (edge.get("subClass") or "").strip()
            sup = (edge.get("superClass") or "").strip()
            if not sub or not sup:
                continue
            key = (_norm_label(sub), _norm_label(sup))
            if key in existing_hier_keys:
                continue
            # Accept if at least one endpoint is a known class
            sub_known = _norm_label(sub) in valid_class_norms
            sup_known = _norm_label(sup) in valid_class_norms
            if not sub_known and not sup_known:
                continue
            # If the superclass is a new intermediate category from CoL,
            # add it as a class so the hierarchy edge is valid
            if not sup_known:
                ontology.classes.append(ClassEntity(
                    label=sup,
                    evidence=f"Intermediate category introduced by Chain-of-Layer taxonomy construction",
                ))
                existing_class_norms.add(sup.lower())
                new_class_labels.append(sup)
                valid_class_norms[_norm_label(sup)] = sup
                classes_added += 1
            if not sub_known:
                ontology.classes.append(ClassEntity(
                    label=sub,
                    evidence=f"Intermediate category introduced by Chain-of-Layer taxonomy construction",
                ))
                existing_class_norms.add(sub.lower())
                new_class_labels.append(sub)
                valid_class_norms[_norm_label(sub)] = sub
                classes_added += 1
            ontology.hierarchy.append({
                "subClass": sub, "superClass": sup,
                "evidence": edge.get("evidence", ""),
                "provenance": ["chain_of_layer"],
                "stratum": "col_hierarchy",
            })
            existing_hier_keys.add(key)
            hierarchy_added += 1
            col_hierarchy_count += 1

    # ── Step 3: Deterministic hierarchy (post-LLM) ──────────────────────
    # Re-scan checklist now that new classes may have been added
    all_class_norms = {_norm_label(c) for c in current_class_labels} | {_norm_label(c) for c in new_class_labels}
    hierarchy_added += _add_deterministic_hierarchy(ontology, existing_hier_keys, all_class_norms)

    # ── Diagnostics ──────────────────────────────────────────────────────
    if run_dir:
        diag = {
            "pass1_raw_len": len(raw1 or ""),
            "pass1_classes_parsed": len(parsed1.get("classes") or []),
            "pass1_relations_parsed": len(parsed1.get("relations") or []),
            "relation_pass_relations_added": rel_pass_relations_added,
            "col_hierarchy_edges": col_hierarchy_count,
            "classes_added": classes_added,
            "relations_added": relations_added,
            "hierarchy_added": hierarchy_added,
        }
        try:
            import json as _json
            with open(os.path.join(prompts_dir, "tgc_diagnostic.json"), "w", encoding="utf-8") as fh:
                _json.dump(diag, fh, indent=2)
        except Exception:
            pass

    if progress_callback is not None:
        progress_callback(1, 1, f"Text-grounded completion: +{classes_added} classes, +{relations_added} relations, +{hierarchy_added} hierarchy.")

    return {
        "classes_added": classes_added,
        "relations_added": relations_added,
        "hierarchy_added": hierarchy_added,
    }


# ---------------------------------------------------------------------------
# Orphan Rescue Pass (post-cleanup)
# After cleanup prunes noisy relations and classes, some legitimate classes
# may become orphaned (no relations, no hierarchy edges). This pass asks
# the LLM to infer connections for those orphans using the source text.
# ---------------------------------------------------------------------------

def _find_orphan_classes(ontology) -> List[str]:
    """Return labels of classes not referenced in any relation or hierarchy edge."""
    connected: set = set()
    for r in ontology.relations:
        if r.domain:
            connected.add(_norm_label(r.domain))
        if r.range:
            connected.add(_norm_label(r.range))
    for h in ontology.hierarchy:
        sub = (h.get("subClass") or "").strip()
        sup = (h.get("superClass") or "").strip()
        if sub:
            connected.add(_norm_label(sub))
        if sup:
            connected.add(_norm_label(sup))
    orphans = []
    for c in ontology.classes:
        label = (c.label or "").strip()
        if label and _norm_label(label) not in connected:
            orphans.append(label)
    return sorted(orphans)


def _build_orphan_rescue_prompt(
    orphan_labels: List[str],
    all_class_labels: List[str],
    existing_relations: List[Dict],
    existing_hierarchy: List[Dict],
    corpus_text: str,
) -> str:
    """Build a prompt that asks the LLM to find relations and hierarchy edges
    for orphaned classes, using the source text as evidence."""
    if not orphan_labels:
        return ""

    orphan_list = "\n".join(f"  - {c}" for c in orphan_labels)
    connected_labels = [c for c in all_class_labels if c not in orphan_labels]
    connected_list = "\n".join(f"  - {c}" for c in sorted(connected_labels))

    rel_list = "\n".join(
        f"  - {r.get('label', '?')}({r.get('domain', '?')} → {r.get('range', '?')})"
        for r in existing_relations[:30]
    ) if existing_relations else "  (none)"

    hier_list = "\n".join(
        f"  - {h.get('subClass', '?')} ⊑ {h.get('superClass', '?')}"
        for h in existing_hierarchy[:30]
    ) if existing_hierarchy else "  (none)"

    text_truncated = corpus_text[:_TGC_MAX_TEXT_CHARS] if len(corpus_text) > _TGC_MAX_TEXT_CHARS else corpus_text

    return (
        "### TASK: Orphan Class Rescue — Connect Isolated Nodes\n"
        "You are a clinical ontology engineer. The ontology below has several classes "
        "that are ORPHANED — they have no relations and no hierarchy edges connecting "
        "them to the rest of the ontology. Your task is to find meaningful connections "
        "for these orphaned classes using the source text as evidence.\n\n"
        "### WHAT TO DO\n"
        "For each orphaned class, search the source text for evidence of:\n"
        "1. **Relations**: semantic links to other classes (e.g., 'Patient receives Therapy')\n"
        "2. **Hierarchy edges**: parent-child relationships (e.g., 'Heart Rate is a type "
        "of Core Monitoring Parameter')\n\n"
        "Use Chain-of-Thought reasoning for each orphan:\n"
        "1. What does this class represent in the clinical domain?\n"
        "2. Which other classes in the ontology would it naturally connect to?\n"
        "3. What is the nature of the connection? (relation type, or is-a hierarchy)\n"
        "4. Does the source text provide evidence for this connection?\n\n"
        "### ORPHANED CLASSES (your main target)\n"
        f"{orphan_list}\n\n"
        "### CONNECTED CLASSES (use these as endpoints for new connections)\n"
        f"{connected_list}\n\n"
        "### EXISTING RELATIONS (for context — do not duplicate)\n"
        f"{rel_list}\n\n"
        "### EXISTING HIERARCHY (for context — do not duplicate)\n"
        f"{hier_list}\n\n"
        "### RELATION PATTERNS\n"
        "  - has monitoring data: Patient → Monitoring Data\n"
        "  - receives therapy: Patient → Therapy\n"
        "  - has outcome: Patient → Outcome\n"
        "  - has clinical assessment: Patient → Clinical Assessment\n"
        "  - has laboratory value: Patient → Laboratory Values\n"
        "  - monitoring indicates condition: Monitoring Data → Condition\n"
        "  - targets condition: Therapy → Condition\n"
        "  - triggers intervention: Condition → Therapy\n"
        "  - includes: Category → Member\n"
        "  - measures parameter: Observation → Parameter\n"
        "  - produced by sensor: Observation → Sensor\n"
        "  - has timepoint: Session → Timepoint\n"
        "  - has observation: Timepoint → Observation\n"
        "  - derived from: DerivedParameter → SourceParameter\n"
        "You may also create other relation labels if the text supports them.\n\n"
        "### EVIDENCE RULES (CRITICAL)\n"
        "- Every relation and hierarchy edge MUST have an 'evidence' field with an "
        "**exact verbatim quote** from the SOURCE TEXT.\n"
        "- Do NOT invent or paraphrase evidence. If you cannot find real evidence, "
        "skip that orphan — not every class needs a connection.\n"
        "- Use the abstract superclass as the range (e.g., 'Monitoring Data' not 'ICP').\n\n"
        "### OUTPUT FORMAT (JSON only, no commentary)\n"
        "{\n"
        '  "relations": [{"label": "rel_name", "domain": "DomainClass", '
        '"range": "RangeClass", "evidence": "exact quote from text"}],\n'
        '  "hierarchy": [{"subClass": "ChildClass", "superClass": "ParentClass", '
        '"evidence": "exact quote from text"}]\n'
        "}\n\n"
        "Connect as many orphans as you can — every rescued class improves the ontology. "
        "But ONLY with real evidence from the text.\n\n"
        "### SOURCE TEXT\n"
        f"{text_truncated}\n"
    )


def run_orphan_rescue(
    ontology,
    llm,
    corpus_text: str,
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    run_dir: Optional[str] = None,
) -> Dict:
    """Post-cleanup orphan rescue: identify classes with no relations or hierarchy
    edges, then ask the LLM to infer connections using the source text.

    Should be called AFTER apply_builtin_cleanup so it targets truly orphaned classes.
    """
    from .vocabulary import resolve_class_synonyms
    from .parse import parse_output
    from ..ontology.model import ClassEntity, RelationEntity
    import os

    orphans = _find_orphan_classes(ontology)
    if not orphans:
        if progress_callback is not None:
            progress_callback(1, 1, "Orphan rescue: no orphaned classes found.")
        return {"orphans_found": 0, "relations_added": 0, "hierarchy_added": 0}

    all_labels = sorted({(c.label or "").strip() for c in ontology.classes if c.label})
    existing_rels = [
        {"label": r.label, "domain": r.domain, "range": r.range}
        for r in ontology.relations
    ]
    existing_hier = list(ontology.hierarchy)
    valid_class_norms: Dict[str, str] = {}
    for c in all_labels:
        valid_class_norms[_norm_label(c)] = c

    prompt = _build_orphan_rescue_prompt(
        orphans, all_labels, existing_rels, existing_hier, corpus_text,
    )
    if not prompt:
        return {"orphans_found": len(orphans), "relations_added": 0, "hierarchy_added": 0}

    if progress_callback is not None:
        progress_callback(0, 1, f"Orphan rescue: {len(orphans)} isolated classes found, asking LLM…")

    raw = _generate_with_cancel_check(
        llm, prompt, progress_callback, 0, 1,
        f"Orphan rescue: connecting {len(orphans)} isolated classes…",
        max_tokens=TEXT_GROUNDED_MAX_TOKENS,
    )

    if run_dir:
        prompts_dir = os.path.join(run_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        try:
            with open(os.path.join(prompts_dir, "orphan_rescue_prompt.txt"), "w", encoding="utf-8") as fh:
                fh.write(prompt)
            with open(os.path.join(prompts_dir, "orphan_rescue_response.txt"), "w", encoding="utf-8") as fh:
                fh.write(raw or "")
        except Exception:
            pass

    parsed = parse_output(raw or "")
    parsed = resolve_class_synonyms(parsed)

    existing_rel_keys = {
        (_norm_label(r.label or ""), _norm_label(r.domain or ""), _norm_label(r.range or ""))
        for r in ontology.relations
    }
    existing_hier_keys = {
        (_norm_label(h.get("subClass", "")), _norm_label(h.get("superClass", "")))
        for h in ontology.hierarchy
    }

    relations_added = 0
    hierarchy_added = 0

    # Filter relations: evidence must appear in text
    parsed["relations"] = _filter_bulk_fabricated(parsed.get("relations") or [])
    parsed["relations"] = [
        r for r in parsed["relations"]
        if _evidence_appears_in_text(r.get("evidence", ""), corpus_text)
    ]

    for r in parsed.get("relations") or []:
        label = (r.get("label") or "").strip()
        domain = (r.get("domain") or "").strip()
        range_ = (r.get("range") or "").strip()
        evidence = (r.get("evidence") or "").strip()
        if not label or not domain or not range_ or not evidence:
            continue
        # At least one endpoint must be known
        dom_known = _norm_label(domain) in valid_class_norms
        ran_known = _norm_label(range_) in valid_class_norms
        if not dom_known and not ran_known:
            continue
        key = (_norm_label(label), _norm_label(domain), _norm_label(range_))
        if key in existing_rel_keys:
            continue
        # If one endpoint is new (e.g. LLM inferred a superclass), add it as a class
        if not dom_known:
            ontology.classes.append(ClassEntity(
                label=domain, evidence=f"Added via orphan rescue: endpoint for relation '{label}'.",
            ))
            valid_class_norms[_norm_label(domain)] = domain
        if not ran_known:
            ontology.classes.append(ClassEntity(
                label=range_, evidence=f"Added via orphan rescue: endpoint for relation '{label}'.",
            ))
            valid_class_norms[_norm_label(range_)] = range_
        ontology.relations.append(RelationEntity(
            label=label, domain=domain, range=range_, evidence=evidence,
            provenance=["orphan_rescue"],
        ))
        existing_rel_keys.add(key)
        relations_added += 1

    for edge in parsed.get("hierarchy") or []:
        sub = (edge.get("subClass") or "").strip()
        sup = (edge.get("superClass") or "").strip()
        evidence = (edge.get("evidence") or "").strip()
        if not sub or not sup or not evidence:
            continue
        if not _evidence_appears_in_text(evidence, corpus_text):
            continue
        key = (_norm_label(sub), _norm_label(sup))
        if key in existing_hier_keys:
            continue
        sub_known = _norm_label(sub) in valid_class_norms
        sup_known = _norm_label(sup) in valid_class_norms
        if not sub_known and not sup_known:
            continue
        if not sup_known:
            ontology.classes.append(ClassEntity(
                label=sup, evidence=f"Superclass introduced by orphan rescue.",
            ))
            valid_class_norms[_norm_label(sup)] = sup
        if not sub_known:
            ontology.classes.append(ClassEntity(
                label=sub, evidence=f"Subclass introduced by orphan rescue.",
            ))
            valid_class_norms[_norm_label(sub)] = sub
        ontology.hierarchy.append({
            "subClass": sub, "superClass": sup, "evidence": evidence,
            "provenance": ["orphan_rescue"],
            "stratum": "orphan_rescue",
        })
        existing_hier_keys.add(key)
        hierarchy_added += 1

    if progress_callback is not None:
        progress_callback(1, 1,
            f"Orphan rescue: {len(orphans)} orphans → +{relations_added} relations, +{hierarchy_added} hierarchy.")

    return {
        "orphans_found": len(orphans),
        "relations_added": relations_added,
        "hierarchy_added": hierarchy_added,
    }


# ---------------------------------------------------------------------------
# LLM Chain-of-Thought Ontology Refinement
# ---------------------------------------------------------------------------

REFINEMENT_MAX_TOKENS = 8192


def _build_refinement_prompt(ontology, corpus_text: str) -> str:
    """Build a domain-agnostic CoT prompt that asks the LLM to review every
    class, relation, and hierarchy edge for clinical relevance and semantic
    correctness.  The LLM identifies items to REMOVE or CORRECT — it does
    not need to list items to keep."""

    classes_block = []
    for c in ontology.classes:
        lab = (c.label or "").strip()
        ev = (getattr(c, "evidence", None) or "")[:200]
        classes_block.append(f"  - {lab}  [evidence: {ev}]" if ev else f"  - {lab}")

    relations_block = []
    for r in ontology.relations:
        lab = (r.label or "").strip()
        dom = (r.domain or "").strip()
        ran = (r.range or "").strip()
        ev = (getattr(r, "evidence", None) or "")[:200]
        relations_block.append(f"  - {lab}({dom} → {ran})  [evidence: {ev}]")

    hierarchy_block = []
    for h in ontology.hierarchy:
        sub = (h.get("subClass") or "").strip()
        sup = (h.get("superClass") or "").strip()
        ev = (h.get("evidence") or "")[:200]
        hierarchy_block.append(f"  - {sub} ⊑ {sup}  [evidence: {ev}]")

    text_preview = corpus_text[:6000] if len(corpus_text) > 6000 else corpus_text

    return (
        "You are an ontology quality reviewer for clinical and biomedical domain ontologies.\n\n"
        "You have been given a generated ontology extracted from clinical/biomedical literature.\n"
        "Your task is to review the ontology and identify ONLY items that are clearly\n"
        "NOT clinical. Be VERY CONSERVATIVE — when in doubt, KEEP the item.\n\n"
        "### CRITICAL: BE CONSERVATIVE\n"
        "- Your primary goal is to PRESERVE the ontology. Only remove items you are\n"
        "  HIGHLY CONFIDENT do not belong.\n"
        "- If a concept COULD be clinical in ANY reasonable interpretation, KEEP it.\n"
        "- Abstract grouping categories are ESSENTIAL in ontologies — they organise\n"
        "  specific concepts into meaningful hierarchies. NEVER remove a class just\n"
        "  because it is abstract or general.\n"
        "- Removing a valid class destroys all its relations and hierarchy edges,\n"
        "  causing massive cascading damage to the ontology. Err on the side of keeping.\n\n"
        "### WHAT BELONGS (KEEP these — do NOT remove)\n"
        "- Clinical conditions, diseases, injuries, syndromes\n"
        "- Physiological parameters and measurements (e.g. heart rate, ICP, blood pressure)\n"
        "- Treatments, therapies, interventions, drugs, surgical procedures\n"
        "- Clinical assessments and scores (e.g. GCS, pupil reactivity, CT scan findings)\n"
        "- Laboratory values and tests (e.g. blood gases, biochemistry, haematology)\n"
        "- Medical devices and sensors that measure clinical parameters\n"
        "- Patient-related concepts (demographics, outcomes, patient)\n"
        "- Data categories that organise clinical observations (e.g. 'Monitoring Data',\n"
        "  'Laboratory Values', 'Admission Data', 'Treatment Data', 'Outcome Data')\n"
        "- Abstract superclasses that group clinical concepts:\n"
        "  'Condition', 'Outcome', 'Observation', 'Parameter', 'Therapy',\n"
        "  'Assessment', 'Intervention', 'Core Monitoring Parameter',\n"
        "  'Optional Monitoring Parameter', 'Baseline Therapy', 'Guidelines',\n"
        "  'Guidelines for Management', etc.\n"
        "  These are VALID ontological grouping classes — do NOT remove them.\n"
        "- Clinical guidelines and management protocols\n\n"
        "### WHAT TO REMOVE (only these categories, with high confidence)\n"
        "- Research project names and consortium names (e.g. 'BrainIT Project')\n"
        "- Governance/administrative concepts (e.g. 'Steering Group', 'Membership')\n"
        "- Data infrastructure: databases, software tools, data collection tools\n"
        "- Economic or social impact concepts (e.g. 'Economic Consequences',\n"
        "  'Social Consequences') — these are about societal impact, not clinical care\n"
        "- Institution names (universities, hospitals as organisations)\n"
        "- Industry/regulatory bodies (e.g. 'Medical Device Industry', 'Health Authorities')\n"
        "- Publishing metadata: journals, authors, DOIs\n\n"
        "### HIERARCHY REVIEW (be thorough here — hierarchy errors are important)\n"
        "Carefully check EVERY hierarchy edge for semantic correctness:\n"
        "- **Nonsensical edges** (ALWAYS remove): the subclass relationship is logically\n"
        "  impossible. Examples:\n"
        "  • 'Patient ⊑ Secondary Insult' — a patient is NOT a type of insult → REMOVE\n"
        "  • 'Monitoring Data ⊑ Patient' — data is NOT a type of patient → REMOVE\n"
        "  • 'Therapy ⊑ Laboratory Values' — therapy is NOT a lab value → REMOVE\n"
        "  Test: 'Is every X a kind of Y?' If the answer is obviously NO, remove it.\n"
        "- **Clear inversions** (CORRECT, do not remove): child and parent are swapped.\n"
        "  Examples:\n"
        "  • 'Brain Injury ⊑ Traumatic Brain Injury' → CORRECT to 'TBI ⊑ Brain Injury'\n"
        "  • 'Condition ⊑ Traumatic Brain Injury' → CORRECT to 'TBI ⊑ Condition'\n"
        "  Test: 'Is every X a kind of Y, or is it the other way around?'\n"
        "- Do NOT remove hierarchy edges just because the grouping seems unusual —\n"
        "  ontologies often have domain-specific groupings.\n\n"
        "### RELATION REVIEW\n"
        "Only flag relations where BOTH endpoints are clearly non-clinical:\n"
        "- If the domain OR range is a valid clinical concept, KEEP the relation.\n"
        "- Do NOT remove relations just because the label is imprecise.\n\n"
        "### CHAIN-OF-THOUGHT\n"
        "For each item you flag, reason step by step:\n"
        "1. What does this concept refer to in the source text?\n"
        "2. Is it clearly non-clinical (governance/infrastructure/economic/publishing)?\n"
        "3. Could it have ANY clinical interpretation? If yes → KEEP.\n"
        "4. Decision: REMOVE or CORRECT (only if highly confident)\n\n"
        "### OUTPUT FORMAT\n"
        "Return ONLY a JSON object (no markdown fences) with these keys:\n"
        "```\n"
        "{\n"
        '  "classes_to_remove": [\n'
        '    {"label": "...", "reason": "..."}\n'
        "  ],\n"
        '  "relations_to_remove": [\n'
        '    {"label": "...", "domain": "...", "range": "...", "reason": "..."}\n'
        "  ],\n"
        '  "hierarchy_to_remove": [\n'
        '    {"subClass": "...", "superClass": "...", "reason": "..."}\n'
        "  ],\n"
        '  "hierarchy_corrections": [\n'
        '    {"old_sub": "...", "old_super": "...", "new_sub": "...", "new_super": "...", "reason": "..."}\n'
        "  ]\n"
        "}\n"
        "```\n"
        "If everything is correct (or you are not confident enough to remove anything),\n"
        "return empty arrays for each key. It is perfectly fine to return all empty arrays.\n"
        "Do NOT wrap the JSON in markdown code fences.\n\n"
        "### SOURCE TEXT (first 6000 chars for context)\n"
        f"{text_preview}\n\n"
        "### CLASSES\n"
        + "\n".join(classes_block) + "\n\n"
        "### RELATIONS\n"
        + "\n".join(relations_block) + "\n\n"
        "### HIERARCHY (subclass ⊑ superclass)\n"
        + "\n".join(hierarchy_block) + "\n"
    )


def _parse_refinement_response(raw: str) -> Dict:
    """Parse the LLM refinement response into structured removal/correction lists.

    Tries JSON parsing with multiple fallbacks (embedded JSON, truncation recovery).
    Returns a dict with classes_to_remove, relations_to_remove, hierarchy_to_remove,
    hierarchy_corrections — each as a list (possibly empty).
    """
    parsed = _try_parse_json_any(raw)
    if parsed is None:
        parsed = _try_parse_embedded_json_any(raw)
    if parsed is None:
        from .parse import _try_recover_truncated_json
        parsed = _try_recover_truncated_json(raw or "")
    if not isinstance(parsed, dict):
        parsed = {}

    def _as_list(val):
        return val if isinstance(val, list) else []

    return {
        "classes_to_remove": _as_list(parsed.get("classes_to_remove")),
        "relations_to_remove": _as_list(parsed.get("relations_to_remove")),
        "hierarchy_to_remove": _as_list(parsed.get("hierarchy_to_remove")),
        "hierarchy_corrections": _as_list(parsed.get("hierarchy_corrections")),
    }


def run_ontology_refinement(
    ontology,
    llm,
    corpus_text: str,
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    run_dir: Optional[str] = None,
) -> Dict:
    """LLM Chain-of-Thought ontology refinement pass.

    Presents the full generated ontology to the LLM and asks it to identify
    non-clinical classes, semantically incorrect relations, and hierarchy
    errors (inversions, nonsensical edges).  Applies the removals and
    corrections, returning counts of changes made.

    Should be called AFTER cleanup and orphan rescue, BEFORE rule-based reasoning.
    """
    from ..ontology.model import ClassEntity, RelationEntity

    if not ontology.classes:
        return {
            "classes_removed": 0, "relations_removed": 0,
            "hierarchy_removed": 0, "hierarchy_corrected": 0,
        }

    if progress_callback is not None:
        progress_callback(0, 1, "LLM refinement: building review prompt…")

    prompt = _build_refinement_prompt(ontology, corpus_text)

    raw = _generate_with_cancel_check(
        llm, prompt, progress_callback, 0, 1,
        "LLM refinement: reviewing ontology for quality…",
        max_tokens=REFINEMENT_MAX_TOKENS,
    )

    if run_dir:
        prompts_dir = os.path.join(run_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        try:
            with open(os.path.join(prompts_dir, "refinement_prompt.txt"), "w", encoding="utf-8") as fh:
                fh.write(prompt)
            with open(os.path.join(prompts_dir, "refinement_response.txt"), "w", encoding="utf-8") as fh:
                fh.write(raw or "")
        except Exception:
            pass

    result = _parse_refinement_response(raw or "")

    # --- Apply class removals ---
    remove_class_norms = {
        _norm_label(item.get("label", ""))
        for item in result["classes_to_remove"]
        if item.get("label")
    }
    classes_before = len(ontology.classes)
    if remove_class_norms:
        ontology.classes[:] = [
            c for c in ontology.classes
            if _norm_label(c.label or "") not in remove_class_norms
        ]
    classes_removed = classes_before - len(ontology.classes)

    # --- Apply relation removals ---
    remove_rel_keys = set()
    for item in result["relations_to_remove"]:
        lab = _norm_label(item.get("label", ""))
        dom = _norm_label(item.get("domain", ""))
        ran = _norm_label(item.get("range", ""))
        if lab or dom or ran:
            remove_rel_keys.add((lab, dom, ran))

    relations_before = len(ontology.relations)
    if remove_rel_keys:
        ontology.relations[:] = [
            r for r in ontology.relations
            if (_norm_label(r.label or ""), _norm_label(r.domain or ""), _norm_label(r.range or ""))
            not in remove_rel_keys
        ]
    # Also remove relations whose domain or range was removed
    if remove_class_norms:
        ontology.relations[:] = [
            r for r in ontology.relations
            if _norm_label(r.domain or "") not in remove_class_norms
            and _norm_label(r.range or "") not in remove_class_norms
        ]
    relations_removed = relations_before - len(ontology.relations)

    # --- Apply hierarchy removals ---
    remove_hier_keys = set()
    for item in result["hierarchy_to_remove"]:
        sub = _norm_label(item.get("subClass", ""))
        sup = _norm_label(item.get("superClass", ""))
        if sub or sup:
            remove_hier_keys.add((sub, sup))

    hierarchy_before = len(ontology.hierarchy)
    if remove_hier_keys:
        ontology.hierarchy[:] = [
            h for h in ontology.hierarchy
            if (_norm_label(h.get("subClass", "")), _norm_label(h.get("superClass", "")))
            not in remove_hier_keys
        ]
    # Also remove hierarchy edges referencing removed classes
    if remove_class_norms:
        ontology.hierarchy[:] = [
            h for h in ontology.hierarchy
            if _norm_label(h.get("subClass", "")) not in remove_class_norms
            and _norm_label(h.get("superClass", "")) not in remove_class_norms
        ]
    hierarchy_removed = hierarchy_before - len(ontology.hierarchy)

    # --- Apply hierarchy corrections (inversions) ---
    hierarchy_corrected = 0
    for correction in result["hierarchy_corrections"]:
        old_sub = _norm_label(correction.get("old_sub", ""))
        old_sup = _norm_label(correction.get("old_super", ""))
        new_sub = (correction.get("new_sub") or "").strip()
        new_sup = (correction.get("new_super") or "").strip()
        if not old_sub or not old_sup or not new_sub or not new_sup:
            continue
        for h in ontology.hierarchy:
            h_sub = _norm_label(h.get("subClass", ""))
            h_sup = _norm_label(h.get("superClass", ""))
            if h_sub == old_sub and h_sup == old_sup:
                h["subClass"] = new_sub
                h["superClass"] = new_sup
                hierarchy_corrected += 1
                break

    if progress_callback is not None:
        progress_callback(1, 1,
            f"LLM refinement: -{classes_removed} classes, -{relations_removed} relations, "
            f"-{hierarchy_removed} hierarchy, {hierarchy_corrected} corrections.")

    return {
        "classes_removed": classes_removed,
        "relations_removed": relations_removed,
        "hierarchy_removed": hierarchy_removed,
        "hierarchy_corrected": hierarchy_corrected,
    }

