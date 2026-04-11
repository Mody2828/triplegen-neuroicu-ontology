"""LLM-driven cluster-to-ontology reconstruction.

Takes semantically clustered ontology classes and asks an LLM (acting as an
ontology engineer + domain expert) to build a rich ontology for each cluster,
then merges all cluster fragments into a single output.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from .merge import merge_ontologies


# ── Prompt construction ───────────────────────────────────────────────

def _build_cluster_prompt(cluster: Dict[str, Any],
                          class_lookup: Dict[str, Dict],
                          all_cluster_names: List[str],
                          allowed_relations: Optional[List[str]] = None) -> str:
    """Build the LLM prompt for a single cluster.

    Args:
        cluster: Dict with keys ``name``, ``members`` (list of class labels).
        class_lookup: label -> full class dict (definition, evidence, synonyms).
        all_cluster_names: Names of ALL clusters (for cross-cluster linking context).
        allowed_relations: Closed vocabulary of relation labels the LLM is
            permitted to use. Harvested from the source ontology so the LLM
            can't invent new relation types that drift from what extraction
            produced. If None or empty, a small default list is used.
    """
    cluster_name = cluster["name"]
    members = cluster["members"]

    # Build class listing
    class_lines = []
    for i, label in enumerate(members, 1):
        info = class_lookup.get(label, {})
        defn = (info.get("definition") or "").strip()
        ev = (info.get("evidence") or "").strip()
        syns = ", ".join(info.get("synonyms") or [])
        parts = [f"{i}. **{label}**"]
        if defn:
            parts.append(f"   Definition: {defn}")
        if ev:
            parts.append(f"   Evidence: {ev[:300]}")
        if syns:
            parts.append(f"   Synonyms: {syns}")
        class_lines.append("\n".join(parts))

    classes_text = "\n".join(class_lines)

    # Other clusters for cross-linking context
    other_clusters = [n for n in all_cluster_names if n != cluster_name]
    other_text = ", ".join(other_clusters) if other_clusters else "(none)"

    # Closed relation vocabulary — show it inline so the LLM sees the exact
    # list of allowed labels and is forced to reuse existing source relations
    # rather than invent new ones (hasComponent, indicatesRiskOf, etc).
    if allowed_relations:
        vocab_line = ", ".join(f'"{r}"' for r in allowed_relations)
    else:
        vocab_line = '"monitors", "treats", "measures", "indicates", "causedBy", "administeredTo", "records", "hasOutcome"'

    return f"""You are an expert ontology engineer and neurointensive care domain specialist.

You are given a semantic cluster of ontology classes extracted from neurointensive care research papers. Your task is to enrich this cluster with meaningful domain relations — NOT to invent new classes or restructure the hierarchy.

## Cluster: "{cluster_name}"

### Classes in this cluster ({len(members)} total):
{classes_text}

### Other clusters in this ontology (for cross-cluster linking):
{other_text}

## Your task — produce ONLY relations:

1. **Relations (object properties)**: Define meaningful domain-specific relations between classes in this cluster. You MUST use ONLY labels from the closed vocabulary below — do NOT invent new relation types.

   **Allowed relation labels (closed vocabulary)**: {vocab_line}

   If none of the allowed labels fit a pair of classes, do not emit a relation for that pair. It is better to emit fewer relations than to invent a new label.

2. **No new classes**: You MUST NOT invent, add, or propose any new classes. Use ONLY the classes listed above as domain/range. Any class you output that is not in the list above will be discarded.

3. **No hierarchy**: Do NOT produce subClassOf / hierarchy edges. The hierarchy is already defined in the source ontology and will be preserved independently. Any hierarchy edges you emit will be discarded.

4. **Cross-cluster relations**: Identify up to 5 important relations that connect classes in THIS cluster to classes that likely belong in OTHER clusters listed above. Cross-cluster relations must also use the closed vocabulary above.

## Output format — strict JSON only:
```json
{{
  "classes": [
    {{"label": "...", "definition": "...", "evidence": "from source text", "synonyms": []}}
  ],
  "relations": [
    {{"label": "...", "domain": "...", "range": "...", "definition": "one-sentence description"}}
  ],
  "hierarchy": []
}}
```

## Rules:
- Include each existing class in your output EXACTLY as given — do not rename or remove.
- DO NOT add any new classes under any circumstances. Zero. None.
- DO NOT emit any hierarchy edges. Leave "hierarchy" as an empty list.
- Every relation must use a label from the closed vocabulary above.
- Every relation must have both domain and range that reference real classes from this cluster (or another cluster listed above).
- Output ONLY the JSON object — no markdown fences, no commentary before or after."""


# ── JSON extraction from LLM response ────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")
_BARE_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[Dict]:
    """Best-effort extraction of a JSON object from LLM output."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try fenced code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # Try bare JSON
    m = _BARE_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ── Main pipeline ─────────────────────────────────────────────────────

def run_cluster_completion(
    merged_ontology: Dict[str, Any],
    cluster_data: Dict[str, Any],
    provider: str,
    model: str | None = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run LLM cluster-to-ontology completion for all clusters.

    Args:
        merged_ontology: The merged ontology dict (from ``merge.py``).
        cluster_data: Output of ``cluster_classes()`` — must have ``clusters`` key.
        provider: LLM provider name (e.g. "openai", "anthropic").
        model: Optional model override.
        progress_callback: ``(current, total, message)`` callback for progress updates.

    Returns:
        Reconstructed ontology dict (same schema as ``ontology.json``).
    """
    from src.prompting.llm_client import LLMClient

    client = LLMClient(provider=provider, model=model or None)
    clusters = cluster_data.get("clusters", [])
    total = len(clusters)

    # Build class lookup from merged ontology for enriching prompts
    class_lookup: Dict[str, Dict] = {}
    for c in merged_ontology.get("classes", []):
        class_lookup[c.get("label", "")] = c

    all_cluster_names = [c.get("name", f"Cluster {c.get('id', '?')}") for c in clusters]

    # Harvest the closed relation vocabulary from the source ontology. This
    # is the single biggest precision lever on relations — the LLM's
    # unmatched inventions (hasComponent, indicatesRiskOf, measures, ...) all
    # come from it freelancing outside the source vocabulary. By passing the
    # exact set of source relation labels into the prompt AND enforcing them
    # post-parse, we pin the reconstruction to labels that evaluation
    # already knows how to match. Preserve the exact casing the source used.
    source_rel_labels: List[str] = []
    seen_rel_lower: set = set()
    for rel in merged_ontology.get("relations", []):
        lbl = (rel.get("label") or "").strip()
        if not lbl:
            continue
        key = lbl.lower()
        if key in seen_rel_lower:
            continue
        seen_rel_lower.add(key)
        source_rel_labels.append(lbl)
    # Lookup for case-insensitive match → preserve source casing on emission
    rel_label_canonical = {lbl.lower(): lbl for lbl in source_rel_labels}

    # Process each cluster
    cluster_fragments: List[Dict[str, Any]] = []
    log_entries: List[Dict[str, Any]] = []

    for i, cluster in enumerate(clusters):
        cluster_name = cluster.get("name", f"Cluster {cluster.get('id', i)}")
        if progress_callback:
            progress_callback(i, total, f"Processing cluster {i + 1}/{total}: {cluster_name}")

        prompt = _build_cluster_prompt(
            cluster,
            class_lookup,
            all_cluster_names,
            allowed_relations=source_rel_labels,
        )

        try:
            response = client.generate(prompt, max_tokens=4096)
            parsed = _extract_json(response)
        except Exception as e:
            parsed = None
            response = f"[ERROR] {e}"

        log_entries.append({
            "cluster_id": cluster.get("id", i),
            "cluster_name": cluster_name,
            "members": cluster.get("members", []),
            "prompt_length": len(prompt),
            "response_length": len(response) if response else 0,
            "parsed_ok": parsed is not None,
            "response_preview": (response or "")[:500],
        })

        if parsed:
            # Hard enforcement: drop any class that is NOT a verbatim member
            # of the original cluster. The LLM is told not to add classes, but
            # we enforce it deterministically here regardless of what it did.
            # This is the single biggest precision lever — adding classes
            # against a small gold standard always hurts precision.
            raw_classes = parsed.get("classes", [])
            member_set = set(cluster.get("members", []))
            kept_classes = []
            dropped_inferred = 0
            for cls in raw_classes:
                lbl = (cls.get("label") or "").strip()
                # Also strip any legacy "[inferred] " prefix so we can match
                # labels the LLM tried to smuggle in under that pattern.
                stripped = lbl
                if stripped.lower().startswith("[inferred]"):
                    stripped = stripped[len("[inferred]"):].strip()
                    cls["label"] = stripped
                if stripped in member_set or stripped in class_lookup:
                    kept_classes.append(cls)
                else:
                    dropped_inferred += 1
            parsed["classes"] = kept_classes

            # Record the drop count on the log entry for this cluster so the
            # diagnostic pane can show how aggressive the filter was.
            log_entries[-1]["dropped_inferred_classes"] = dropped_inferred
            log_entries[-1]["kept_classes"] = len(kept_classes)

            # Tag provenance on all surviving items
            for cls in kept_classes:
                cls.setdefault("provenance", [f"cluster_completion:{cluster_name}"])
                cls["stratum"] = "core"
                if not cls.get("evidence") and cls.get("label") in class_lookup:
                    cls["evidence"] = class_lookup[cls["label"]].get("evidence", "")

            # Valid label set for relation/hierarchy endpoints = every class
            # in the full source ontology. Any edge referencing a label not
            # in this set was pointing at a class we just dropped (or one the
            # LLM invented) and must be removed to avoid dangling edges.
            valid_labels = set(class_lookup.keys())

            kept_relations = []
            dropped_relations_endpoints = 0
            dropped_relations_vocab = 0
            for rel in parsed.get("relations", []):
                dom = (rel.get("domain") or "").strip()
                rng = (rel.get("range") or "").strip()
                if dom not in valid_labels or rng not in valid_labels:
                    dropped_relations_endpoints += 1
                    continue
                # Closed-vocabulary enforcement: the prompt tells the LLM
                # which labels are allowed, but we enforce it deterministically
                # here too. Case-insensitive match, then snap to the source's
                # exact casing so merge_ontologies dedupes cleanly against
                # the source fragment.
                raw_label = (rel.get("label") or "").strip()
                canon = rel_label_canonical.get(raw_label.lower())
                if not canon:
                    dropped_relations_vocab += 1
                    continue
                rel["label"] = canon
                rel.setdefault("provenance", [f"cluster_completion:{cluster_name}"])
                rel.setdefault("stratum", "inferred")
                rel.setdefault("evidence", "[inferred] cluster completion")
                kept_relations.append(rel)
            parsed["relations"] = kept_relations

            # Hierarchy is now fully delegated to the source seed. Any edges
            # the LLM emits despite the prompt instruction are discarded
            # wholesale — the source ontology's hierarchy is already richer
            # and better-grounded than anything the per-cluster LLM view can
            # produce, and LLM hierarchy was dragging overall F1 down by
            # ~0.05 on test runs. Count what we drop so the diagnostic pane
            # can show it.
            dropped_hierarchy = len(parsed.get("hierarchy", []) or [])
            parsed["hierarchy"] = []

            log_entries[-1]["dropped_relations_endpoints"] = dropped_relations_endpoints
            log_entries[-1]["dropped_relations_vocab"] = dropped_relations_vocab
            log_entries[-1]["dropped_relations"] = (
                dropped_relations_endpoints + dropped_relations_vocab
            )
            log_entries[-1]["dropped_hierarchy"] = dropped_hierarchy

            cluster_fragments.append(parsed)

    if progress_callback:
        progress_callback(total, total, "Merging cluster fragments into final ontology...")

    # Merge the ORIGINAL source ontology in as the first fragment, followed
    # by the LLM cluster fragments. merge_ontologies dedupes by label, so
    # anything the LLM preserved collapses naturally against the source copy.
    # The payoff: any class, hierarchy edge, or relation the LLM forgot to
    # emit is rescued from the source; the LLM can only ADD gold-aligned
    # structure, never delete it. This also reconnects the disconnected
    # cluster islands because the source's cross-cluster hierarchy edges
    # survive intact.
    source_fragment = {
        "classes": merged_ontology.get("classes", []),
        "relations": merged_ontology.get("relations", []),
        "hierarchy": merged_ontology.get("hierarchy", []),
    }
    all_fragments = [source_fragment] + cluster_fragments

    if all_fragments:
        result = merge_ontologies(all_fragments, metadata={
            "method": "cluster_completion",
            "provider": provider,
            "model": model or "",
            "n_clusters": total,
            "n_successful": len(cluster_fragments),
            "source_seeded": True,
            "closed_relation_vocab": True,
            "n_allowed_relations": len(source_rel_labels),
            "hierarchy_from_source_only": True,
        })
    else:
        result = {"classes": [], "relations": [], "hierarchy": [], "metadata": {}}

    return result, log_entries
