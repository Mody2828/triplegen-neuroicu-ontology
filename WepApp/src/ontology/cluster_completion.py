"""LLM-driven cluster-to-ontology reconstruction.

Takes semantically clustered ontology classes and asks an LLM (acting as an
ontology engineer + domain expert) to build a rich ontology for each cluster,
then merges all cluster fragments into a single output.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .merge import merge_ontologies


# ── Noise-relation rule filters ───────────────────────────────────────
#
# The LLM-as-judge qualitative evaluation on the proxy-gold OE run showed
# that ~80% of the revise verdicts on OE-added relations fall into two
# highly patterned buckets:
#
#   4.9  (32/82): relation restates existing hierarchy as a flat edge
#                 (e.g. ``CVP has_monitoring_data Parameter`` where
#                 ``Parameter`` is already a transitive ancestor of
#                 ``CVP`` in the source hierarchy).
#   4.4  (36/82): wrong relation label — judgment call, can't filter.
#
# Layer-1 rule filters below catch the 4.9 cases deterministically plus
# any self-loops the LLM slips through. They run AFTER closed-vocab
# enforcement and BEFORE the merge.

def _norm_noun(label: str) -> str:
    """Lowercase, strip non-alnum, collapse common plurals."""
    s = re.sub(r"[^a-z0-9]+", "", (label or "").lower())
    if s.endswith("ies") and len(s) > 3:
        s = s[:-3] + "y"
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 1:
        s = s[:-1]
    return s


def _build_hierarchy_maps(
    hierarchy: List[Dict[str, Any]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Compute transitive ancestors and descendants for every class.

    Returns ``(ancestors, descendants)`` — both dicts mapping label to set
    of all transitively-related labels. Used by the hierarchy_restate
    filter: if the filler is in the ancestor OR descendant set of the
    domain, the proposed relation duplicates existing subclass structure.

    Descendant coverage is needed to catch reverse-direction re-statements
    like ``Biochemistry hasLabValue Sodium``, where ``Sodium`` is a
    subclass of ``Biochemistry`` in the source.
    """
    parents: Dict[str, List[str]] = {}
    children: Dict[str, List[str]] = {}
    for h in hierarchy or []:
        sub = (h.get("subClass") or h.get("subclass") or h.get("child") or "").strip()
        sup = (h.get("superClass") or h.get("superclass") or h.get("parent") or "").strip()
        if sub and sup:
            parents.setdefault(sub, []).append(sup)
            children.setdefault(sup, []).append(sub)

    all_labels = set(parents.keys()) | set(children.keys())

    def _close(adjacency: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        cache: Dict[str, Set[str]] = {}
        for lbl in all_labels:
            seen: Set[str] = set()
            stack = list(adjacency.get(lbl, []))
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(adjacency.get(n, []))
            cache[lbl] = seen
        return cache

    return _close(parents), _close(children)


def _classify_noise_relation(
    domain: str,
    rel_label: str,
    filler: str,
    ancestor_map: Dict[str, Set[str]],
    descendant_map: Dict[str, Set[str]],
) -> Optional[str]:
    """Return a reason code if the relation is noisy, else None.

    Reason codes:
        ``self_loop``          — domain and filler normalize to the same class.
        ``hierarchy_restate``  — filler is a transitive ancestor OR descendant
                                 of domain in the source hierarchy; the
                                 proposed relation duplicates subclass
                                 structure and will be flagged as 4.9.
    """
    d_norm = _norm_noun(domain)
    f_norm = _norm_noun(filler)
    if d_norm and d_norm == f_norm:
        return "self_loop"

    if filler in ancestor_map.get(domain, set()):
        return "hierarchy_restate"
    if filler in descendant_map.get(domain, set()):
        return "hierarchy_restate"

    return None


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

You are given a semantic cluster of ontology classes extracted from neurointensive care research papers. Your task is to enrich each class with OWL-style existential restrictions (constraints) that describe how it relates to other classes — NOT to invent new classes or restructure the hierarchy.

## Cluster: "{cluster_name}"

### Classes in this cluster ({len(members)} total):
{classes_text}

### Other clusters in this ontology (for cross-cluster linking):
{other_text}

## Your task — produce per-class constraints:

1. **Constraints (existential restrictions)**: For each class, define constraints that describe its relationships to other classes. Each constraint says: this class has *property* **some** *TargetClass*. You MUST use ONLY property labels from the closed vocabulary below — do NOT invent new property types.

   **Allowed property labels (closed vocabulary)**: {vocab_line}

   For example, if SecondaryInsult worsens HeadTrauma, express this as a constraint on SecondaryInsult:
   `{{"property": "Worsens", "filler": "HeadTrauma"}}`

   This means: SecondaryInsult ⊑ ∃Worsens.HeadTrauma (every SecondaryInsult worsens some HeadTrauma).

   If none of the allowed labels fit a class, give it an empty constraints list. It is better to emit fewer constraints than to invent a new property label.

2. **No new classes**: You MUST NOT invent, add, or propose any new classes. Use ONLY the classes listed above in constraints. Any class you reference that is not in the list above (or in another cluster) will be discarded.

3. **No hierarchy**: Do NOT produce subClassOf / hierarchy edges. The hierarchy is already defined in the source ontology and will be preserved independently.

4. **Cross-cluster constraints**: You may add constraints that reference classes from OTHER clusters listed above (up to 5 total). These must also use the closed vocabulary.

## Output format — strict JSON only:
```json
{{
  "classes": [
    {{
      "label": "...",
      "definition": "...",
      "evidence": "from source text",
      "synonyms": [],
      "constraints": [
        {{"property": "...", "filler": "...", "definition": "one-sentence description of this constraint"}}
      ]
    }}
  ],
  "hierarchy": []
}}
```

## Rules:
- Include each existing class in your output EXACTLY as given — do not rename or remove.
- DO NOT add any new classes under any circumstances. Zero. None.
- DO NOT emit any hierarchy edges. Leave "hierarchy" as an empty list.
- Every constraint must use a property from the closed vocabulary above.
- Every constraint filler must reference a real class from this cluster or another cluster listed above.
- Each constraint is scoped to its owning class — the class the constraint is listed under is the domain, and the filler is the range.

## Forbidden patterns (the two most common noise cases — do NOT emit these):

1. **No self-loops.** The filler must refer to a DIFFERENT class than the owning class. Never emit a constraint whose filler is the same class it is attached to. (Bad: `MonitoringData — hasMonitoringData some MonitoringData`.)

2. **No redundant hierarchy re-statement — in either direction.** If the filler is already an ancestor OR a descendant of the owning class in the source hierarchy, the constraint duplicates subclass structure and MUST be dropped. (Bad, when `CVP` is a subclass of `MonitoringData`: `CVP — hasMonitoringData some MonitoringData`.) (Bad, when `Sodium` is a subclass of `Biochemistry`: `Biochemistry — hasLaboratoryValue some Sodium`.) Constraints should describe relationships BETWEEN branches of the hierarchy, not redescribe existing subclass chains.

Valid constraints describe a NON-HIERARCHICAL clinical relationship between classes that are NOT in each other's subclass chain. Examples:
- `SecondaryInsult — worsens some HeadTrauma` (different branches)
- `Therapy — treats some SecondaryInsult` (cross-branch causal/treatment link)
- `GuidelineAdherence — forInsultTreatment some TraumaticBrainInjury`

Output ONLY the JSON object — no markdown fences, no commentary before or after."""


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

    # Precompute transitive ancestor + descendant maps from the source
    # hierarchy. The hierarchy_restate filter uses both directions to drop
    # constraints that just re-state an existing subclass chain as a flat
    # relation (e.g. ``Biochemistry hasLabValue Sodium`` where ``Sodium``
    # is a subclass of ``Biochemistry``).
    ancestor_map, descendant_map = _build_hierarchy_maps(
        merged_ontology.get("hierarchy", [])
    )

    # Aggregate drop counters across all clusters — surfaced in the log so
    # the diagnostic pane can show how much noise the rule filters caught.
    total_noise_self_loop = 0
    total_noise_hierarchy_restate = 0

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
            "prompt": prompt,
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

            # Tag provenance on all surviving items and convert per-class
            # constraints into flat relation dicts for downstream compatibility.
            valid_labels = set(class_lookup.keys())
            kept_relations = []
            dropped_relations_endpoints = 0
            dropped_relations_vocab = 0
            dropped_relations_self_loop = 0
            dropped_relations_hierarchy_restate = 0

            for cls in kept_classes:
                cls.setdefault("provenance", [f"cluster_completion:{cluster_name}"])
                cls["stratum"] = "core"
                if not cls.get("evidence") and cls.get("label") in class_lookup:
                    cls["evidence"] = class_lookup[cls["label"]].get("evidence", "")

                # Convert OWL-style constraints to flat (domain, label, range)
                # relations. Each constraint on a class becomes a relation
                # where domain = this class, range = constraint filler.
                domain_label = (cls.get("label") or "").strip()
                for constraint in cls.pop("constraints", []) or []:
                    filler = (constraint.get("filler") or "").strip()
                    raw_prop = (constraint.get("property") or "").strip()
                    defn = (constraint.get("definition") or "").strip()

                    # Validate endpoints
                    if domain_label not in valid_labels or filler not in valid_labels:
                        dropped_relations_endpoints += 1
                        continue

                    # Closed-vocabulary enforcement: case-insensitive match,
                    # then snap to the source's exact casing so
                    # merge_ontologies dedupes cleanly against the source.
                    canon = rel_label_canonical.get(raw_prop.lower())
                    if not canon:
                        dropped_relations_vocab += 1
                        continue

                    # Judge-informed noise filter: catch self-loops and
                    # redundant hierarchy re-statements (filler is a
                    # transitive ancestor OR descendant of domain, which
                    # means the relation duplicates existing subclass
                    # structure — the 4.9 signature from the qualitative
                    # judge).
                    noise = _classify_noise_relation(
                        domain_label, canon, filler,
                        ancestor_map, descendant_map,
                    )
                    if noise == "self_loop":
                        dropped_relations_self_loop += 1
                        continue
                    if noise == "hierarchy_restate":
                        dropped_relations_hierarchy_restate += 1
                        continue

                    kept_relations.append({
                        "label": canon,
                        "domain": domain_label,
                        "range": filler,
                        "definition": defn,
                        "provenance": [f"cluster_completion:{cluster_name}"],
                        "stratum": "inferred",
                        "evidence": "[inferred] cluster completion",
                    })

            parsed["relations"] = kept_relations

            # Also handle any legacy flat relations the LLM might still emit
            # (e.g. if it ignores the constraint format and falls back to the
            # old domain/range style). This is a safety net, not the main path.
            for rel in parsed.get("_legacy_relations", parsed.get("relations_flat", [])):
                dom = (rel.get("domain") or "").strip()
                rng = (rel.get("range") or "").strip()
                if dom not in valid_labels or rng not in valid_labels:
                    dropped_relations_endpoints += 1
                    continue
                raw_label = (rel.get("label") or "").strip()
                canon = rel_label_canonical.get(raw_label.lower())
                if not canon:
                    dropped_relations_vocab += 1
                    continue
                noise = _classify_noise_relation(
                    dom, canon, rng, ancestor_map, descendant_map,
                )
                if noise == "self_loop":
                    dropped_relations_self_loop += 1
                    continue
                if noise == "hierarchy_restate":
                    dropped_relations_hierarchy_restate += 1
                    continue
                rel["label"] = canon
                rel.setdefault("provenance", [f"cluster_completion:{cluster_name}"])
                rel.setdefault("stratum", "inferred")
                rel.setdefault("evidence", "[inferred] cluster completion")
                kept_relations.append(rel)

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
            log_entries[-1]["dropped_relations_self_loop"] = dropped_relations_self_loop
            log_entries[-1]["dropped_relations_hierarchy_restate"] = (
                dropped_relations_hierarchy_restate
            )
            log_entries[-1]["dropped_relations"] = (
                dropped_relations_endpoints
                + dropped_relations_vocab
                + dropped_relations_self_loop
                + dropped_relations_hierarchy_restate
            )
            log_entries[-1]["dropped_hierarchy"] = dropped_hierarchy

            total_noise_self_loop += dropped_relations_self_loop
            total_noise_hierarchy_restate += dropped_relations_hierarchy_restate

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
            "noise_filter": {
                "self_loop": total_noise_self_loop,
                "hierarchy_restate": total_noise_hierarchy_restate,
                "total": (
                    total_noise_self_loop + total_noise_hierarchy_restate
                ),
            },
        })
    else:
        result = {"classes": [], "relations": [], "hierarchy": [], "metadata": {}}

    return result, log_entries
