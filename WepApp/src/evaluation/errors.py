"""Error taxonomy extraction."""

from __future__ import annotations

from typing import Dict, List, Optional


def error_taxonomy(
    unmatched_generated: List[Dict],
    unmatched_gold: List[Dict],
    relations: Optional[List[Dict]] = None,
) -> Dict[str, int]:
    plausible_out_of_ontology = sum(
        1 for u in unmatched_generated if (u.get("evidence") or "").strip()
    )
    implausible_hallucinations = len(unmatched_generated) - plausible_out_of_ontology
    schema_violations = 0
    if relations:
        for rel in relations:
            if not (rel.get("domain") and rel.get("range")):
                schema_violations += 1
    return {
        "hallucinations": implausible_hallucinations,
        "plausible_out_of_ontology": plausible_out_of_ontology,
        "schema_violations": schema_violations,
        "omissions": len(unmatched_gold),
    }
