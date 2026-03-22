"""Diagnose performance issues and suggest improvements."""

from __future__ import annotations

from typing import Dict, List, Tuple


def diagnose(metrics: Dict) -> List[Tuple[str, str]]:
    coverage = metrics.get("coverage", 0.0)
    precision = metrics.get("precision", 0.0)
    errors = metrics.get("errors", {})
    hallucinations = errors.get("hallucinations", 0)

    actions: List[Tuple[str, str]] = []

    if coverage < 0.3:
        actions.append(
            ("Low coverage", "Increase chunk overlap or add retrieval-based few-shot with better examples.")
        )
    if precision < 0.5:
        actions.append(
            ("Low precision", "Tighten prompt instructions and add schema-guided constraints.")
        )
    if hallucinations > 10:
        actions.append(
            ("High hallucinations", "Add post-filtering and require provenance for each extracted entity.")
        )

    if not actions:
        actions.append(("Healthy metrics", "No critical issues detected; consider minor tuning."))
    return actions
