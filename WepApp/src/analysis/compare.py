"""Compare baseline vs improved runs."""

from __future__ import annotations

from typing import Dict


def compare_metrics(baseline: Dict, improved: Dict) -> Dict[str, float]:
    keys = ["coverage", "precision", "recall"]
    deltas = {}
    for key in keys:
        deltas[f"delta_{key}"] = improved.get(key, 0.0) - baseline.get(key, 0.0)
    return deltas
