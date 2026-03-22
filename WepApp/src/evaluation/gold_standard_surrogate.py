"""Public-mode surrogate gold standard."""

from __future__ import annotations

from typing import Dict


def build_surrogate() -> Dict:
    return {
        "classes": [{"label": "Patient"}, {"label": "Monitoring"}],
        "relations": [{"label": "hasMonitoring", "domain": "Patient", "range": "Monitoring"}],
        "hierarchy": [],
    }
