"""Normalization helpers for evaluation."""

from __future__ import annotations

import re


def normalize_label(label: str) -> str:
    """Normalize for display and tokenization; keeps spaces."""
    label = (label or "").lower().strip()
    label = re.sub(r"[^a-z0-9]+", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def normalize_label_for_match(label: str) -> str:
    """Strict form for matching: no spaces, so 'Monitoring Data' and 'MonitoringData' match."""
    n = normalize_label(label)
    return n.replace(" ", "")
