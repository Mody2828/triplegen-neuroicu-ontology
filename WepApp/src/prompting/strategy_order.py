from __future__ import annotations

from typing import List, Tuple

STRATEGY_ORDER: List[Tuple[str, str]] = [
    ("baseline", "Zero-Shot"),
    ("one_shot", "One-Shot (MMR-1)"),
    ("phased_3step", "Few-Shot"),
]

# Retained for backward compatibility with old run metadata and comparison registry.
_STRATEGY_ORDER_LEGACY: List[Tuple[str, str]] = [
    ("simple_fewshot", "Few-Shot I (legacy)"),
    ("phased_2step", "Few-Shot II (legacy)"),
]

STRATEGY_IDS: List[str] = [s[0] for s in STRATEGY_ORDER]
STRATEGY_LABELS: dict[str, str] = {s[0]: s[1] for s in STRATEGY_ORDER}


def strategy_number(strategy_id: str) -> int:
    try:
        return STRATEGY_IDS.index(strategy_id) + 1
    except ValueError:
        return 0


def strategy_id_from_number(n: int) -> str | None:
    if 1 <= n <= len(STRATEGY_IDS):
        return STRATEGY_IDS[n - 1]
    return None
