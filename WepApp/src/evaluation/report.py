"""Evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def write_metrics(path: str | Path, metrics: Dict) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_table(path: str | Path, rows: List[Dict]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[h]) for h in headers))
    Path(path).write_text("\n".join(lines), encoding="utf-8")
