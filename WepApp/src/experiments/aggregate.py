"""Aggregate metrics across runs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from ..prompting.parse import safe_json_loads


def load_metrics(run_dir: Path) -> Dict:
    metrics_path = run_dir / "evaluation" / "metrics.json"
    return safe_json_loads(metrics_path.read_text(encoding="utf-8"))


def aggregate_runs(runs_root: str | Path = "runs") -> List[Dict]:
    root = Path(runs_root)
    rows = []
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        try:
            metrics = load_metrics(run_dir)
            rows.append({"run_id": run_dir.name, **metrics})
        except FileNotFoundError:
            continue
    return rows
