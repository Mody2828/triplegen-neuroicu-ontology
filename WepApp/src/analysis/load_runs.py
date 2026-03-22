"""Load run metadata and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ..prompting.parse import safe_json_loads


@dataclass(frozen=True)
class RunData:
    run_id: str
    root: Path
    metadata: Dict
    metrics: Dict


def load_run(run_path: str | Path) -> RunData:
    root = Path(run_path)
    metadata_path = root / "metadata.json"
    metrics_path = root / "evaluation" / "metrics.json"
    if not root.exists():
        raise FileNotFoundError(f"Run folder not found: {root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in run folder: {root}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation/metrics.json in run folder: {root}")
    metadata = safe_json_loads(metadata_path.read_text(encoding="utf-8"))
    metrics = safe_json_loads(metrics_path.read_text(encoding="utf-8"))
    return RunData(run_id=root.name, root=root, metadata=metadata, metrics=metrics)
