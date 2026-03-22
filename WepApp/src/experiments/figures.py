"""Generate basic figures for reports/posters."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def plot_coverage(rows: List[Dict], output_path: str | Path) -> None:
    if not rows:
        return
    labels = [r["run_id"] for r in rows]
    values = [r.get("coverage", 0) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel("Coverage")
    plt.title("Gold Standard Coverage by Run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
