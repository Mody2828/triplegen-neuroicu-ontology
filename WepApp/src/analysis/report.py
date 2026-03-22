"""Write analysis reports and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from .compare import compare_metrics
from .diagnose import diagnose


def write_comparison(run_root: Path, baseline: Dict, improved: Dict) -> None:
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    deltas = compare_metrics(baseline, improved)
    rows = [
        f"coverage,{baseline.get('coverage', 0.0)},{improved.get('coverage', 0.0)},{deltas['delta_coverage']}",
        f"precision,{baseline.get('precision', 0.0)},{improved.get('precision', 0.0)},{deltas['delta_precision']}",
        f"recall,{baseline.get('recall', 0.0)},{improved.get('recall', 0.0)},{deltas['delta_recall']}",
    ]
    (analysis_dir / "comparison.csv").write_text(
        "metric,baseline,improved,delta\n" + "\n".join(rows), encoding="utf-8"
    )

    actions = diagnose(improved)
    summary_lines = ["Performance diagnosis:"]
    for title, action in actions:
        summary_lines.append(f"- {title}: {action}")
    (analysis_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def write_plot(run_root: Path, baseline: Dict, improved: Dict) -> None:
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    labels = ["coverage", "precision", "recall"]
    baseline_vals = [baseline.get(k, 0.0) for k in labels]
    improved_vals = [improved.get(k, 0.0) for k in labels]

    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar([i - 0.2 for i in x], baseline_vals, width=0.4, label="baseline")
    plt.bar([i + 0.2 for i in x], improved_vals, width=0.4, label="improved")
    plt.xticks(list(x), labels)
    plt.ylabel("Score")
    plt.title("Baseline vs Improved")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "comparison.png")
    plt.close()
