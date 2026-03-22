"""Write failure summaries for runs."""

from __future__ import annotations

from pathlib import Path
from typing import List


def write_failure_summary(run_root: Path, errors: List[str]) -> None:
    if not errors:
        return
    path = run_root / "failure_summary.txt"
    path.write_text("\n".join(errors), encoding="utf-8")
