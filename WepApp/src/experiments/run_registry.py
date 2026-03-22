"""Run registry utilities for consistent output folders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    generated: Path
    evaluation: Path
    figures: Path
    prompts: Path


def create_run_dirs(base_dir: str | Path = "runs", run_id: Optional[str] = None) -> RunPaths:
    if run_id is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{uuid4().hex[:8]}"
    root = (Path(base_dir) / run_id).resolve()
    generated = root / "generated"
    evaluation = root / "evaluation"
    figures = root / "figures"
    prompts = root / "prompts"
    for path in (generated, evaluation, figures, prompts):
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=run_id, root=root, generated=generated, evaluation=evaluation, figures=figures, prompts=prompts)

