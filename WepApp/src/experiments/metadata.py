"""Metadata writer for experiment runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_commit_hash() -> str:
    """Return the short git commit hash, or the CODE_VERSION env var, or 'unknown'."""
    env_ver = os.getenv("CODE_VERSION")
    if env_ver:
        return env_ver
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _extract_input_papers(
    docs: Optional[List[Dict[str, Any]]],
    paper_name_override: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build a list of input paper descriptors from ingested docs for traceability.

    When a user pastes text, the corpus is always saved as ``corpus.txt`` — an
    uninformative name.  If the caller supplies a ``paper_name_override`` (from
    the optional "Paper title" form field), that string replaces the filename for
    any doc whose id is the generic sentinel value ``"corpus"``.
    """
    if not docs:
        return []
    papers = []
    seen = set()
    for d in docs:
        source = d.get("source", "")
        stem = d.get("id", "")
        friendly = Path(source).name if source else (stem or "unknown")
        # Apply user-supplied title for pasted text (saved as corpus.txt by the UI).
        if paper_name_override and stem == "corpus":
            friendly = paper_name_override
        key = friendly.lower()
        if key not in seen:
            seen.add(key)
            papers.append({"name": friendly, "stem": stem, "path": source})
    return papers


def build_metadata(
    config: Dict[str, Any],
    run_id: str,
    corpus_snapshot_ids: List[str],
    docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    paper_name = (config.get("paper_name") or "").strip() or None
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "corpus_snapshots": corpus_snapshot_ids,
        "input_papers": _extract_input_papers(docs, paper_name_override=paper_name),
        "config": config,
        "code_version": _git_commit_hash(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_metadata(path: str | Path, metadata: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

