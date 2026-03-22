from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..prompting.parse import safe_json_loads
from ..prompting.vocabulary import PROVENANCE_KEYWORDS


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_retrieval_pool_concepts(path: Path | None = None) -> List[Dict]:
    """Task-specific pool for class/concept extraction examples."""
    if path is None:
        path = _project_root() / "resources" / "pool_strict_concepts.json"
    return _load_pool(path)


def load_retrieval_pool_relations(path: Path | None = None) -> List[Dict]:
    """Task-specific pool for relation extraction examples."""
    if path is None:
        path = _project_root() / "resources" / "pool_strict_relations.json"
    return _load_pool(path)


def load_retrieval_pool_hierarchy(path: Path | None = None) -> List[Dict]:
    """Task-specific pool for hierarchy extraction examples."""
    if path is None:
        path = _project_root() / "resources" / "pool_strict_hierarchy.json"
    return _load_pool(path)


def load_retrieval_pool_one_shot(path: Path | None = None) -> List[Dict]:
    """Comprehensive pool for one-shot extraction (classes + relations + hierarchy in each example)."""
    if path is None:
        path = _project_root() / "resources" / "pool_one_shot_comprehensive.json"
    return _load_pool(path)


def _load_pool(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    try:
        out = safe_json_loads(raw)
        return out if isinstance(out, list) else []
    except (ValueError, json.JSONDecodeError):
        return []


def is_provenance_input(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in PROVENANCE_KEYWORDS)


def pool_texts(pool: List[Dict]) -> List[str]:
    return [p["text"] for p in pool]


def format_pool_example(entry: Dict, include_output: bool = True) -> str:
    lines = [f"Text: {entry['text']}"]
    if include_output and entry.get("output"):
        out = entry["output"]
        if isinstance(out, dict):
            out = json.dumps(out, indent=2)
        lines.append(f"Output (JSON):\n{out}")
    return "\n".join(lines)
