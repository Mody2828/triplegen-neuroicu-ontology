"""Load and render JSON prompt templates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

# Replace control characters so JSON with accidental unescaped newlines/tabs still loads
_CONTROL_CHAR = re.compile(r"[\x00-\x1f]")


def _safe_json_loads(text: str, *, source: str = "") -> Dict:
    """Parse JSON after replacing control characters to avoid Invalid control character errors."""
    if not text or not text.strip():
        return {}
    cleaned = _CONTROL_CHAR.sub(" ", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in template {source}: {exc}") from exc


def load_template(name: str) -> Dict:
    base = Path(__file__).parent / "templates" / f"{name}.json"
    if not base.exists():
        raise FileNotFoundError(f"Prompt template not found: {base}")
    return _safe_json_loads(base.read_text(encoding="utf-8"), source=str(base))


def render_template(template: Dict, *, text: str, examples: str = "") -> str:
    formatter = template.get("formatter", "{text}")
    # Escape braces in user-controlled text to prevent KeyError/IndexError
    # from .format() when clinical text contains {braces} (e.g. "{n=42}").
    safe_text = text.replace("{", "{{").replace("}", "}}")
    safe_examples = examples.replace("{", "{{").replace("}", "}}")
    body = formatter.format(text=safe_text, examples=safe_examples)
    parts = [template.get("prefix", "").strip(), body.strip(), template.get("suffix", "").strip()]
    return "\n\n".join(p for p in parts if p)
