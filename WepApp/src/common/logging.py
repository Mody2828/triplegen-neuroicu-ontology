"""Structured logging helpers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict


def log_event(event: str, payload: Dict[str, Any]) -> str:
    record = {"event": event, "timestamp_utc": datetime.utcnow().isoformat(), **payload}
    return json.dumps(record)
