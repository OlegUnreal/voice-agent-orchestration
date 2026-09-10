"""Strip secrets before a turn becomes training data."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(sk-|rk-|xai-|api[_-]?key['\"]?\s*[:=]\s*)[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9._-]+"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b0x[a-fA-F0-9]{40,}\b"),
]


def scrub_text(text: str | None) -> str:
    if not text:
        return ""
    out = text
    for pat in _PATTERNS:
        out = pat.sub("[redacted]", out)
    return out


def scrub_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, list):
        return [scrub_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    return obj
