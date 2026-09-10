"""Teacher log: model draft vs verifier-final. Distillation pairs when they differ."""

from __future__ import annotations

import time
from typing import Any

from helix.ledger import new_id
from helix.sanitize import scrub_obj
from helix.store import append_jsonl, read_jsonl

TEACHERS = "teachers.jsonl"
PREFS = "preferences.jsonl"


def log_teacher(
    trace_id: str | None,
    query: str,
    draft: str,
    final: str,
    model: str,
    leaks: list[str] | None = None,
) -> dict[str, Any]:
    held = (draft or "").strip() != (final or "").strip()
    rec = scrub_obj(
        {
            "id": new_id("td"),
            "ts": int(time.time() * 1000),
            "traceId": trace_id,
            "query": query,
            "draftSpoken": draft,
            "finalSpoken": final,
            "model": model,
            "held": held,
            "leaks": leaks or [],
        }
    )
    append_jsonl(TEACHERS, rec)
    if held and rec["draftSpoken"] and rec["finalSpoken"]:
        append_jsonl(
            PREFS,
            scrub_obj(
                {
                    "id": new_id("distill"),
                    "ts": rec["ts"],
                    "traceId": trace_id,
                    "kind": "distill",
                    "prompt": query,
                    "rejected": rec["draftSpoken"],
                    "chosen": rec["finalSpoken"],
                }
            ),
        )
    return rec


def list_teachers(limit: int = 50) -> list[dict[str, Any]]:
    return list(reversed(read_jsonl(TEACHERS)[-limit:]))
