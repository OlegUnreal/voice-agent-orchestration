"""Append-only traces, preference pairs, verify fails — the training flywheel."""

from __future__ import annotations

import time
import uuid
from typing import Any

from helix.store import append_jsonl, read_jsonl

TRACES = "traces.jsonl"
PREFS = "preferences.jsonl"
FAILS = "verify_fails.jsonl"


def new_id(prefix: str = "tr") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def log_turn(row: dict[str, Any]) -> str:
    tid = row.get("id") or new_id()
    row = {**row, "id": tid, "ts": row.get("ts") or int(time.time() * 1000)}
    append_jsonl(TRACES, row)
    return tid


def log_verify_fail(row: dict[str, Any]) -> None:
    append_jsonl(FAILS, {**row, "id": row.get("id") or new_id("vf"), "ts": int(time.time() * 1000)})


def log_feedback(
    trace_id: str,
    verdict: str,
    spoken: str,
    query: str,
    correction: str | None = None,
    tools: list[str] | None = None,
) -> dict[str, Any]:
    rec = {
        "id": new_id("fb"),
        "ts": int(time.time() * 1000),
        "traceId": trace_id,
        "verdict": verdict,
        "query": query,
        "spoken": spoken,
        "correction": (correction or "").strip() or None,
        "tools": tools or [],
    }
    append_jsonl(PREFS, rec)
    if verdict == "down" and rec["correction"]:
        append_jsonl(
            PREFS,
            {
                "id": new_id("dpo"),
                "ts": rec["ts"],
                "traceId": trace_id,
                "kind": "dpo",
                "prompt": query,
                "rejected": spoken,
                "chosen": rec["correction"],
            },
        )
    elif verdict == "up":
        append_jsonl(
            PREFS,
            {
                "id": new_id("sft"),
                "ts": rec["ts"],
                "traceId": trace_id,
                "kind": "sft",
                "prompt": query,
                "chosen": spoken,
                "rejected": None,
            },
        )
    return rec


def list_traces(limit: int = 80) -> list[dict[str, Any]]:
    rows = read_jsonl(TRACES)
    return list(reversed(rows[-limit:]))


def export_dataset() -> dict[str, Any]:
    traces = read_jsonl(TRACES)
    prefs = read_jsonl(PREFS)
    fails = read_jsonl(FAILS)
    sft = [p for p in prefs if p.get("kind") == "sft"]
    dpo = [p for p in prefs if p.get("kind") == "dpo"]
    return {
        "traces": len(traces),
        "preferences": len(prefs),
        "verifyFails": len(fails),
        "sftPairs": len(sft),
        "dpoPairs": len(dpo),
        "sft": sft[-50:],
        "dpo": dpo[-50:],
    }
