"""Facts with provenance. The model does not 'remember' — this tool does."""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

from helix.store import data_dir


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "memory.sqlite")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            observed_at INTEGER NOT NULL,
            ttl_ms INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS facts_key ON facts(key)")
    return conn


def remember(key: str, value: str, source: str = "user", ttl_ms: int | None = None) -> dict[str, Any]:
    key = key.strip()[:80] or "note"
    value = value.strip()[:800]
    if not value:
        raise ValueError("empty memory value")
    row = {
        "id": f"mem-{uuid.uuid4().hex[:10]}",
        "key": key,
        "value": value,
        "source": source,
        "observedAt": int(time.time() * 1000),
        "ttlMs": ttl_ms,
    }
    conn = _db()
    try:
        conn.execute(
            "INSERT INTO facts(id, key, value, source, observed_at, ttl_ms) VALUES (?,?,?,?,?,?)",
            (row["id"], row["key"], row["value"], row["source"], row["observedAt"], ttl_ms),
        )
        conn.commit()
    finally:
        conn.close()
    return row


def _alive(observed_at: int, ttl_ms: int | None, now: int) -> bool:
    if ttl_ms is None:
        return True
    return observed_at + ttl_ms >= now


def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
    now = int(time.time() * 1000)
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT id, key, value, source, observed_at, ttl_ms FROM facts ORDER BY observed_at DESC"
        ).fetchall()
    finally:
        conn.close()
    q = {t for t in query.lower().replace("'", "").split() if len(t) > 2}
    scored: list[tuple[int, dict[str, Any]]] = []
    for rid, key, value, source, observed_at, ttl_ms in rows:
        if not _alive(observed_at, ttl_ms, now):
            continue
        blob = f"{key} {value}".lower()
        score = sum(1 for t in q if t in blob) if q else 1
        if not q or score > 0:
            scored.append(
                (
                    score,
                    {
                        "id": rid,
                        "key": key,
                        "value": value,
                        "source": source,
                        "observedAt": observed_at,
                    },
                )
            )
    scored.sort(key=lambda x: (-x[0], -x[1]["observedAt"]))
    return [item for _, item in scored[:k]]


def forget(key: str) -> int:
    conn = _db()
    try:
        cur = conn.execute("DELETE FROM facts WHERE key = ? OR id = ?", (key, key))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def list_facts(limit: int = 40) -> list[dict[str, Any]]:
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT id, key, value, source, observed_at FROM facts ORDER BY observed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "key": r[1], "value": r[2], "source": r[3], "observedAt": r[4]}
        for r in rows
    ]


def parse_remember(query: str) -> tuple[str, str] | None:
    q = query.strip()
    low = q.lower()
    for prefix in ("remember that ", "remember: ", "remember ", "note that "):
        if low.startswith(prefix):
            rest = q[len(prefix) :].strip()
            if not rest:
                return None
            key = rest.split(",")[0].split(" is ")[0][:60]
            return key, rest
    return None
