"""Optional Postgres + pgvector memory. SQLite is enough for one process; this is for two Helixes."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from helix.embeddings import embed


def database_url() -> str:
    return (os.environ.get("DATABASE_URL") or os.environ.get("HELIX_DATABASE_URL") or "").strip()


def available() -> bool:
    if not database_url():
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def _connect():
    import psycopg

    conn = psycopg.connect(database_url())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS helix_facts (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            observed_at BIGINT NOT NULL,
            ttl_ms BIGINT,
            embedding JSONB
        )
        """
    )
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception:
        conn.rollback()
    conn.commit()
    return conn


def remember(key: str, value: str, source: str = "user", ttl_ms: int | None = None) -> dict[str, Any]:
    row = {
        "id": f"mem-{uuid.uuid4().hex[:10]}",
        "key": key.strip()[:80] or "note",
        "value": value.strip()[:800],
        "source": source,
        "observedAt": int(time.time() * 1000),
        "ttlMs": ttl_ms,
    }
    vec = embed(f"{row['key']} {row['value']}").tolist()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO helix_facts(id, key, value, source, observed_at, ttl_ms, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (row["id"], row["key"], row["value"], row["source"], row["observedAt"], ttl_ms, json.dumps(vec)),
        )
        conn.commit()
    finally:
        conn.close()
    return row


def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
    qv = embed(query)
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, key, value, source, observed_at, ttl_ms, embedding FROM helix_facts ORDER BY observed_at DESC"
        ).fetchall()
    finally:
        conn.close()
    now = int(time.time() * 1000)
    scored: list[tuple[float, dict[str, Any]]] = []
    for rid, key, value, source, observed_at, ttl_ms, embedding in rows:
        if ttl_ms is not None and observed_at + ttl_ms < now:
            continue
        vec = embedding if isinstance(embedding, list) else json.loads(embedding or "[]")
        score = 0.0
        if vec and len(vec) == len(qv):
            score = float(sum(a * b for a, b in zip(qv, vec, strict=False)))
        blob = f"{key} {value}".lower()
        score += 0.05 * sum(1 for t in query.lower().split() if len(t) > 2 and t in blob)
        scored.append(
            (score, {"id": rid, "key": key, "value": value, "source": source, "observedAt": observed_at})
        )
    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:k]]


def list_facts(limit: int = 40) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, key, value, source, observed_at FROM helix_facts ORDER BY observed_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "key": r[1], "value": r[2], "source": r[3], "observedAt": r[4]} for r in rows]


def forget(key: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM helix_facts WHERE key = %s OR id = %s", (key, key))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()
