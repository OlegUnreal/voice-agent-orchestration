"""Optional Postgres + pgvector memory. SQLite is enough for one process; this is for two Helixes.

Public API (`remember` / `recall` / `list_facts` / `forget`) is backend
agnostic — `helix.vector_store` picks pgvector when `DATABASE_URL` and
`psycopg` are both present and falls back to NumPy brute force otherwise. The
two paths share one ranking function, so switching backends must not change the
answer (`tests/test_pgvector.py` pins that).
"""

from __future__ import annotations

from typing import Any

from helix import vector_store as vs
from helix.vector_store import Fact, backend_kind, blend, get_store, make_fact


def available() -> bool:
    """Kept for `helix.memory`, which only delegates here for the Postgres path."""
    return backend_kind() == "pgvector"


def status() -> dict[str, Any]:
    """What /health reports instead of a vague `available: true`."""
    kind = backend_kind()
    out: dict[str, Any] = {
        "backend": kind,
        "annIndex": "hnsw-vector_cosine_ops" if kind == "pgvector" else "brute-force-numpy",
        "embedBackend": vs.embed_backend_name(),
        "embedDim": vs.EMBED_DIM,
        "databaseUrlSet": bool(vs.database_url()),
        "psycopgInstalled": vs.psycopg_installed(),
    }
    if kind == "pgvector":
        out["migration"] = "migrations/0002_pgvector.sql"
    return out


def remember(key: str, value: str, source: str = "user", ttl_ms: int | None = None) -> dict[str, Any]:
    fact = make_fact(key, value, source=source, ttl_ms=ttl_ms)
    get_store().upsert(fact)
    return fact.to_dict()


def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
    import time as _time

    import numpy as np

    now = int(_time.time() * 1000)
    store = get_store()
    qv = np.asarray(vs.embed(query), dtype=float)
    candidates = store.search(qv, max(k * 4, vs.SEARCH_POOL), now)
    scored = [(fact, blend(query, fact, sim, now)) for fact, sim in candidates]
    return vs.finalize(query, scored, k, now)


def list_facts(limit: int = 40) -> list[dict[str, Any]]:
    return [f.to_dict() for f in get_store().list_facts(limit)]


def forget(key: str) -> int:
    return get_store().delete(key)


__all__ = ["Fact", "available", "explain", "forget", "list_facts", "recall", "remember", "status"]
