"""Two interchangeable memory-vector backends over one ranking function.

`helix.pgmem` is the public API; this module owns *how* nearest neighbours are
found. Both backends hand the same `Fact` rows to the same `blend()` /
`finalize()` scorer, so the ANN path and the pure-Python path are not two
different products that happen to look alike — the only difference is where the
top-k search happens:

    PgvectorBackend   psycopg3 + pgvector: `ORDER BY embedding <=> %s::vector`
                      served by the HNSW index (see pgvector_sql.py). Required
                      as soon as two Helix processes must share one memory.
    PythonBackend     NumPy brute force over a JSONL file. No driver, no
                      database, offline, deterministic — the test backend and
                      the single-process default.

`tests/test_pgvector.py` pins that agreement: it drives both backends with one
fixture (the Postgres side through a wire-protocol fake that answers exactly the
SQL this module generates) and asserts byte-identical rankings.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from helix import pgvector_sql as pgs
from helix.embeddings import backend as embed_backend_name
from helix.embeddings import cosine, embed, tokenize
from helix.settings import EMBED_DIM

# Carried over from the pre-pgvector recall(): a small containment bonus so a
# fact that literally names the query token still outranks a semantic near-miss.
LEXICAL_BONUS = 0.05

# Rows pulled from the vector layer before the blend. ANN returns candidates
# ordered by distance; the blend can promote a row that sits just outside the
# vector ordering, so the pool must be wider than `k`.
SEARCH_POOL = 64

FACTS_FILE = "helix_facts.jsonl"


@dataclass
class Fact:
    id: str
    key: str
    value: str
    source: str
    observed_at: int
    ttl_ms: int | None = None
    embedding: list[float] = field(default_factory=list)
    embed_backend: str = "hash"
    embed_dim: int = EMBED_DIM

    @property
    def text(self) -> str:
        return f"{self.key} {self.value}"

    def alive(self, now: int) -> bool:
        return self.ttl_ms is None or self.observed_at + self.ttl_ms >= now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "observedAt": self.observed_at,
            "ttlMs": self.ttl_ms,
        }


def new_id() -> str:
    return f"mem-{uuid.uuid4().hex[:10]}"


def make_fact(key: str, value: str, source: str = "user", ttl_ms: int | None = None, now: int | None = None) -> Fact:
    clean_key = key.strip()[:80] or "note"
    clean_value = value.strip()[:800]
    if not clean_value:
        raise ValueError("empty memory value")
    text = f"{clean_key} {clean_value}"
    vec = embed(text)
    return Fact(
        id=new_id(),
        key=clean_key,
        value=clean_value,
        source=source,
        observed_at=int(now if now is not None else time.time() * 1000),
        ttl_ms=ttl_ms,
        embedding=[float(x) for x in vec],
        embed_backend=embed_backend_name(),
        embed_dim=int(vec.size),
    )


# --------------------------------------------------------------------------------------
# shared scorer — the only place ranking decisions live
# --------------------------------------------------------------------------------------


def blend(query: str, fact: Fact, similarity: float, now: int | None = None) -> dict[str, float]:
    q_tokens = {t for t in tokenize(query)}
    blob = fact.text.lower()
    overlap = sum(1 for t in q_tokens if len(t) > 2 and t in blob)
    lexical = LEXICAL_BONUS * overlap
    return {
        "similarity": float(similarity),
        "lexical": float(lexical),
        "score": float(similarity + lexical),
    }


def finalize(query: str, scored: list[tuple[Fact, dict[str, float]]], k: int, now: int | None = None) -> list[dict[str, Any]]:
    """TTL filter + deterministic order. Shared by both backends verbatim."""
    now = int(now if now is not None else time.time() * 1000)
    kept: list[tuple[float, int, str, dict[str, Any]]] = []
    for fact, parts in scored:
        if not fact.alive(now):
            continue
        row = fact.to_dict()
        row["score"] = round(parts["score"], 12)
        row["breakdown"] = {kk: round(vv, 12) for kk, vv in parts.items()}
        kept.append((-parts["score"], -fact.observed_at, fact.id, row))
    kept.sort(key=lambda x: (x[0], x[1], x[2]))
    return [item[3] for item in kept[:k]]


class VectorBackend:
    """Contract both backends satisfy. `search` returns (fact, similarity) pairs."""

    name = "abstract"

    def upsert(self, fact: Fact) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, limit: int, now: int) -> list[tuple[Fact, float]]:
        raise NotImplementedError

    def list_facts(self, limit: int) -> list[Fact]:
        raise NotImplementedError

    def delete(self, key_or_id: str) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# pure-Python / NumPy fallback
# --------------------------------------------------------------------------------------


class PythonBackend(VectorBackend):
    """Brute-force cosine over a JSONL file. O(n) per query, zero dependencies."""

    name = "python-numpy"

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        from helix.store import data_dir

        return data_dir() / FACTS_FILE

    def _load(self) -> dict[str, Fact]:
        path = self.path
        if not path.exists():
            return {}
        out: dict[str, Fact] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["id"]] = Fact(**row)
        return out

    def _save(self, facts: dict[str, Fact]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(asdict(f), separators=(",", ":")) + "\n" for f in facts.values())
        path.write_text(body, encoding="utf-8", newline="\n")

    def upsert(self, fact: Fact) -> None:
        facts = self._load()
        facts[fact.id] = fact
        self._save(facts)

    def search(self, query_vector: np.ndarray, limit: int, now: int) -> list[tuple[Fact, float]]:
        rows = [f for f in self._load().values() if f.alive(now)]
        scored = [(f, cosine(query_vector, np.asarray(f.embedding, dtype=float))) for f in rows]
        # distance-ASC == similarity-DESC; ties break the same way in SQL.
        scored.sort(key=lambda x: (-x[1], -x[0].observed_at, x[0].id))
        return scored[:limit]

    def list_facts(self, limit: int) -> list[Fact]:
        rows = sorted(self._load().values(), key=lambda f: (-f.observed_at, f.id))
        return rows[:limit]

    def delete(self, key_or_id: str) -> int:
        facts = self._load()
        doomed = [fid for fid, f in facts.items() if f.key == key_or_id or f.id == key_or_id]
        for fid in doomed:
            del facts[fid]
        if doomed:
            self._save(facts)
        return len(doomed)

    def count(self) -> int:
        return len(self._load())


# --------------------------------------------------------------------------------------
# pgvector
# --------------------------------------------------------------------------------------


def connect_factory(database_url: str):
    """Real psycopg3 connection. Imported here and nowhere else, so the driver
    stays optional and `import helix.pgmem` never fails without it."""
    import psycopg

    conn = psycopg.connect(database_url)
    conn.execute(pgs.extension_ddl())
    conn.commit()
    return conn


class PgvectorBackend(VectorBackend):
    """ANN over a `vector(dim)` column, served by the HNSW index.

    The SQL comes from `pgvector_sql` unchanged; this class only binds
    parameters and converts rows. `SET LOCAL hnsw.ef_search` widens the ANN
    candidate list for the over-fetch — the default 40 is enough for k<=40 but
    not for SEARCH_POOL, and silently under-searching is exactly the kind of bug
    that shows up as "recall got worse after we added an index".
    """

    name = "pgvector"

    def __init__(self, database_url: str, connect=None, embed_dim: int = EMBED_DIM) -> None:
        self.database_url = database_url
        self.embed_dim = embed_dim
        self._connect = connect or connect_factory
        self._conn = None

    # -- connection ---------------------------------------------------------------
    def connection(self):
        if self._conn is None:
            self._conn = self._connect(self.database_url)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # -- writes -------------------------------------------------------------------
    def upsert(self, fact: Fact) -> None:
        if fact.embed_dim != self.embed_dim:
            raise ValueError(
                f"fact embedded at {fact.embed_dim}-d but the column is vector({self.embed_dim}); "
                "regenerate migrations/0002_pgvector.sql with `python -m helix.pgvector_sql --write`"
            )
        conn = self.connection()
        conn.execute(
            pgs.upsert_sql(self.embed_dim),
            (
                fact.id,
                fact.key,
                fact.value,
                fact.source,
                fact.observed_at,
                fact.ttl_ms,
                pgs.vector_literal(fact.embedding),
                fact.embed_backend,
                fact.embed_dim,
            ),
        )
        conn.commit()

    def search(self, query_vector: np.ndarray, limit: int, now: int) -> list[tuple[Fact, float]]:
        literal = pgs.vector_literal([float(x) for x in query_vector])
        conn = self.connection()
        conn.execute(pgs.ef_search_stmt(max(pgs.HNSW_EF_SEARCH, limit * 2)))
        rows = conn.execute(pgs.cosine_topk_sql(self.embed_dim), (literal, now, literal, limit)).fetchall()
        conn.commit()
        out: list[tuple[Fact, float]] = []
        for rid, key, value, source, observed_at, ttl_ms, similarity in rows:
            out.append(
                (
                    Fact(
                        id=rid,
                        key=key,
                        value=value,
                        source=source,
                        observed_at=observed_at,
                        ttl_ms=ttl_ms,
                        embedding=[],
                        embed_backend="pgvector",
                        embed_dim=self.embed_dim,
                    ),
                    float(similarity),
                )
            )
        return out

    def list_facts(self, limit: int) -> list[Fact]:
        conn = self.connection()
        try:
            rows = conn.execute(pgs.list_sql(), (limit,)).fetchall()
        finally:
            conn.commit()
        return [
            Fact(id=r[0], key=r[1], value=r[2], source=r[3], observed_at=r[4])
            for r in rows
        ]

    def delete(self, key_or_id: str) -> int:
        conn = self.connection()
        cur = conn.execute(pgs.delete_sql(), (key_or_id, key_or_id))
        conn.commit()
        return int(getattr(cur, "rowcount", 0) or 0)

    def count(self) -> int:
        conn = self.connection()
        try:
            row = conn.execute(pgs.count_sql()).fetchone()
        finally:
            conn.commit()
        return int(row[0]) if row else 0


# --------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------


def database_url() -> str:
    return (os.environ.get("DATABASE_URL") or os.environ.get("HELIX_DATABASE_URL") or "").strip()


def psycopg_installed() -> bool:
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def backend_kind() -> str:
    """`pgvector` when a URL and driver exist, else the NumPy fallback.

    Explicit opt-out (`HELIX_MEMORY_BACKEND=python`) matters: it lets a staging
    box with DATABASE_URL set keep memory local while still exercising the same
    ranking code.
    """
    forced = (os.environ.get("HELIX_MEMORY_BACKEND") or "").strip().lower()
    if forced in {"python", "numpy", "local"}:
        return "python"
    if forced == "pgvector":
        return "pgvector"
    if database_url() and psycopg_installed():
        return "pgvector"
    return "python"


_STORE: VectorBackend | None = None
_STORE_KIND: str | None = None


def get_store(force: str | None = None) -> VectorBackend:
    global _STORE, _STORE_KIND
    kind = force or backend_kind()
    if _STORE is None or _STORE_KIND != kind:
        if kind == "pgvector":
            _STORE = PgvectorBackend(database_url())
        else:
            _STORE = PythonBackend()
        _STORE_KIND = kind
    return _STORE


def reset_store() -> None:
    global _STORE, _STORE_KIND
    store = _STORE
    _STORE = None
    _STORE_KIND = None
    if isinstance(store, PgvectorBackend):
        store.close()


__all__ = [
    "Fact",
    "LEXICAL_BONUS",
    "PgvectorBackend",
    "PythonBackend",
    "SEARCH_POOL",
    "VectorBackend",
    "backend_kind",
    "blend",
    "finalize",
    "get_store",
    "make_fact",
    "reset_store",
]
