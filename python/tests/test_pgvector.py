"""pgvector path without a database: SQL text, DDL, and backend agreement.

The Postgres side is driven by `FakePgServer`, which is *not* a MagicMock. It is
a tiny emulator that (a) rejects SQL it does not recognise, so the backend
cannot quietly invent a query nobody reviewed, and (b) answers the cosine
top-k query with the same distance function pgvector uses. That buys a real
regression test for the ANN path in CI, where no database exists.

Set `HELIX_PG_TESTS=1` + `DATABASE_URL` + `pip install "helix-engines[pg]"` to
also run it against the live cluster (`test_live_pgvector_matches_numpy_ranking`).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

from helix import pgvector_sql as pgs
from helix import vector_store as vs
from helix.embeddings import embed
from helix.pgmem import available, recall, remember, status
from helix.settings import EMBED_DIM

MIGRATION_PATH = Path(__file__).resolve().parents[2] / "migrations" / "0002_pgvector.sql"


def committed_migration() -> str:
    """Migration as checked in, newlines normalized.

    `pgvector_sql.main` writes LF; a Windows checkout with core.autocrlf hands
    back CRLF. Comparing bytes then fails for a reason that has nothing to do
    with the schema, so normalize before asserting the file equals the
    generator's output.
    """
    return MIGRATION_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")


class FakeCursor:
    def __init__(self, rows, rowcount=None):
        self._rows = rows
        self.rowcount = len(rows) if rowcount is None else rowcount

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakePgServer:
    """Answers exactly the statements helix.pgvector_sql emits, nothing else."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.rows: dict[str, tuple] = {}
        self.log: list[tuple[str, tuple]] = []
        self.ef_search: int | None = None
        self.closed = False

    # -- driver surface used by PgvectorBackend ----------------------------------
    def execute(self, sql: str, params: tuple = ()):
        self.log.append((sql, params))
        head = sql.strip().upper()
        if head.startswith("SET LOCAL HNSW.EF_SEARCH"):
            self.ef_search = int(re.search(r"=\s*(\d+)", sql).group(1))
            return FakeCursor([])
        if head.startswith("CREATE EXTENSION"):
            return FakeCursor([])
        if head.startswith("INSERT INTO HELIX_FACTS"):
            rid, key, value, source, observed_at, ttl_ms, literal, backend, edim = params
            assert backend in {"hash", "lsa", "st", "pgvector"}
            assert edim == self.dim, "column is vector(%d)" % self.dim
            assert literal.startswith("[") and literal.endswith("]")
            vec = [float(x) for x in json.loads(literal)]
            assert len(vec) == self.dim, "pgvector rejects a width mismatch at INSERT"
            self.rows[rid] = (rid, key, value, source, observed_at, ttl_ms, vec)
            return FakeCursor([], rowcount=1)
        if head.startswith("SELECT ID, KEY, VALUE, SOURCE, OBSERVED_AT, TTL_MS"):
            literal, now, _literal2, limit = params
            query = np.asarray([float(x) for x in json.loads(literal)], dtype=float)
            scored = []
            for row in self.rows.values():
                rid, key, value, source, observed_at, ttl_ms, vec = row
                if ttl_ms is not None and observed_at + ttl_ms < now:
                    continue  # the WHERE clause filters before ORDER BY
                sim = _cosine_similarity(query, vec)
                scored.append((sim, observed_at, rid, row))
            # ORDER BY embedding <=> query  ==  cosine distance ASC  ==  similarity DESC
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
            out = [
                (row[0], row[1], row[2], row[3], row[4], row[5], sim)
                for sim, _observed, _rid, row in scored[:limit]
            ]
            return FakeCursor(out)
        if head.startswith("SELECT ID, KEY, VALUE, SOURCE, OBSERVED_AT"):
            (limit,) = params
            ordered = sorted(self.rows.values(), key=lambda r: (-r[4], r[0]))[:limit]
            return FakeCursor([(r[0], r[1], r[2], r[3], r[4]) for r in ordered])
        if head.startswith("DELETE FROM HELIX_FACTS"):
            key, _id = params
            doomed = [rid for rid, r in self.rows.items() if r[1] == key or rid == key]
            for rid in doomed:
                del self.rows[rid]
            return FakeCursor([], rowcount=len(doomed))
        if head.startswith("SELECT COUNT(*)"):
            return FakeCursor([(len(self.rows),)])
        raise AssertionError(f"FakePgServer got unreviewed SQL:\n{sql}")

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _cosine_similarity(a: np.ndarray, values: list[float]) -> float:
    b = np.asarray(values, dtype=float)
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


FIXTURE = [
    ("risk cap", "max net risk 1 USDT per position, isolated margin 1x", None),
    ("btc exchange inflow", "spot exchange inflow z-score above 3 means distribution risk", None),
    ("eth staking", "net staking inflows slowed, liquid staking tokens at a discount", None),
    ("nvda earnings", "implied vol crushed after the print, covered call overlay", None),
    ("portfolio book", "default book BTC 40 ETH 25 SOL 15 NVDA 10 AAPL 10", None),
    ("stale note", "this fact is expired and must never be recalled", 1),
]


def fixture_facts(now: int) -> list[vs.Fact]:
    """Built ONCE and upserted into both stores — identical ids, vectors, clocks.

    Ranking equality across backends is meaningless if the two sides invent
    different row ids, so the fixture is data, not a per-store generator.
    """
    facts: list[vs.Fact] = []
    for i, (key, value, ttl_days) in enumerate(FIXTURE):
        ttl = None if ttl_days is None else ttl_days * 86_400_000
        fact = vs.make_fact(key, value, source="test", ttl_ms=ttl, now=now - (i + 1) * 1000)
        if ttl_days is not None:  # already dead at query time
            fact.observed_at = now - 10 * 86_400_000
        facts.append(fact)
    return facts


def _seed(backend: vs.VectorBackend, facts: list[vs.Fact]) -> None:
    for fact in facts:
        backend.upsert(fact)


@pytest.fixture
def now_ms() -> int:
    return 1_757_500_000_000  # fixed clock: 2026-09-09-ish, deterministic TTL filter


def _python_store(tmp_path: Path) -> vs.PythonBackend:
    return vs.PythonBackend(tmp_path / "facts.jsonl")


def _pg_store() -> tuple[vs.PgvectorBackend, FakePgServer]:
    server = FakePgServer()
    return vs.PgvectorBackend("postgresql://fake/helix", connect=lambda _url: server), server


# --------------------------------------------------------------------------- DDL


def test_committed_migration_is_generated_from_code():
    assert MIGRATION_PATH.exists(), "migrations/0002_pgvector.sql missing"
    assert committed_migration() == pgs.migration_sql(EMBED_DIM)


def test_migration_creates_extension_and_typed_column():
    body = committed_migration()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in body
    assert f"embedding       vector({EMBED_DIM}) NOT NULL" in body
    assert "embedding       JSONB" not in body


def test_hnsw_index_ddl_carries_documented_knobs():
    ddl = pgs.hnsw_index_ddl(EMBED_DIM)
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert "m = 16" in ddl and "ef_construction = 64" in ddl
    ivf = pgs.ivfflat_index_ddl(EMBED_DIM)
    assert "USING ivfflat (embedding vector_cosine_ops)" in ivf and "lists = " in ivf
    # ivfflat stays the commented alternative so a deploy never builds two ANN graphs
    assert pgs.ivfflat_index_ddl(EMBED_DIM).split("\n")[0].startswith("CREATE INDEX")
    assert "--   " + pgs.ivfflat_index_ddl(EMBED_DIM).replace("\n", "\n--   ") in pgs.migration_sql(EMBED_DIM)


def test_ttl_partial_index_is_opt_in_and_documented():
    ddl = pgs.ttl_partial_index_ddl(EMBED_DIM)
    assert "WHERE ttl_ms IS NULL" in ddl
    body = pgs.migration_sql(EMBED_DIM)
    assert ddl.splitlines()[-1].startswith("    WHERE"), "the partial index wraps"
    # It ships commented, and *every* line has to carry the marker: a hand
    # commented first line leaves the WHERE clause as live SQL.
    assert pgs.commented_ddl(ddl) in body
    assert "\n    WHERE ttl_ms" not in body
    assert "\n    WHERE ttl_ms" not in committed_migration()
    assert "AND (ttl_ms IS NULL OR observed_at + ttl_ms >= %s)" in pgs.cosine_topk_sql(EMBED_DIM)


def test_dim_follows_the_embedder():
    assert f"vector({EMBED_DIM})" in pgs.table_ddl()
    assert f"vector({384})" in pgs.table_ddl(384)
    with pytest.raises(ValueError):
        pgs.table_ddl(0)


# ---------------------------------------------------------------------------- SQL


def test_ann_query_uses_the_cosine_distance_operator():
    sql = pgs.cosine_topk_sql(EMBED_DIM)
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "1 - (embedding <=> %s::vector) AS similarity" in sql
    assert "LIMIT %s" in sql
    assert "observed_at + ttl_ms >= %s" in sql


def test_recall_issues_only_reviewed_sql_and_widens_ef_search(now_ms):
    store, server = _pg_store()
    _seed(store, fixture_facts(now_ms))
    store.search(embed("risk cap"), vs.SEARCH_POOL, now_ms)
    ann = [sql for sql, _ in server.log if sql.strip().upper().startswith("SELECT ID, KEY")]
    assert ann and "ORDER BY embedding <=> %s::vector" in ann[-1]
    assert server.ef_search is not None and server.ef_search >= vs.SEARCH_POOL
    assert store.count() == len(FIXTURE)
    store.close()
    assert server.closed


# ------------------------------------------------------------- backend agreement


@pytest.mark.parametrize(
    "query",
    [
        "what is my risk cap",
        "BTC exchange inflow spike",
        "staking flows",
        "book weights 40 25 15",
        "expired note",
    ],
)
def test_both_backends_return_identical_rankings(tmp_path, now_ms, query):
    facts = fixture_facts(now_ms)
    py = _python_store(tmp_path)
    pg, _server = _pg_store()
    _seed(py, facts)
    _seed(pg, facts)
    pg.close()

    def scored(store):
        qv = np.asarray(embed(query), dtype=float)
        candidates = store.search(qv, vs.SEARCH_POOL, now_ms)
        pairs = [(f, vs.blend(query, f, sim, now_ms)) for f, sim in candidates]
        return vs.finalize(query, pairs, 5, now_ms)

    a, b = scored(py), scored(pg)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [(r["key"], r["score"]) for r in a] == [(r["key"], pytest.approx(r["score"], abs=1e-9)) for r in b]
    assert all("stale note" != r["key"] for r in a), "TTL-expired fact leaked into recall"
    assert a and set(a[0]["breakdown"]) == {"similarity", "lexical", "score"}


def test_pgmem_falls_back_to_the_numpy_backend_without_a_driver(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("HELIX_DATABASE_URL", raising=False)
    monkeypatch.delenv("HELIX_MEMORY_BACKEND", raising=False)
    monkeypatch.setenv("HELIX_DATA_DIR", str(tmp_path))
    vs.reset_store()
    assert available() is False
    row = remember("risk cap", "isolated 1x and 1 USDT max net risk")
    remember("lunch", "ate a sandwich at noon")
    hits = recall("what is my risk cap", k=3)
    assert hits[0]["id"] == row["id"], "a paraphrased query must still find the cap"
    assert [h["key"] for h in hits] == ["risk cap", "lunch"], "cosine has to beat an unrelated fact"
    assert hits[0]["breakdown"]["lexical"] > 0
    st = status()
    # psycopg is absent in CI, so the fallback answers — and status() says so.
    # This assertion is what keeps /health from advertising ANN it does not have.
    assert st["backend"] == "python"
    assert st["annIndex"] == "brute-force-numpy"
    assert st["embedDim"] == EMBED_DIM and st["psycopgInstalled"] is False
    assert "migration" not in st
    assert vs.PythonBackend(tmp_path / vs.FACTS_FILE).count() == 2
    vs.reset_store()


def test_status_names_the_ann_index_when_pgvector_is_selected(monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_BACKEND", "pgvector")
    monkeypatch.setenv("DATABASE_URL", "postgresql://someone@box/helix")
    vs.reset_store()
    try:
        st = status()
        assert st["backend"] == "pgvector"
        assert st["annIndex"] == "hnsw-vector_cosine_ops"
        assert st["migration"] == "migrations/0002_pgvector.sql"
        assert available() is True
    finally:
        monkeypatch.delenv("HELIX_MEMORY_BACKEND", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        vs.reset_store()


def test_pgmem_refuses_to_mix_embedding_widths(tmp_path):
    store, _server = _pg_store()
    fact = vs.make_fact("k", "v")
    fact.embed_dim = EMBED_DIM + 8
    with pytest.raises(ValueError, match="regenerate migrations/0002_pgvector.sql"):
        store.upsert(fact)


def test_backend_selection_is_forced_by_env(monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_BACKEND", "python")
    monkeypatch.setenv("DATABASE_URL", "postgresql://someone@box/helix")
    assert vs.backend_kind() == "python"
    monkeypatch.setenv("HELIX_MEMORY_BACKEND", "pgvector")
    assert vs.backend_kind() == "pgvector"
    monkeypatch.delenv("HELIX_MEMORY_BACKEND")
    monkeypatch.setattr(vs, "psycopg_installed", lambda: False)
    assert vs.backend_kind() == "python"


def test_list_and_delete_round_trip_through_the_fake_server(now_ms):
    store, _server = _pg_store()
    _seed(store, fixture_facts(now_ms))
    facts = store.list_facts(10)
    assert len(facts) == len(FIXTURE)
    assert facts[0].observed_at >= facts[-1].observed_at
    assert store.delete("eth staking") == 1
    assert store.count() == len(FIXTURE) - 1


# ------------------------------------------------------------------- live option


@pytest.mark.skipif(
    os.environ.get("HELIX_PG_TESTS") != "1" or not vs.psycopg_installed() or not vs.database_url(),
    reason="live pgvector test needs HELIX_PG_TESTS=1, DATABASE_URL and pip install 'helix-engines[pg]'",
)
def test_live_pgvector_matches_numpy_ranking(tmp_path, now_ms):  # pragma: no cover - opt-in
    conn = vs.connect_factory(vs.database_url())
    conn.execute(pgs.table_ddl(EMBED_DIM))
    conn.execute(pgs.hnsw_index_ddl(EMBED_DIM))
    conn.commit()
    conn.close()
    live = vs.PgvectorBackend(vs.database_url())
    py = _python_store(tmp_path)
    facts = fixture_facts(now_ms)
    _seed(live, facts)
    _seed(py, facts)
    try:
        for query in ("risk cap", "BTC exchange inflow"):
            a = [r["id"] for r in vs.finalize(query, [
                (f, vs.blend(query, f, s, now_ms)) for f, s in live.search(embed(query), vs.SEARCH_POOL, now_ms)
            ], 5, now_ms)]
            b = [r["id"] for r in vs.finalize(query, [
                (f, vs.blend(query, f, s, now_ms)) for f, s in py.search(embed(query), vs.SEARCH_POOL, now_ms)
            ], 5, now_ms)]
            assert a == b
            live.delete(FIXTURE[0][0])
    finally:
        live.close()


def _statements(body: str) -> list[str]:
    live = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("--"))
    return [s.strip() for s in live.split(";") if s.strip()]


def test_migration_is_idempotent_sql():
    body = pgs.migration_sql(EMBED_DIM)
    statements = _statements(body)
    assert body.rstrip().endswith(";"), "final statement is unterminated"
    assert len(statements) == 5, statements  # extension, table, hnsw, observed_at, key
    for stmt in statements:
        assert stmt.upper().startswith("CREATE "), stmt
        assert "IF NOT EXISTS" in stmt, "re-applying a deploy migration must be a no-op"
    assert "DROP TABLE" not in body and "TRUNCATE" not in body
    assert any("USING hnsw" in s for s in statements)
