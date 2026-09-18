"""pgvector DDL and SQL builders — plain strings, no driver import.

Why a module of strings: everything interesting about the pgvector backend lives
in SQL text (`ORDER BY embedding <=> %s::vector`, the HNSW `WITH (...)` knobs,
the partial-index predicate). If those strings are not assertable from a unit
test, "we support pgvector" is a claim nobody can check without a live cluster.
So this module is pure and import-free, `vector_store.PgvectorBackend` only
plumbs these strings through `psycopg`, and `tests/test_pgvector.py` checks the
text itself. `migrations/0002_pgvector.sql` is generated from here too, so the
committed schema and the code that writes rows cannot drift apart.

Operators used (pgvector semantics, all on `vector`):
  `<=>`  cosine distance  = 1 - cosine_similarity  -> our ranking
  `<->`  L2 distance      -> needs normalized inputs to mean the same thing
  `<#>`  negative inner product
Embeddings are L2-normalized by `helix.embeddings`, so `<=>` and `<->` rank the
same; `<=>` is still the right choice because it stays meaningful if a future
backend (e.g. raw SentenceTransformer output) forgets to normalize.
"""

from __future__ import annotations

from collections.abc import Sequence

from helix.settings import EMBED_DIM

TABLE = "helix_facts"

# HNSW knobs. Defaults are deliberate, not cargo-culted:
#   m = 16            edges per node. Higher = better recall/graph quality and a
#                     slower build + more RAM. 16 is pgvector's own default and
#                     sits right for < ~5M rows at 64-768 dims; 32 buys little
#                     recall once the corpus is this small and doubles index size.
#   ef_construction   = 64  candidate list size while *building*. This is the
#                     knob that trades build time for graph quality; 64-128 is
#                     the documented sweet spot. Raising it never degrades query
#                     recall, so it is the cheap lever at load time.
# Query-time `ef_search` (default 40) is set per statement by the backend —
# ANN recall for a top-k of 5-20 needs ef_search >= k, which the over-fetch in
# `cosine_topk_sql` already assumes.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100

# ivfflat alternative: `lists` ~ 4 * sqrt(n_rows) for 1M rows -> 4000; the tiny
# Helix fixture would build 4 lists and put almost every row in one of them.
IVFFLAT_LISTS = 100

_VECTOR_TYPE = "vector"


def dim_for(embed_dim: int | None = None) -> int:
    """Column width = embedder width. A mismatch is a hard error, not a pad."""
    dim = EMBED_DIM if embed_dim is None else int(embed_dim)
    if not 1 <= dim <= 16000:  # pgvector's vector limit
        raise ValueError(f"embedding dimension {dim} is outside pgvector's range")
    return dim


def vector_literal(values: Sequence[float]) -> str:
    """`[a,b,c]` text form, for `%s::vector` binds and for INSERT defaults."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def extension_ddl() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector"


def table_ddl(embed_dim: int | None = None) -> str:
    dim = dim_for(embed_dim)
    return f"""CREATE TABLE IF NOT EXISTS {TABLE} (
    id              TEXT PRIMARY KEY,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'user',
    observed_at     BIGINT NOT NULL,
    ttl_ms          BIGINT,
    embedding       vector({dim}) NOT NULL,
    embed_backend   TEXT NOT NULL,
    embed_dim       INTEGER NOT NULL DEFAULT {dim},
    CHECK (embed_dim = {dim})
)"""


def hnsw_index_ddl(embed_dim: int | None = None, m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION) -> str:
    """HNSW on cosine — the default and the one to keep.

    HNSW builds a navigable small-world graph: query cost ~ O(log n), recall is
    stable as the table grows, and inserts do not need a rebuild. It wins for
    Helix because the fact table is write-mostly and unbounded (every remembered
    fact is an insert) and because top-k here is small.
    """
    dim = dim_for(embed_dim)
    return (
        f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw ON {TABLE} "
        f"USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction})"
    )


def ivfflat_index_ddl(embed_dim: int | None = None, lists: int = IVFFLAT_LISTS) -> str:
    """ivfflat — kept as the documented alternative, commented out in the migration.

    ivfflat partitions vectors into `lists` k-means cells and probes only
    `nprobes` of them. It builds far faster and is much smaller on disk than
    HNSW, so it wins when (a) the corpus is large enough for k-means to mean
    something (> ~1M rows), (b) writes arrive in batches that allow periodic
    `REINDEX`, and (c) insert latency matters more than tail recall. It loses
    badly when the table is small (most rows land in one cell) or when the
    working set drifts and `nprobes` has to climb toward `lists` anyway.
    """
    dim = dim_for(embed_dim)
    return (
        f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_ivfflat ON {TABLE} "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
    )


def key_index_ddl() -> str:
    return f"CREATE INDEX IF NOT EXISTS {TABLE}_key ON {TABLE}(key)"


def recent_index_ddl() -> str:
    return f"CREATE INDEX IF NOT EXISTS {TABLE}_observed_at ON {TABLE}(observed_at DESC)"


def ttl_partial_index_ddl(embed_dim: int | None = None, m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION) -> str:
    """HNSW over live rows only.

    The ANN scan is ordered by distance, so a TTL predicate normally becomes a
    post-filter: HNSW walks neighbours, throws half of them away, and recall
    drops. A partial index over the rows `recall()` actually accepts keeps the
    graph and the filter in agreement, and stops deleted-by-TTL facts from
    costing query time. Replaces the full index rather than adding to it.
    """
    dim = dim_for(embed_dim)
    return (
        f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw_live ON {TABLE} "
        f"USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction})\n"
        f"    WHERE ttl_ms IS NULL OR observed_at + ttl_ms >= 0"
    )


def upsert_sql(embed_dim: int | None = None) -> str:
    dim_for(embed_dim)
    return f"""INSERT INTO {TABLE}
    (id, key, value, source, observed_at, ttl_ms, embedding, embed_backend, embed_dim)
VALUES
    (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    key = EXCLUDED.key,
    value = EXCLUDED.value,
    source = EXCLUDED.source,
    observed_at = EXCLUDED.observed_at,
    ttl_ms = EXCLUDED.ttl_ms,
    embedding = EXCLUDED.embedding,
    embed_backend = EXCLUDED.embed_backend,
    embed_dim = EXCLUDED.embed_dim"""


def cosine_topk_sql(embed_dim: int | None = None, with_ttl: bool = True) -> str:
    """The ANN query: nearest neighbours by cosine distance.

    `embedding <=> %s::vector` is cosine *distance*; ORDER BY ... ASC is nearest
    first. pgvector only uses the HNSW index when the ORDER BY is exactly this
    expression and there is no other sort — a `1 - (embedding <=> x)` in the
    SELECT list is fine, an extra `ORDER BY observed_at` silently turns the index
    off and you are back to a sequential scan. Hence over-fetch + Python blend
    in `vector_store` instead of a smarter query.
    """
    dim_for(embed_dim)
    ttl = "\n  AND (ttl_ms IS NULL OR observed_at + ttl_ms >= %s)" if with_ttl else ""
    return f"""SELECT id, key, value, source, observed_at, ttl_ms,
       1 - (embedding <=> %s::vector) AS similarity
FROM {TABLE}
WHERE 1 = 1{ttl}
ORDER BY embedding <=> %s::vector
LIMIT %s"""


def ef_search_stmt(ef_search: int = HNSW_EF_SEARCH) -> str:
    """Transaction-local HNSW candidate list. Must be >= k + over-fetch."""
    return f"SET LOCAL hnsw.ef_search = {int(ef_search)}"


def list_sql() -> str:
    return f"""SELECT id, key, value, source, observed_at
FROM {TABLE}
ORDER BY observed_at DESC
LIMIT %s"""


def delete_sql() -> str:
    return f"DELETE FROM {TABLE} WHERE key = %s OR id = %s"


def count_sql() -> str:
    return f"SELECT count(*) FROM {TABLE}"


def commented_ddl(ddl: str) -> str:
    """Render a DDL statement as an inert comment block, terminated by `;`.

    Every line gets the prefix. Hand-commenting only the first line of a
    multi-line statement is the classic migration footgun: the wrapped
    `CREATE INDEX ... ` takes the `--` and its trailing `WHERE ttl_ms IS NULL`
    clause then executes as SQL, so the migration dies with a syntax error at
    deploy time. `test_migration_is_idempotent_sql` parses the file back into
    statements, which catches any line that escaped this helper.
    """
    lines = ddl.rstrip().splitlines()
    block = [f"--   {line}" for line in lines]
    block[-1] = block[-1] + ";"
    return "\n".join(block)


def migration_sql(embed_dim: int | None = None) -> str:
    """Body of `migrations/0002_pgvector.sql`. Single source of truth for schema."""
    dim = dim_for(embed_dim)
    return f"""-- 0002_pgvector.sql — Helix memory facts as real vectors with an ANN index.
--
-- GENERATED by `python -m helix.pgvector_sql --write` (module
-- `helix/pgvector_sql.py`). tests/test_pgvector.py asserts the committed file
-- equals that output for the pinned embedder width ({dim}), so changing
-- EMBED_DIM without regenerating here fails CI on purpose: a `vector({dim})`
-- column and a {dim + 4}-d embedder would otherwise fail at the first INSERT.
--
-- 0001_auth.sql is Better Auth's own schema and stays untouched; this file owns
-- only Helix tables. Applied by scripts/migrate.mjs (deploy / Vercel build) and
-- by the PGLite path in src/lib/db.ts, both keyed by basename in `_migrations`.

-- pgvector is not in the search path by default on RDS/Neon — this needs a
-- role that may create extensions, and the migration fails loudly if not.
{extension_ddl()};

-- One row per remembered fact. The embedding is a typed `vector({dim})`, not
-- JSONB: JSONB gives you no distance operator, no index, and a seq-scan that
-- re-parses {dim} floats per row per query.
{table_ddl(dim)};

-- Nearest-neighbour index. cosine ops because embeddings are L2-normalized in
-- helix/embeddings.py, and the score Helix reports is cosine similarity.
-- m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}: see pgvector_sql.py.
{hnsw_index_ddl(dim)};

-- Bookkeeping indexes: `recall` filters on TTL, `list_facts`/`forget` by key.
{recent_index_ddl()};
{key_index_ddl()};

-- Alternative ANN index — COMMENTED OUT; pick one strategy per deployment.
--
-- ivfflat (lists = {IVFFLAT_LISTS} ~ 4*sqrt(n) for ~40k rows) builds in seconds
-- instead of minutes and is a fraction of HNSW's size, so it is the better
-- choice when the fact table is bulk-loaded and mostly read, or when the table
-- is big enough that HNSW's RAM cost (every vector is reachable through the
-- graph) dominates. It is the wrong choice here: Helix writes one row at a
-- time from a voice turn, ivfflat's centroid map goes stale as new facts
-- arrive, and recall collapses to a seq scan on a small table where most rows
-- share a cell. Switch by dropping the HNSW index and running:
--
--   DROP INDEX {TABLE}_embedding_hnsw;
{commented_ddl(ivfflat_index_ddl(dim, IVFFLAT_LISTS))}
--   SET ivfflat.probes = 10;   -- per-query: probes/lists is your recall dial
--
-- If facts expire and the table grows, prefer the partial variant below over
-- either full index, so dead rows stop costing query time:
--
--   DROP INDEX {TABLE}_embedding_hnsw;
{commented_ddl(ttl_partial_index_ddl(dim))}
"""


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="helix.pgvector_sql")
    parser.add_argument("--write", type=str, default="", help="path of the migration file to (re)generate")
    parser.add_argument("--dim", type=int, default=EMBED_DIM)
    parser.add_argument("--print", action="store_true", dest="print_ddl")
    args = parser.parse_args(argv)

    body = migration_sql(args.dim)
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    if args.print_ddl or not args.write:
        print(body)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
