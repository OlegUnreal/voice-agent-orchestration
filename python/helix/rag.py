"""Hybrid retrieval: BM25 + dense cosine, fused with RRF, then reranked.

Pipeline (each stage is a separate function so the eval ablation can run it):

    candidate generation   BM25 top-N (helix/bm25.py) and dense cosine top-N
    fusion                 Reciprocal Rank Fusion of the two rank lists
    rerank                 learned logistic probe over the documented features
                           (helix/reranker.py; artifact fitted on qrels, prior
                           weights as the deterministic fallback)

`retrieve()` is the product path (fusion="rrf", rerank=True). The two older
scorers stay reachable and measured:

    blend_retrieve()    v1 production scorer: 0.4*rerank + 0.4*overlap +
                        0.12*ticker + 0.08*recency. Kept because the ablation
                        table has to be able to reproduce the "before" row.
    overlap_retrieve()  v1 lexical baseline (raw token overlap, no idf/saturation).

Both fusions expose a per-candidate `breakdown`, which is what the trace ledger
and the RAG view render - a single scalar score is not debuggable.
"""

from __future__ import annotations

import math

import numpy as np

from helix import reranker
from helix.bm25 import RRF_K0, bm25_index, rrf, rrf_trace
from helix.corpus import CORPUS
from helix.embeddings import cosine, embed, tokenize
from helix.models import RetrievedDoc
from helix.reranker import feat, score_rerank
from helix.settings import AS_OF, HALF_LIFE_MS

# First-stage over-fetch. Fusing 12 lexical + 12 dense before the reranker
# costs one pass over 18 docs here; on a real corpus it is the difference
# between "the right doc was ranked 9th by BM25 and 4th by cosine" and never
# seeing it at all. Keep it a multiple of k.
POOL_MIN = 12
# BM25 is unbounded, so it enters the feature vector through a saturating
# transform. 4.0 is roughly the score of a two-strong-term match on this
# corpus (see bm25.explain); past that, more keyword hits add little.
BM25_SATURATION = 4.0

_INDEX = None
_INDEX_BACKEND = None


def _index():
    global _INDEX, _INDEX_BACKEND
    from helix.embeddings import backend as emb_backend

    b = emb_backend()
    if _INDEX is None or _INDEX_BACKEND != b:
        _INDEX = [(doc, embed(f"{doc.title} {doc.body} {doc.ticker or ''} {doc.type}")) for doc in CORPUS]
        _INDEX_BACKEND = b
    return _INDEX


def invalidate_index() -> None:
    """Drop the embedding cache. The LSA backend fits on the corpus, so a test
    that swaps backends or documents has to be able to force a rebuild."""
    global _INDEX, _INDEX_BACKEND
    _INDEX = None
    _INDEX_BACKEND = None


def recency_score(ts: int, as_of: int = AS_OF) -> float:
    age = max(0, as_of - ts)
    return float(math.exp(-math.log(2) * age / HALF_LIFE_MS))


def _live(as_of: int, chronological: bool) -> set[str]:
    """Doc ids a query at `as_of` is allowed to see. Masks the future."""
    allowed: set[str] = set()
    for doc, _vec in _index():
        if chronological and doc.ts > as_of:
            continue
        allowed.add(doc.id)
    return allowed


def dense_rank(query: str, allow: set[str] | None = None, pool: int = POOL_MIN) -> list[tuple[str, float]]:
    """Dense first stage: cosine, best first, `allow` restricting candidates."""
    qv = embed(query)
    scored: list[tuple[str, float]] = []
    for doc, vec in _index():
        if allow is not None and doc.id not in allow:
            continue
        scored.append((doc.id, cosine(qv, vec)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[: max(pool, 1)]


def dense_scores(query: str) -> dict[str, float]:
    qv = embed(query)
    return {doc.id: cosine(qv, vec) for doc, vec in _index()}


def _doc_by_id() -> dict[str, object]:
    return {doc.id: doc for doc, _ in _index()}


def first_stage(query: str, as_of: int = AS_OF, chronological: bool = True, pool: int = POOL_MIN, k0: float = RRF_K0) -> dict:
    """Candidate generation + RRF fusion, with every intermediate kept.

    Returns a dict with the two rank lists, the fused votes and the per-doc
    numbers the reranker needs. Splitting this out is what lets `retrieve`
    (rerank on) and the "RRF only" ablation row share one candidate set, so the
    ablation compares reordering and not different pools.
    """
    allow = _live(as_of, chronological)
    qv = embed(query)
    cos = {doc.id: cosine(qv, vec) for doc, vec in _index() if doc.id in allow}
    lex_hits = bm25_index().hits(query, allow)
    bm25_raw = {h.doc_id: h.score for h in lex_hits}
    lex_pool = [h.doc_id for h in lex_hits][:pool]
    dense_ids = sorted(cos, key=lambda i: (-cos[i], i))[:pool]
    fused = rrf([lex_pool, dense_ids], k0)
    trace = rrf_trace({"bm25": lex_pool, "dense": dense_ids}, k0)
    best = 2.0 / (k0 + 1.0)  # first in both lists: the fusion ceiling on this pool
    candidates: dict[str, dict] = {}
    for doc_id in set(lex_pool) | set(dense_ids):
        rank_lex = lex_pool.index(doc_id) + 1 if doc_id in lex_pool else len(lex_pool) + 1
        rank_dense = dense_ids.index(doc_id) + 1 if doc_id in dense_ids else len(dense_ids) + 1
        raw_bm25 = bm25_raw.get(doc_id, 0.0)
        candidates[doc_id] = {
            "cosine": float(cos.get(doc_id, 0.0)),
            "bm25": float(raw_bm25),
            "bm25Norm": float(raw_bm25 / (raw_bm25 + BM25_SATURATION)),
            "rrf": float(fused.get(doc_id, 0.0)),
            "rrfNorm": float(fused.get(doc_id, 0.0) / best),
            "rankLexical": rank_lex,
            "rankDense": rank_dense,
            "votes": trace.get(doc_id, {}).get("parts", {}),
        }
    order = sorted(candidates, key=lambda i: (-candidates[i]["rrf"], i))
    return {
        "query": query,
        "asOf": as_of,
        "lexicalRank": lex_pool,
        "denseRank": dense_ids,
        "candidates": candidates,
        "order": order,
        "k0": k0,
        "pool": pool,
    }


def features_for(query: str, doc_id: str, stage: dict | None = None, k0: float = RRF_K0) -> np.ndarray:
    """The documented reranker feature vector for one (query, doc) pair.

    Feature order is `reranker.FEATURE_NAMES` and is part of the model
    artifact contract: a coefficient vector trained on a different order would
    still multiply cleanly and silently score nonsense, so the artifact pins
    both the names and their order.
    """
    stage = stage or first_stage(query, k0=k0)
    row = stage["candidates"].get(doc_id)
    doc = _doc_by_id().get(doc_id)
    if row is None or doc is None:  # labelled doc outside the pool: score it anyway
        qv = embed(query)
        cos = cosine(qv, embed(f"{doc.title} {doc.body} {doc.ticker or ''} {doc.type}")) if doc else 0.0
        raw = bm25_index().scores(query, {doc_id}).get(doc_id, 0.0)
        row = {
            "cosine": float(cos),
            "bm25Norm": float(raw / (raw + BM25_SATURATION)),
            "rrfNorm": 0.0,
            "rankLexical": POOL_MIN + 1,
        }
    q_tokens = set(tokenize(query))
    title_tokens = set(tokenize(getattr(doc, "title", ""))) if doc is not None else set()
    ticker = getattr(doc, "ticker", None)
    ticker_hit = 1.0 if ticker and ticker in query.upper() else 0.0
    title_hit = 1.0 if title_tokens & q_tokens else 0.0
    rec = recency_score(int(getattr(doc, "ts", 0) or 0), stage["asOf"]) if doc is not None else 0.0
    return reranker.features(
        cosine=row["cosine"],
        bm25_norm=row["bm25Norm"],
        rrf_norm=row["rrfNorm"],
        recency=rec,
        ticker_hit=ticker_hit,
        title_hit=title_hit,
        lexical_rank=row["rankLexical"],
        query_terms=len(q_tokens),
    )


def retrieve(
    query: str,
    k: int = 8,
    as_of: int = AS_OF,
    chronological: bool = True,
    weights: np.ndarray | None = None,
    *,
    fusion: str = "rrf",
    rerank: bool = True,
    k0: float = RRF_K0,
) -> list[RetrievedDoc]:
    """Hybrid retrieval. `fusion="blend"` restores the v1 weighted-sum scorer.

    `rerank=False` stops after fusion - that row is what proves the reranker
    earns its latency instead of a bigger first stage.
    """
    if fusion == "blend":
        return blend_retrieve(query, k=k, as_of=as_of, chronological=chronological, weights=weights)
    # HELIX_RERANK_MODEL=off is an operator kill switch: fall back to the fused
    # first stage rather than scoring with a model someone just disabled.
    rerank = rerank and (weights is not None or reranker.enabled())
    stage = first_stage(query, as_of, chronological, pool=max(POOL_MIN, k), k0=k0)
    docs = _doc_by_id()
    # Caller weights go through as_weights: a legacy 5-coefficient adapter must
    # be projected by feature name, not padded positionally, or the 9-d v2 rows
    # it multiplies are a different model than the one that was trained.
    if weights is not None:
        w = reranker.as_weights(weights)
    else:
        w = reranker.current_weights() if rerank else None
    rows: list[RetrievedDoc] = []
    for doc_id in stage["order"]:
        c = stage["candidates"][doc_id]
        doc = docs[doc_id]
        fv = None
        prob = None
        if rerank:
            fv = features_for(query, doc_id, stage, k0=k0)
            prob = score_rerank(fv, w)
        score = prob if prob is not None else c["rrf"]
        rows.append(
            RetrievedDoc(
                doc=doc,
                score=float(score),
                cosine=c["cosine"],
                recency=recency_score(doc.ts, as_of),
                rerank=None if prob is None else float(prob),
                lexical=c["bm25"],
                rrf=c["rrf"],
                snippet=doc.body[:180],
                breakdown={
                    "bm25": round(c["bm25"], 6),
                    "bm25Norm": round(c["bm25Norm"], 6),
                    "cosine": round(c["cosine"], 6),
                    "rrf": round(c["rrf"], 9),
                    "rankLexical": float(c["rankLexical"]),
                    "rankDense": float(c["rankDense"]),
                    "fusion": 1.0 if fusion == "rrf" else 0.0,
                    "reranked": 1.0 if rerank else 0.0,
                },
            )
        )
    if rerank:
        order = {doc_id: i for i, doc_id in enumerate(stage["order"])}
        rows.sort(key=lambda r: (-r.score, order.get(r.doc.id, 1 << 30)))
    else:
        rows.sort(key=lambda r: (-r.score, r.doc.id))
    from helix.cross_encoder import rerank as ce_rerank

    return ce_rerank(query, rows[: max(k, 12)], top=k)


def hybrid_retrieve(
    query: str,
    k: int = 8,
    as_of: int = AS_OF,
    weights: np.ndarray | None = None,
    *,
    rerank: bool = False,
) -> list[RetrievedDoc]:
    """Explicit name for the fusion path; `rerank=True` adds the probe."""
    return retrieve(query, k=k, as_of=as_of, weights=weights, rerank=rerank)


def dense_retrieve(query: str, k: int = 8, as_of: int = AS_OF, chronological: bool = True) -> list[RetrievedDoc]:
    """Dense-only row of the ablation: cosine, no lexical signal at all."""
    allow = set(_live(as_of, chronological))
    docs = _doc_by_id()
    rows = []
    for doc_id, cos in dense_rank(query, allow, pool=max(POOL_MIN, k)):
        doc = docs[doc_id]
        rows.append(
            RetrievedDoc(
                doc=doc,
                score=float(cos),
                cosine=float(cos),
                recency=recency_score(doc.ts, as_of),
                lexical=0.0,
                rrf=0.0,
                snippet=doc.body[:180],
                breakdown={"cosine": round(cos, 6), "fusion": 0.0, "reranked": 0.0},
            )
        )
    return rows[:k]


def lexical_retrieve(query: str, k: int = 8, as_of: int = AS_OF, chronological: bool = True) -> list[RetrievedDoc]:
    """BM25-only row: idf, term-frequency saturation and length normalisation."""
    allow = set(_live(as_of, chronological))
    docs = _doc_by_id()
    rows = []
    for hit in bm25_index().hits(query, allow)[:k]:
        doc = docs[hit.doc_id]
        cos = dense_scores(query).get(hit.doc_id, 0.0)
        rows.append(
            RetrievedDoc(
                doc=doc,
                score=float(hit.score),
                cosine=float(cos),
                recency=recency_score(doc.ts, as_of),
                lexical=float(hit.score),
                rrf=0.0,
                snippet=doc.body[:180],
                breakdown={"bm25": round(hit.score, 6), "fusion": 0.0, "reranked": 0.0, **{f"t:{t}": round(v, 6) for t, v in hit.terms.items()}},
            )
        )
    return rows


def overlap_retrieve(query: str, k: int = 8, as_of: int = AS_OF, chronological: bool = True) -> list[RetrievedDoc]:
    """v1 lexical scorer (raw token overlap). Kept as the documented baseline.

    Not a good retriever: no idf, no saturation, sqrt length penalty. It stays
    in the codebase because `evals.run_eval_suite("overlap")` has to be able to
    produce the number BM25 is compared against.
    """
    from helix.bm25 import overlap_scores

    allow = _live(as_of, chronological)
    docs = _doc_by_id()
    scores = overlap_scores(query, [docs[i] for i in allow])
    rows = []
    for doc_id, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]:
        doc = docs[doc_id]
        rows.append(
            RetrievedDoc(
                doc=doc,
                score=float(score),
                cosine=float(score),
                recency=recency_score(doc.ts, as_of),
                lexical=float(score),
                rrf=0.0,
                snippet=doc.body[:180],
                breakdown={"overlap": round(score, 6), "fusion": 0.0, "reranked": 0.0},
            )
        )
    return rows


def blend_retrieve(
    query: str,
    k: int = 8,
    as_of: int = AS_OF,
    chronological: bool = True,
    weights: np.ndarray | None = None,
) -> list[RetrievedDoc]:
    """v1 scorer, verbatim: 0.4*rerank + 0.4*overlap + 0.12*ticker + 0.08*rec.

    The hardcoded 5-feature weights are gone from the product path; this is the
    regression row that documents what they bought (see docs/vector-search.md).
    """
    qv = embed(query)
    q_tokens = set(tokenize(query))
    # v1 pairs with v1 weights: None -> EMBED_WEIGHTS verbatim, a v2 vector is
    # projected by feature name (bm25/rrf simply have no v1 counterpart).
    w = reranker.v1_weights(weights)
    scored: list[RetrievedDoc] = []
    q_upper = query.upper()
    for doc, vec in _index():
        if chronological and doc.ts > as_of:
            continue
        cos = cosine(qv, vec)
        rec = recency_score(doc.ts, as_of)
        ticker_hit = 1.0 if doc.ticker and doc.ticker in q_upper else 0.0
        title_hit = 1.0 if any(t in tokenize(doc.title) for t in q_tokens) else 0.0
        doc_toks = tokenize(f"{doc.title} {doc.body} {doc.id}")
        hit = sum(1 for t in doc_toks if t in q_tokens)
        overlap = hit / math.sqrt((len(q_tokens) or 1) * (len(doc_toks) or 1))
        features = feat(cos, rec, ticker_hit, title_hit)
        rerank = score_rerank(features, w)
        score = 0.4 * rerank + 0.4 * overlap + 0.12 * ticker_hit + 0.08 * rec
        scored.append(
            RetrievedDoc(
                doc=doc,
                score=float(score),
                cosine=cos,
                recency=rec,
                rerank=rerank,
                lexical=float(overlap),
                rrf=0.0,
                snippet=doc.body[:180],
                breakdown={"overlap": round(overlap, 6), "ticker": ticker_hit, "fusion": 0.0, "reranked": 1.0},
            )
        )
    scored.sort(key=lambda r: r.score, reverse=True)
    from helix.cross_encoder import rerank as ce_rerank

    return ce_rerank(query, scored[: max(k, 12)], top=k)
