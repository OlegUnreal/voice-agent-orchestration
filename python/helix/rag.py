"""Hybrid RAG: cosine + token overlap + recency + chronological mask + logistic rerank."""

from __future__ import annotations

import math

import numpy as np

from helix.corpus import CORPUS
from helix.embeddings import cosine, embed, tokenize
from helix.models import RetrievedDoc
from helix.reranker import EMBED_WEIGHTS, feat, score_rerank
from helix.settings import AS_OF, HALF_LIFE_MS

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


def recency_score(ts: int, as_of: int = AS_OF) -> float:
    age = max(0, as_of - ts)
    return float(math.exp(-math.log(2) * age / HALF_LIFE_MS))


def retrieve(
    query: str,
    k: int = 8,
    as_of: int = AS_OF,
    chronological: bool = True,
    weights: np.ndarray | None = None,
) -> list[RetrievedDoc]:
    qv = embed(query)
    q_tokens = set(tokenize(query))
    w = EMBED_WEIGHTS if weights is None else weights
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
                cosine=cos,
                recency=rec,
                rerank=rerank,
                score=float(score),
                snippet=doc.body[:180],
            )
        )
    scored.sort(key=lambda r: r.score, reverse=True)
    from helix.cross_encoder import rerank as ce_rerank

    return ce_rerank(query, scored[: max(k, 12)], top=k)


def lexical_retrieve(query: str, k: int = 8, as_of: int = AS_OF) -> list[RetrievedDoc]:
    q = set(tokenize(query))
    scored: list[RetrievedDoc] = []
    for doc, _ in _index():
        if doc.ts > as_of:
            continue
        toks = tokenize(f"{doc.title} {doc.body}")
        hit = sum(1 for t in toks if t in q)
        score = hit / math.sqrt(len(toks) or 1)
        scored.append(
            RetrievedDoc(
                doc=doc,
                cosine=score,
                recency=recency_score(doc.ts, as_of),
                score=float(score),
                snippet=doc.body[:180],
            )
        )
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:k]
