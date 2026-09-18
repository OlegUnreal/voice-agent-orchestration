"""Okapi BM25 + Reciprocal Rank Fusion: the lexical half of hybrid retrieval.

Why a hand-rolled BM25 instead of the token-overlap scorer that used to sit in
`rag.lexical_retrieve`

    Overlap counted "how many query tokens appear in the doc", divided by
    sqrt(|q| * |d|). Three things wrong with that as a production retriever:

    1. No saturation. A doc that says "inflow" 12 times scores 12x a doc that
       says it once, so keyword-stuffed boilerplate outranks the note you
       wanted. BM25's `tf / (tf + k1*(1-b+b*dl/avgdl))` flattens that curve.
    2. No document frequency. "btc" and "the-ish rare term" weighed the same,
       so an informative token moved the ranking as much as a token that is in
       every doc. BM25 multiplies by idf.
    3. Length normalisation by sqrt() is arbitrary: long docs are punished
       quadratically, and the corpus here mixes a 40-word note with a 120-word
       one. `b` is an explicit, tunable dial for that.

    The overlap scorer is kept, renamed `overlap_scores`, as the documented
    baseline the eval ablation still runs.

Why fusion, not score blending

    cosine in [0,1], BM25 in [0, inf), a logistic reranker in (0,1) - there is
    no shared scale, so a weighted sum of raw scores is a coin flip dressed up
    as a formula. Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) sums
    `1/(k0 + rank)` over each ranked list, so only the order matters. It is
    robust to a miscalibrated list, has one interpretable knob (k0, the rank at
    which a list's vote halves), and it is exactly why a doc ranked 2nd by both
    systems beats one that is 1st in one and 40th in the other.

`rank_bm25` is installed in some environments and not others; it is used only
as a cross-check in the test suite (see `reference_scores`), never at runtime,
so CI does not gain a hard dependency.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from helix.embeddings import tokenize
from helix.models import KnowledgeDoc

# Tuned by hand on the 18-doc Helix corpus and re-checked by the eval ablation.
# k1=1.4 (slightly below the classic 1.2-2.0 middle) because the notes are
# short: high k1 rewards term repetition that these docs almost never show.
# b=0.4 (below the default 0.5) because titles carry the signal and bodies are
# similar in length, so full length normalisation only hurts.
K1 = 1.4
B = 0.4
# A title token is worth this many body occurrences. Titles are curated
# ("BTC exchange inflow spike window"); bodies mention the same words in
# passing. Implemented as repetition, so it saturates like any other tf.
TITLE_REPEAT = 2
# RRF half-vote rank. 60 is the paper's default and stays: with 18 docs a
# smaller k0 makes the fusion nearly a pure "who is first" copy of BM25.
RRF_K0 = 60


def _idf(n_docs: int, df: int) -> float:
    """Lucene-style non-negative idf: log(1 + (N - df + 0.5) / (df + 0.5)).

    The classic Robertson idf, log((N-df+0.5)/(df+0.5)), goes *negative* once a
    term is in more than half the corpus, and a negative weight lets a
    keyword-matching doc be penalised into last place. Clamping at 0 with the
    +1 form is the standard fix and keeps the ordering monotone in df.
    """
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def doc_terms(doc: KnowledgeDoc, title_repeat: int = TITLE_REPEAT) -> list[str]:
    """The indexed token stream for a doc: repeated title, then body, then facets."""
    toks: list[str] = []
    title = tokenize(doc.title)
    toks.extend(title * max(1, int(title_repeat)))
    toks.extend(tokenize(doc.body))
    toks.extend(tokenize(doc.type))
    if doc.ticker:
        toks.append(doc.ticker.lower())
    return toks


@dataclass
class TermTrace:
    """Per-query-term explanation, for the trace ledger / RAG view."""

    term: str
    df: int
    idf: float
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class Bm25Hit:
    doc_id: str
    score: float
    terms: dict[str, float] = field(default_factory=dict)  # term -> its share of score


class BM25Index:
    """In-memory BM25 over a fixed doc list.

    In production this is Postgres: `tsvector` + `ts_rank_cd` (or a PgSearch /
    ParadeDB BM25 index) with the same df/idf bookkeeping pushed into the
    database. The Python class exists so the ranking math is testable offline
    and identical on both sides of the TS/Python parity check.
    """

    def __init__(
        self,
        docs: Sequence[KnowledgeDoc],
        k1: float = K1,
        b: float = B,
        title_repeat: int = TITLE_REPEAT,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.title_repeat = int(title_repeat)
        self.docs: dict[str, KnowledgeDoc] = {}
        self.tf: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.df: Counter[str] = Counter()
        for doc in docs:
            self.docs[doc.id] = doc
            counts = Counter(doc_terms(doc, self.title_repeat))
            self.tf[doc.id] = counts
            self.lengths[doc.id] = sum(counts.values())
            for term in counts:
                self.df[term] += 1
        self.n = len(self.docs)
        self.avgdl = (sum(self.lengths.values()) / self.n) if self.n else 0.0

    # ------------------------------------------------------------------ scoring
    def _term_score(self, term: str, doc_id: str) -> float:
        df = self.df.get(term, 0)
        if not df:
            return 0.0
        tf = self.tf[doc_id].get(term, 0)
        if not tf:
            return 0.0
        norm = self.k1 * (1.0 - self.b + self.b * self.lengths[doc_id] / (self.avgdl or 1.0))
        return _idf(self.n, df) * (tf * (self.k1 + 1.0)) / (tf + norm)

    def hits(self, query: str, allow: set[str] | None = None) -> list[Bm25Hit]:
        """Score every doc matching >=1 query term. `allow` restricts candidate ids."""
        terms = tokenize(query)
        if not terms:
            return []
        acc: dict[str, dict[str, float]] = defaultdict(dict)
        for term in set(terms):
            for doc_id in self.docs:
                if allow is not None and doc_id not in allow:
                    continue
                share = self._term_score(term, doc_id)
                if share > 0.0:
                    acc[doc_id][term] = acc[doc_id].get(term, 0.0) + share
        out = [Bm25Hit(doc_id, sum(parts.values()), parts) for doc_id, parts in acc.items()]
        out.sort(key=lambda h: (-h.score, h.doc_id))
        return out

    def scores(self, query: str, allow: set[str] | None = None) -> dict[str, float]:
        return {h.doc_id: h.score for h in self.hits(query, allow)}

    def rank(self, query: str, allow: set[str] | None = None) -> list[str]:
        """Doc ids best-first, zero-score docs excluded. What RRF consumes."""
        return [h.doc_id for h in self.hits(query, allow)]

    def explain(self, query: str, allow: set[str] | None = None, limit: int = 5) -> list[dict]:
        """Score breakdown per candidate: which query term bought the rank."""
        return [
            {"docId": h.doc_id, "score": round(h.score, 6), "terms": {t: round(v, 6) for t, v in sorted(h.terms.items())}}
            for h in self.hits(query, allow)[:limit]
        ]

    def term_stats(self, query: str) -> list[TermTrace]:
        """df/idf per query term - the diagnostic that explains a bad lexical rank."""
        traces: list[TermTrace] = []
        for term in dict.fromkeys(tokenize(query)):
            df = self.df.get(term, 0)
            traces.append(
                TermTrace(
                    term=term,
                    df=df,
                    idf=_idf(self.n, df) if df else 0.0,
                    contributions={d: round(self._term_score(term, d), 6) for d in self.docs if self._term_score(term, d) > 0},
                )
            )
        return traces

    def top_terms(self, limit: int = 12) -> list[tuple[str, int]]:
        return sorted(self.df.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


# --------------------------------------------------------------------- fusion


def rrf(ranked_lists: Sequence[Sequence[str]], k0: float = RRF_K0) -> dict[str, float]:
    """Reciprocal Rank Fusion over 1-based-ordered id lists.

    score(d) = sum_lists 1/(k0 + rank(d)); a doc absent from a list contributes
    nothing, which is the property that makes fusion safe with a partial index.
    """
    fused: dict[str, float] = defaultdict(float)
    for order in ranked_lists:
        for i, doc_id in enumerate(order):
            fused[doc_id] += 1.0 / (k0 + i + 1.0)
    return dict(fused)


def rrf_trace(ranked_lists: dict[str, Sequence[str]], k0: float = RRF_K0) -> dict[str, dict]:
    """Same maths, keyed by list name, with each list's contribution kept.

    Returns {doc_id: {"score": ..., "parts": {list_name: {"rank": r, "vote": v}}}}.
    The trace UI renders this so a reviewer can see *why* a doc fused to the
    top instead of trusting a scalar.
    """
    out: dict[str, dict] = {}
    for name, order in ranked_lists.items():
        for i, doc_id in enumerate(order):
            row = out.setdefault(doc_id, {"score": 0.0, "parts": {}})
            vote = 1.0 / (k0 + i + 1.0)
            row["score"] += vote
            row["parts"][name] = {"rank": i + 1, "vote": vote}
    for row in out.values():
        row["score"] = round(row["score"], 9)
    return out


def overlap_scores(query: str, docs: Sequence[KnowledgeDoc]) -> dict[str, float]:
    """The pre-BM25 baseline, kept verbatim for the ablation table.

    hit-count over sqrt(|q||d|): no idf, no tf saturation, sqrt length penalty.
    """
    q = set(tokenize(query))
    out: dict[str, float] = {}
    for doc in docs:
        toks = tokenize(f"{doc.title} {doc.body}")
        hit = sum(1 for t in toks if t in q)
        out[doc.id] = hit / math.sqrt(len(toks) or 1)
    return out


# ------------------------------------------------------------------ reference


def reference_backend() -> str | None:
    """Name of the installed reference BM25, if any. Test-only, never runtime."""
    try:
        import rank_bm25  # noqa: F401
    except Exception:
        return None
    return "rank_bm25.BM25Okapi"


def reference_scores(index: "BM25Index", query: str) -> dict[str, float]:
    """`rank_bm25.BM25Okapi` scores on the same token streams, for cross-checking.

    Not used by `helix.rag`: importing a library at request time for a number
    nothing reads would add a dependency for free. Returns {} when the optional
    package is absent.
    """
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return {}
    ids = list(index.docs)
    corpus = [doc_terms(index.docs[i], index.title_repeat) for i in ids]
    okapi = BM25Okapi(corpus, k1=index.k1, b=index.b)
    return dict(zip(ids, [float(s) for s in okapi.get_scores(tokenize(query))], strict=True))


_INDEX: BM25Index | None = None


def bm25_index() -> BM25Index:
    """BM25 over the Helix corpus, built once per process."""
    global _INDEX
    if _INDEX is None:
        from helix.corpus import CORPUS

        _INDEX = BM25Index(CORPUS)
    return _INDEX


def lexical_rank(query: str, allow: set[str] | None = None) -> list[str]:
    return bm25_index().rank(query, allow)


def lexical_scores(query: str, allow: set[str] | None = None) -> dict[str, float]:
    return bm25_index().scores(query, allow)
