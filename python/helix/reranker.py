"""Learned reranker: documented feature schema + logistic probe fitted on qrels.

Two feature schemas exist and they are *named*, never positional:

    V1  bias, cosine, recency, tickerHit, titleHit
        The two hardcoded vectors in this module's history
        (`BASELINE_WEIGHTS`, `EMBED_WEIGHTS`). Still used by the v1 scorer in
        `rag.blend_retrieve`, because the eval ablation must be able to
        reproduce the before row.
    V2  bias, cosine, bm25, rrf, recency, tickerHit, titleHit,
        lexicalRankInv, queryTerms
        The product schema. It is what the fitted artifact stores coefficients
        for, and it is what `rag.features_for` emits.

Why names matter: a 5-coefficient vector and a 5-feature row multiply just as
happily in the wrong order as the right one, and the failure looks like a
slightly worse model, not an error. `as_weights` / `v1_weights` map between the
schemas *by feature name*, `score_rerank` refuses a length mismatch, and the
artifact records `featureNames` so a schema change invalidates it loudly.

Runtime model resolution, in order:
    1. `weights=` passed explicitly by the caller (ablation, A/B).
    2. the fitted artifact `helix/rerank_model.json` (see `load_model`).
    3. `PRIOR_WEIGHTS` - the deterministic, hand-set fallback. Documented below.
`HELIX_RERANK_MODEL=prior` forces 3, `=off` makes `retrieve` skip reranking.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("rerank_model.json")
QRELS_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "rerank_qrels.jsonl"
SCHEMA_VERSION = "rerank-v2"

V1_FEATURE_NAMES: tuple[str, ...] = ("bias", "cosine", "recency", "tickerHit", "titleHit")
FEATURE_NAMES: tuple[str, ...] = (
    "bias",  # 1.0, the intercept
    "cosine",  # query/doc embedding cosine in [-1, 1]
    "bm25",  # BM25 saturated at rag.BM25_SATURATION -> [0, 1)
    "rrf",  # fusion vote, normalised by the pool ceiling -> [0, 1]
    "recency",  # exp half-life decay from settings.HALF_LIFE_MS
    "tickerHit",  # 1.0 when the doc ticker appears in the query
    "titleHit",  # 1.0 when any query token is in the title
    "lexicalRankInv",  # 1 / (1 + BM25 rank): *ordinal* lexical evidence
    "queryTerms",  # min(1, |query tokens| / 8); overlap-style features degrade here
)
WEIGHT_DIM = len(FEATURE_NAMES)

# v1 history, kept verbatim for the ablation row (schema V1).
BASELINE_WEIGHTS = np.array([0.0, 1.4, 0.2, 0.4, 0.5], dtype=float)
EMBED_WEIGHTS = np.array([-0.2, 2.2, 0.6, 0.8, 0.4], dtype=float)

# Deterministic fallback: hand-set, on purpose, and never fitted on eval data.
# The signs are the argument: lexical evidence (bm25, rrf, lexicalRankInv) and
# ticker/title are worth more than raw cosine on this corpus, recency is a
# small nudge, and the negative bias keeps a doc with no lexical match below
# 0.5. If someone deletes rerank_model.json the product still behaves the same
# on every run - that is the point of a prior you can read.
_PRIOR_TABLE = {
    "bias": -1.60,
    "cosine": 1.60,
    "bm25": 2.40,
    "rrf": 1.20,
    "recency": 0.50,
    "tickerHit": 0.70,
    "titleHit": 0.50,
    "lexicalRankInv": 0.90,
    "queryTerms": 0.00,
}

# Built in FEATURE_NAMES order, so the readable table above can never drift out
# of sync with the schema; an unknown or missing name is a hard error at import.
_missing = set(FEATURE_NAMES) - set(_PRIOR_TABLE)
_extra = set(_PRIOR_TABLE) - set(FEATURE_NAMES)
if _missing or _extra:
    raise ValueError(f"prior table/schema mismatch: missing={sorted(_missing)} unexpected={sorted(_extra)}")
PRIOR_WEIGHTS = np.array([_PRIOR_TABLE[name] for name in FEATURE_NAMES], dtype=float)


@dataclass
class TrainPoint:
    """One labelled example. `features` may be V1 or V2; the trainer reads the
    dimensionality from the row, so a schema change cannot silently truncate."""

    features: np.ndarray
    y: int


@dataclass
class Pair:
    """One labelled (query, doc) row from the qrels fixture."""

    query: str
    doc_id: str
    rel: int
    split: str
    features: np.ndarray


def binary_rel(rel: int) -> int:
    """Graded qrel relevance -> the binary label the probe is fitted on.

    The fixture stores 0 (irrelevant) .. 3 (core evidence) because graded labels
    are what make nDCG meaningful, but a logistic loss needs one decision
    boundary. Rel >= 2 is "a doc a reviewer would cite"; centralising that here
    keeps the trainer, the fitter and the metrics from drifting apart.
    """
    return 1 if int(rel) >= 2 else 0


def feat(cosine: float, recency: float, ticker_hit: float, title_hit: float) -> np.ndarray:
    """V1 feature row. Kept for `rag.blend_retrieve`, which scores v1."""
    return np.array([1.0, cosine, recency, ticker_hit, title_hit], dtype=float)


def features(
    cosine: float,
    bm25_norm: float,
    rrf_norm: float,
    recency: float,
    ticker_hit: float,
    title_hit: float,
    lexical_rank: int,
    query_terms: int,
) -> np.ndarray:
    """V2 feature row, in `FEATURE_NAMES` order. rag.features_for is the caller."""
    return np.array(
        [
            1.0,
            float(cosine),
            float(bm25_norm),
            float(rrf_norm),
            float(recency),
            float(ticker_hit),
            float(title_hit),
            1.0 / (1.0 + max(1, int(lexical_rank))),
            min(1.0, max(0, int(query_terms)) / 8.0),
        ],
        dtype=float,
    )


def _map_by_name(vec: Sequence[float], src: tuple[str, ...], dst: tuple[str, ...]) -> np.ndarray:
    """Re-express coefficients for another schema, by feature name.

    Missing names become 0.0: a v1 vector simply does not know about bm25/rrf,
    and inventing a weight for them is how a silent regression ships.
    """
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.shape[0] != len(src):
        raise ValueError(f"expected {len(src)} coefficients for {src[:2]}..., got {v.shape[0]}")
    lookup = dict(zip(src, v, strict=True))
    return np.array([lookup.get(name, 0.0) for name in dst], dtype=float)


def as_weights(vec: Sequence[float] | None) -> np.ndarray:
    """Coerce any accepted weight vector to the V2 (product) schema."""
    if vec is None:
        return current_weights()
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.shape[0] == WEIGHT_DIM:
        return v
    if v.shape[0] == len(V1_FEATURE_NAMES):
        return _map_by_name(v, V1_FEATURE_NAMES, FEATURE_NAMES)
    raise ValueError(f"unknown reranker schema: {v.shape[0]} coefficients")


def v1_weights(vec: Sequence[float] | None) -> np.ndarray:
    """Project onto the V1 schema for the v1 scorer. Unknown dims -> EMBED_WEIGHTS."""
    if vec is None:
        return EMBED_WEIGHTS.copy()
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.shape[0] == len(V1_FEATURE_NAMES):
        return v.copy()
    if v.shape[0] == WEIGHT_DIM:
        return _map_by_name(v, FEATURE_NAMES, V1_FEATURE_NAMES)
    return EMBED_WEIGHTS.copy()


# ------------------------------------------------------------------ artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_model(path: Path | None = None) -> tuple[np.ndarray, dict, str | None]:
    """(weights, meta, fallback_reason). Never raises: a bad artifact degrades."""
    p = path or MODEL_PATH
    if os.environ.get("HELIX_RERANK_MODEL", "").lower() in {"prior", "fallback", "off"}:
        return PRIOR_WEIGHTS.copy(), {"source": "prior"}, "env-forced-prior"
    if not p.exists():
        return PRIOR_WEIGHTS.copy(), {"source": "prior"}, "artifact-missing"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        names = tuple(raw["featureNames"])
        if names != FEATURE_NAMES:
            return PRIOR_WEIGHTS.copy(), {"source": "prior"}, "feature-schema-drift"
        w = np.asarray(raw["coefficients"], dtype=float)
        if w.shape[0] != WEIGHT_DIM or not np.all(np.isfinite(w)):
            return PRIOR_WEIGHTS.copy(), {"source": "prior"}, "corrupt-coefficients"
        if not raw.get("accepted", False):
            # Fitted but not accepted: valid numbers, worse than the prior on
            # the holdout. Loading it anyway is how a regression ships quietly.
            why = (raw.get("acceptance") or {}).get("reason", "gate-rejected")
            meta = {"source": "prior", "rejected": str(p.name), "rejectedReason": why, "rejectedMetrics": raw.get("metrics")}
            return PRIOR_WEIGHTS.copy(), meta, f"artifact-rejected:{why}"
        meta = {k: raw.get(k) for k in ("version", "algorithm", "metrics", "dataset", "trainedAt", "sklearnVersion")}
        meta["source"] = str(p.name)
        meta["accepted"] = bool(raw.get("accepted"))
        return w, meta, None
    except Exception as exc:  # malformed JSON, unreadable, missing key
        return PRIOR_WEIGHTS.copy(), {"source": "prior"}, f"artifact-unreadable:{type(exc).__name__}"


_CACHE: tuple[float, np.ndarray, dict, str | None] | None = None


def current_weights() -> np.ndarray:
    """Weights the product path scores with. Cached on the artifact mtime."""
    global _CACHE
    try:
        stamp = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    except OSError:
        stamp = 0.0
    if _CACHE is None or _CACHE[0] != stamp:
        w, meta, reason = load_model()
        _CACHE = (stamp, w, meta, reason)
    return _CACHE[1]


def model_status() -> dict:
    """What the gateway / trace ledger reports about the active reranker."""
    global _CACHE
    w, meta, reason = load_model()
    _CACHE = (0.0, w, meta, reason) if _CACHE is None else _CACHE
    return {
        "schema": SCHEMA_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "source": meta.get("source"),
        "algorithm": meta.get("algorithm"),
        "trainedAt": meta.get("trainedAt"),
        "version": meta.get("version"),
        "fallbackReason": reason,
        "accepted": meta.get("accepted"),
        "rejected": meta.get("rejected"),
        "rejectedReason": meta.get("rejectedReason"),
        "rejectedMetrics": meta.get("rejectedMetrics"),
        "weights": {n: round(float(c), 6) for n, c in zip(FEATURE_NAMES, w, strict=True)},
        "prior": {n: round(float(c), 6) for n, c in zip(FEATURE_NAMES, PRIOR_WEIGHTS, strict=True)},
        "metrics": meta.get("metrics"),
        "dataset": meta.get("dataset"),
    }


def enabled() -> bool:
    return os.environ.get("HELIX_RERANK_MODEL", "").lower() not in {"off", "none", "0"}


def score_rerank(features_: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Sigmoid of the linear probe. Schema mismatches raise instead of guessing."""
    f = np.asarray(features_, dtype=float).reshape(-1)
    w = np.asarray(EMBED_WEIGHTS if (weights is None and f.shape[0] == len(V1_FEATURE_NAMES)) else (weights if weights is not None else current_weights()), dtype=float).reshape(-1)
    if f.shape != w.shape:
        raise ValueError(
            f"reranker schema mismatch: {f.shape[0]}-d features vs {w.shape[0]}-d coefficients "
            f"(v1 rows pair with v1 weights; use as_weights/v1_weights to convert by name)"
        )
    z = float(np.dot(w, f))
    return float(1.0 / (1.0 + np.exp(-z)))


# ------------------------------------------------------------- SGD trainer


def train_logistic(
    data: list[TrainPoint],
    epochs: int = 48,
    lr: float = 0.35,
    l2: float = 0.01,
    rank_drop: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Full-batch SGD with L2, optionally freezing trailing weights (QLoRA analog).

    Dimension comes from the data, not a hardcoded 5, so the LoRA/QLoRA rows of
    the ablation train on the same V2 features the artifact is fitted on.
    `rank_drop` freezes the last `rank_drop` coefficients at their initial value
    - a rank-limited adapter in the same spirit as LoRA, and honest about being
    a linear analogue.
    """
    dim = int(data[0].features.shape[0]) if data else WEIGHT_DIM
    w = np.full(dim, 0.0, dtype=float)
    w[0] = 0.0
    w[1:] = 0.1
    loss: list[float] = []
    trainable = max(1, dim - int(rank_drop))
    x = np.stack([p.features for p in data]) if data else np.zeros((0, dim))
    y = np.array([p.y for p in data], dtype=float)
    n = max(len(data), 1)
    for _ in range(int(epochs)):
        z = x[:, :trainable] @ w[:trainable]
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = np.clip(pred, 1e-9, 1 - 1e-9)
        loss.append(float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))))
        err = pred - y
        g = (x[:, :trainable].T @ err) / n + l2 * w[:trainable]
        w[:trainable] = w[:trainable] - lr * g
    return w, loss


def sklearn_check(data: list[TrainPoint]) -> np.ndarray:
    """Independent sklearn fit on the same rows, as a sanity check on the SGD."""
    if len(data) < 4:
        return EMBED_WEIGHTS.copy()
    x = np.stack([p.features[1:] for p in data])
    y = np.array([p.y for p in data])
    if len(set(y.tolist())) < 2:
        return EMBED_WEIGHTS.copy()
    clf = LogisticRegression(max_iter=200, C=4.0)
    clf.fit(x, y)
    w = np.zeros(x.shape[1] + 1)
    w[0] = float(clf.intercept_[0])
    w[1:] = clf.coef_[0]
    return w


# ------------------------------------------------------------------- fitting


QRELS_SPLITS = ("train", "holdout", "golden")


def load_qrels(path: Path | None = None) -> list[dict]:
    """Read the labelled fixture, validating every record.

    A silent bad line here costs a model fitted on half the data, so each
    failure names its file and line.
    """
    p = path or QRELS_PATH
    if not p.exists():
        return []
    rows = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        where = f"{p.name}:{lineno}"
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{where}: qrels record must be an object")
        missing = {"query", "docId", "rel"} - set(row)
        if missing:
            raise ValueError(f"{where}: missing field(s) {sorted(missing)}")
        if not str(row["query"]).strip() or not str(row["docId"]).strip():
            raise ValueError(f"{where}: empty query or docId")
        rel = row["rel"]
        if not isinstance(rel, int) or isinstance(rel, bool) or not 0 <= rel <= 3:
            raise ValueError(f"{where}: rel must be an integer graded 0..3, got {rel!r}")
        split = str(row.get("split", "train"))
        if split not in QRELS_SPLITS:
            raise ValueError(f"{where}: unknown split {split!r}; expected one of {list(QRELS_SPLITS)}")
        rows.append(row)
    return rows


def rank_metrics(scored: list[tuple[float, int]]) -> dict[str, float]:
    """Ranking metrics for one query: (score, rel) pairs, best score first."""
    order = sorted(scored, key=lambda t: (-t[0], -t[1]))
    rels = [r for _, r in order]
    mrr = 0.0
    for i, r in enumerate(rels):
        if r:
            mrr = 1.0 / (i + 1)
            break
    p1 = 1.0 if rels and rels[0] else 0.0
    gains = [(2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels[:5])]
    ideal = sorted((r for _, r in scored), reverse=True)[:5]
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return {"mrr": mrr, "precisionAt1": p1, "ndcg5": (sum(gains) / idcg) if idcg else 0.0}


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def build_pairs(rows: Sequence[dict], pool: int = 12) -> tuple[list[Pair], list[dict]]:
    """Materialise feature rows for labelled qrel pairs.

    Features are computed with the same `rag.first_stage` the product uses, so
    the fitted model cannot see a feature at train time it cannot get at query
    time. Returns (pairs, skipped) - a skipped row is a (query, doc) pair whose
    doc is outside the candidate pool, and that is worth reporting, not
    swallowing.
    """
    from helix.rag import first_stage, features_for

    pairs: list[Pair] = []
    skipped: list[dict] = []
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query"], []).append(row)
    for query, group in by_query.items():
        as_of = int(group[0].get("asOf", 0)) or None
        stage = first_stage(query, as_of=as_of or _default_as_of(), pool=pool)
        for row in group:
            doc_id = row["docId"]
            if doc_id not in stage["candidates"]:
                # Not a training row: the pool never showed this doc to the
                # reranker for this query, so any feature vector built for it
                # would be a number the product path cannot produce.
                skipped.append({"query": query, "docId": doc_id, "reason": "outside-pool"})
                continue
            pairs.append(
                Pair(
                    query=query,
                    doc_id=doc_id,
                    rel=int(row["rel"]),
                    split=str(row.get("split", "train")),
                    features=features_for(query, doc_id, stage),
                )
            )
    return pairs, skipped


def _default_as_of() -> int:
    from helix.settings import AS_OF

    return AS_OF


def evaluate(weights: np.ndarray, pairs: list[Pair]) -> dict[str, float]:
    """Ranking + calibration metrics for one split-defining set of pairs."""
    per_query: dict[str, list[tuple[float, int]]] = {}
    ys: list[float] = []
    ps: list[float] = []
    for p in pairs:
        s = score_rerank(p.features, weights)
        per_query.setdefault(p.query, []).append((s, p.rel))
        # Ranking keeps the graded label; calibration asks the sigmoid to
        # predict "would a reviewer cite this", which is 0/1 by construction.
        ys.append(float(binary_rel(p.rel)))
        ps.append(s)
    ranked = [rank_metrics(v) for v in per_query.values() if any(r for _, r in v)]
    n = max(len(ranked), 1)
    out = {
        "mrr": round(sum(r["mrr"] for r in ranked) / n, 6),
        "precisionAt1": round(sum(r["precisionAt1"] for r in ranked) / n, 6),
        "ndcg5": round(sum(r["ndcg5"] for r in ranked) / n, 6),
        "logLoss": round(_log_loss(np.array(ys), np.array(ps)), 6) if ys else 1.0,
        "queries": float(len(per_query)),
        "pairs": float(len(pairs)),
    }
    return out


def ranknet_rows(pairs: Sequence[Pair]) -> list[tuple[str, np.ndarray]]:
    """Within-query (better - worse) feature differences.

    Pointwise labels teach "is this doc citable", which is a calibration
    question; the product decision is *which of these twelve comes first*. A
    pairwise objective optimises the ordering directly, which matters when the
    whole fixture is ~90 rows. Pairs with equal labels are skipped: they carry
    no ordering information and pretending otherwise injects noise as signal.
    """
    by_query: dict[str, list[Pair]] = {}
    for p in pairs:
        by_query.setdefault(p.query, []).append(p)
    rows: list[tuple[str, np.ndarray]] = []
    for query, group in by_query.items():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.rel == b.rel:
                    continue
                hi, lo = (a, b) if a.rel > b.rel else (b, a)
                rows.append((query, hi.features - lo.features))
    return rows


def fit_pairwise(rows: Sequence[tuple[str, np.ndarray]], c: float) -> np.ndarray:
    """RankNet on feature differences: minimise -log sigmoid(w . (f+ - f-)).

    The bias column is dropped because it cancels in a difference; the fitted
    intercept is put back into slot 0 so the resulting vector is a complete
    `FEATURE_NAMES`-ordered weight vector. sklearn needs two classes, so every
    Δ is mirrored with -Δ labelled 0 - which is also the symmetric form of the
    RankNet loss, since preferring A over B must mean disliking B over A.
    """
    if not rows:
        raise ValueError("no ordered pairs: cannot fit a pairwise reranker")
    x = np.stack([d for _, delta in rows for d in (delta[1:], -delta[1:])])
    y = np.tile([1, 0], len(rows))
    clf = LogisticRegression(max_iter=2000, C=float(c), fit_intercept=True, solver="lbfgs")
    clf.fit(x, y)
    w = np.zeros(WEIGHT_DIM)
    w[0] = float(clf.intercept_[0])
    w[1:] = clf.coef_[0]
    return w


def fit_pointwise(pairs: Sequence[Pair], c: float) -> np.ndarray:
    """The comparator: plain binary logistic on the same feature rows."""
    x = np.stack([p.features[1:] for p in pairs])
    y = np.array([binary_rel(p.rel) for p in pairs])
    clf = LogisticRegression(max_iter=2000, C=float(c), fit_intercept=True, solver="lbfgs")
    clf.fit(x, y)
    w = np.zeros(WEIGHT_DIM)
    w[0] = float(clf.intercept_[0])
    w[1:] = clf.coef_[0]
    return w


def select_c(pairs: Sequence[Pair], grid: Sequence[float] = (0.05, 0.2, 1.0, 5.0), folds: int = 4) -> tuple[float, list[dict]]:
    """Pick C by deterministic query-grouped CV *inside the train split*.

    Grouped by query, not by row: two rows from one query share a candidate
    pool, so a row-level fold leaks the ordering the model is being scored on.
    Selecting on the holdout would do the same thing one level up, which is why
    the holdout below is only ever read by the acceptance gate.
    """
    queries = sorted({p.query for p in pairs})
    k = max(2, min(int(folds), len(queries)))
    by_query: dict[str, list[Pair]] = {}
    for p in pairs:
        by_query.setdefault(p.query, []).append(p)
    curve: list[dict] = []
    for c in grid:
        scores: list[float] = []
        for f in range(k):
            held = {q for i, q in enumerate(queries) if i % k == f}
            train = [p for q in queries if q not in held for p in by_query[q]]
            test = [p for q in sorted(held) for p in by_query[q]]
            if not train or not test or len({binary_rel(p.rel) for p in train}) < 2:
                continue
            try:
                w = fit_pairwise(ranknet_rows(train), c)
            except ValueError:
                continue
            scores.append(evaluate(w, test)["ndcg5"])
        curve.append({"C": float(c), "cvNdcg5": round(sum(scores) / len(scores), 6) if scores else 0.0, "folds": len(scores)})
    best = max(curve, key=lambda r: (r["cvNdcg5"], -r["C"]))
    tied = [r for r in curve if abs(r["cvNdcg5"] - best["cvNdcg5"]) < 1e-9]
    strongest = min(tied, key=lambda r: r["C"])  # same CV score -> more regularisation
    return float(strongest["C"]), curve


C_GRID: tuple[float, ...] = (0.05, 0.2, 1.0, 5.0)
CV_FOLDS = 4
FIT_ALGORITHM = (
    "RankNet - sklearn LogisticRegression on within-query feature differences "
    "(f+ minus f-, mirrored for two classes), L2 penalty, C selected by "
    "query-grouped 4-fold CV inside the train split"
)


def fit(path: Path | None = None, qrels: Path | None = None, write: bool = True) -> dict:
    """Fit the pairwise probe on the qrels fixture and persist the artifact.

    Split discipline, because this is the part a reviewer should check first:
      * `train`   - the only split that shapes the model: ordered pairs feed the
                    loss, and the C grid is scored by query-grouped CV within it.
      * `holdout` - read by the acceptance gate and by the reported metrics,
                    never by selection.
      * `golden`  - reported only, never fitted on and never used to choose.
    A fitted-but-rejected artifact is written anyway and refused at load time by
    `load_model`, so the run is auditable rather than silently discarded.
    """
    t0 = time.perf_counter()
    rows = load_qrels(qrels)
    if not rows:
        return {"ok": False, "reason": "qrels-missing", "path": str(qrels or QRELS_PATH)}
    pairs, skipped = build_pairs(rows)
    train = [p for p in pairs if p.split == "train"]
    hold = [p for p in pairs if p.split == "holdout"]
    gold = [p for p in pairs if p.split == "golden"]
    train_rows = ranknet_rows(train)
    if not train_rows:
        return {
            "ok": False,
            "reason": "no-ordered-train-pairs",
            "trainPairs": len(train),
            "hint": "a query needs two candidates with different rel to contribute an ordering",
        }
    queries_in_rows = len({q for q, _ in train_rows})

    c_best, curve = select_c(train, C_GRID, CV_FOLDS)
    w = fit_pairwise(train_rows, c_best)
    try:
        w_pointwise = fit_pointwise(train, c_best)
    except ValueError:  # single-class train split: no comparator to report
        w_pointwise = None

    metrics: dict = {
        "train": evaluate(w, train),
        "holdout": evaluate(w, hold) if hold else None,
        "golden": evaluate(w, gold) if gold else None,
        "priorTrain": evaluate(PRIOR_WEIGHTS, train),
        "priorHoldout": evaluate(PRIOR_WEIGHTS, hold) if hold else None,
        "priorGolden": evaluate(PRIOR_WEIGHTS, gold) if gold else None,
        "pointwiseComparator": evaluate(w_pointwise, hold) if (hold and w_pointwise is not None) else None,
    }
    selection = {
        "method": "query-grouped K-fold CV, scored on ndcg@5 of the held-out folds",
        "why": "rows from one query share a candidate pool; row-level folds leak the ordering being scored",
        "splitUsed": "train",
        "grid": list(C_GRID),
        "folds": CV_FOLDS,
        "curve": curve,
        "selected": c_best,
        "orderedPairs": len(train_rows),
        "queriesWithOrdering": queries_in_rows,
    }
    acceptance = {
        "rule": "fitted must be >= prior on holdout ndcg5 and mrr",
        "accepted": True,
        "reason": "no-holdout-split",
    }
    if hold:
        fit_h, prior_h = metrics["holdout"], metrics["priorHoldout"]
        worse = [k for k in ("ndcg5", "mrr") if fit_h[k] + 1e-12 < prior_h[k]]
        acceptance["reason"] = "beats-prior-on-holdout" if not worse else f"loses-to-prior-on-{worse}"
        acceptance["accepted"] = not worse
        acceptance["holdout"] = {k: fit_h[k] for k in ("ndcg5", "mrr", "logLoss")}
        acceptance["priorHoldout"] = {k: prior_h[k] for k in ("ndcg5", "mrr", "logLoss")}
    accepted = bool(acceptance["accepted"])
    import sklearn

    payload = {
        "version": SCHEMA_VERSION,
        "accepted": accepted,
        "algorithm": FIT_ALGORITHM,
        "algorithmParams": {
            "objective": "pairwise-logistic (RankNet)",
            "C": c_best,
            "maxIter": 2000,
            "solver": "lbfgs",
            "penalty": "l2",
            "biasDropped": True,
        },
        "featureNames": list(FEATURE_NAMES),
        "coefficients": [round(float(c), 8) for c in w],
        "metrics": metrics,
        "acceptance": acceptance,
        "selection": selection,
        "dataset": {
            "path": str((qrels or QRELS_PATH).name),
            "sha256": _sha256(qrels or QRELS_PATH),
            "pairs": len(pairs),
            "trainPairs": len(train),
            "holdoutPairs": len(hold),
            "goldenPairs": len(gold),
            "queries": len({p.query for p in pairs}),
            "skipped": skipped,
        },
        "prior": [round(float(c), 8) for c in PRIOR_WEIGHTS],
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearnVersion": sklearn.__version__,
        "fitMs": round((time.perf_counter() - t0) * 1000, 3),
    }
    if write:
        (path or MODEL_PATH).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return {"ok": True, "accepted": accepted, "acceptance": acceptance, "skipped": skipped, "summary": payload, "weights": w}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(prog="helix.reranker")
    parser.add_argument("--fit", action="store_true", help="fit on the qrels fixture and write the artifact")
    parser.add_argument("--status", action="store_true", help="print the active model and why")
    parser.add_argument("--qrels", type=str, default="")
    args = parser.parse_args(argv)
    if args.fit:
        out = fit(qrels=Path(args.qrels) if args.qrels else None)
        if not out.get("ok"):
            print(json.dumps(out, indent=2))
            return 1
        s = out["summary"]
        print(f"{'ACCEPTED' if out['accepted'] else 'REJECTED'}: {out['acceptance']['reason']}")
        print(json.dumps({
            "acceptance": out["acceptance"],
            "skippedOutOfPool": len(out["skipped"]),
            "metrics": s["metrics"],
            "dataset": {k: v for k, v in s["dataset"].items() if k != "skipped"},
            "coefficients": s["coefficients"],
        }, indent=2))
        return 0
    print(json.dumps(model_status(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
