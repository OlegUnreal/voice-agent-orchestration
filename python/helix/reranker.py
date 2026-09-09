"""Cross-encoder analog: logistic probe on retrieval features.

LoRA r8 = full 5-d logistic (bias + cosine + recency + ticker + title).
QLoRA r4 = rank-dropped (first 4 dims updated, last frozen) — same promotion
loop you would use for a real adapter, trained with NumPy SGD + sklearn check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

BASELINE_WEIGHTS = np.array([0.0, 1.4, 0.2, 0.4, 0.5], dtype=float)
EMBED_WEIGHTS = np.array([-0.2, 2.2, 0.6, 0.8, 0.4], dtype=float)


@dataclass
class TrainPoint:
    features: np.ndarray  # [1, cosine, recency, tickerHit, titleHit]
    y: int


def feat(cosine: float, recency: float, ticker_hit: float, title_hit: float) -> np.ndarray:
    return np.array([1.0, cosine, recency, ticker_hit, title_hit], dtype=float)


def score_rerank(features: np.ndarray, weights: np.ndarray | None = None) -> float:
    w = EMBED_WEIGHTS if weights is None else weights
    z = float(np.dot(w, features))
    return float(1.0 / (1.0 + np.exp(-z)))


def train_logistic(
    data: list[TrainPoint],
    epochs: int = 48,
    lr: float = 0.35,
    l2: float = 0.01,
    rank_drop: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """NumPy SGD. rank_drop>0 freezes trailing weights (QLoRA analog)."""
    w = np.array([0.0, 0.4, 0.1, 0.1, 0.1], dtype=float)
    loss: list[float] = []
    dim = 5 - rank_drop
    x = np.stack([p.features for p in data]) if data else np.zeros((0, 5))
    y = np.array([p.y for p in data], dtype=float)
    n = max(len(data), 1)
    for _ in range(epochs):
        z = x[:, :dim] @ w[:dim]
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = np.clip(pred, 1e-9, 1 - 1e-9)
        l = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        err = pred - y
        g = (x[:, :dim].T @ err) / n + l2 * w[:dim]
        w[:dim] = w[:dim] - lr * g
        loss.append(float(l))
    return w, loss


def sklearn_check(data: list[TrainPoint]) -> np.ndarray:
    """Independent sklearn LogisticRegression on the same features (sanity)."""
    if len(data) < 4:
        return EMBED_WEIGHTS.copy()
    x = np.stack([p.features[1:] for p in data])
    y = np.array([p.y for p in data])
    if len(set(y.tolist())) < 2:
        return EMBED_WEIGHTS.copy()
    clf = LogisticRegression(max_iter=200, C=4.0)
    clf.fit(x, y)
    w = np.zeros(5)
    w[0] = float(clf.intercept_[0])
    w[1:] = clf.coef_[0]
    return w
