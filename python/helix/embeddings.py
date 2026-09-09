"""Hashing-trick embeddings (64-d). Same idea as the MLLLM layer: no GPU, deterministic."""

from __future__ import annotations

import math
import re

import numpy as np

from helix.prng import hash32
from helix.settings import EMBED_DIM

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "as", "at", "by", "from", "that", "this", "be", "are", "was", "it", "into",
}
_TOKEN = re.compile(r"[^a-z0-9%\-]+")


def tokenize(text: str) -> list[str]:
    parts = _TOKEN.sub(" ", text.lower()).split()
    return [t for t in parts if len(t) > 1 and t not in STOP]


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    tokens = tokenize(text)
    grams: list[str] = []
    for i, tok in enumerate(tokens):
        grams.append(tok)
        if i + 1 < len(tokens):
            grams.append(f"{tok}_{tokens[i + 1]}")
    for g in grams:
        for k in range(3):
            h = hash32(g, k + 1) % dim
            sign = 1 if hash32(g, k + 17) % 2 == 0 else -1
            v[h] += sign
    n = float(np.linalg.norm(v)) or 1.0
    v /= n
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
