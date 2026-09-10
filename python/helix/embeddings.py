"""Hashing-trick embeddings (64-d). Optional sentence-transformers when HELIX_EMBEDDINGS=st."""

from __future__ import annotations

import math
import os
import re

import numpy as np

from helix.prng import hash32
from helix.settings import EMBED_DIM

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "as", "at", "by", "from", "that", "this", "be", "are", "was", "it", "into",
}
_TOKEN = re.compile(r"[^a-z0-9%\-]+")
_ST = None


def tokenize(text: str) -> list[str]:
    parts = _TOKEN.sub(" ", text.lower()).split()
    return [t for t in parts if len(t) > 1 and t not in STOP]


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
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


def backend() -> str:
    raw = os.environ.get("HELIX_EMBEDDINGS", "hash").lower()
    return "st" if raw in {"st", "minilm", "sentence-transformers"} else "hash"


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    if backend() == "st":
        global _ST
        try:
            if _ST is None:
                from sentence_transformers import SentenceTransformer

                _ST = SentenceTransformer(os.environ.get("HELIX_ST_MODEL", "all-MiniLM-L6-v2"))
            vec = np.asarray(_ST.encode(text), dtype=np.float64)
            n = float(np.linalg.norm(vec)) or 1.0
            return vec / n
        except Exception:
            pass
    return _hash_embed(text, dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        m = min(a.size, b.size)
        a, b = a[:m], b[:m]
    return float(np.dot(a, b))
