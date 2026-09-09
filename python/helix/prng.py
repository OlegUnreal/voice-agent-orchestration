"""Deterministic PRNG and FNV-style hash. Unsigned 32-bit, no NumPy overflow."""

from __future__ import annotations

import math

TICKER_SEEDS = {"BTC": 11, "ETH": 23, "SOL": 41, "NVDA": 59, "AAPL": 73}

MASK = 0xFFFFFFFF


def _imul32(a: int, b: int) -> int:
    prod = ((a & MASK) * (b & MASK)) & MASK
    return prod - 0x100000000 if prod >= 0x80000000 else prod


def mulberry32(seed: int):
    a = seed & MASK

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & MASK
        t = _imul32(a ^ (a >> 15), 1 | a)
        t = (t + _imul32(t ^ (t >> 7), 61 | t)) ^ t
        t &= MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0

    return rand


def hash32(text: str, seed: int = 0) -> int:
    h = seed & MASK
    for ch in text:
        h = _imul32(h ^ ord(ch), 16777619) & MASK
    return h


def box_muller(rand) -> float:
    u = max(1e-9, rand())
    v = rand()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
