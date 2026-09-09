"""Shared constants. Keep in lockstep with the TypeScript shell."""

from datetime import datetime, timezone

TICKERS = ("BTC", "ETH", "SOL", "NVDA", "AAPL")
DAY_MS = 86_400_000
SERIES_LEN = 420
AS_OF = int(datetime(2026, 9, 9, tzinfo=timezone.utc).timestamp() * 1000)
START = AS_OF - (SERIES_LEN - 1) * DAY_MS
EMBED_DIM = 64
HALF_LIFE_MS = 14 * DAY_MS
DEFAULT_BOOK = {"BTC": 0.4, "ETH": 0.25, "SOL": 0.15, "NVDA": 0.1, "AAPL": 0.1}
PORTFOLIO_VALUE = 1_000_000.0
GATES = {
    "ndcgAt10": 0.72,
    "recallAt5": 0.80,
    "groundedness": 0.85,
    "p95Ms": 800.0,
    "costUsd": 0.02,
}
