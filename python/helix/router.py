"""Supervisor intent classifier: lexical rules first, then embedding prototypes."""

from __future__ import annotations

import re

from helix.embeddings import cosine, embed
from helix.models import Intent

PROTOTYPES: list[tuple[Intent, str]] = [
    ("regime", "regime trend bull bear high vol range sma market state"),
    ("anomaly", "anomaly dump spike inflow funding zscore unusual stress"),
    ("backtest", "backtest sma crossover strategy sharpe drawdown trades"),
    ("risk", "risk var cvar shock scenario pnl portfolio drop crash"),
    ("research", "cite evidence research note on-chain why explain grounded"),
    ("portfolio", "portfolio review book weights allocation ballast copilot"),
    ("eval", "eval metrics recall ndcg promotion gates golden regression"),
    ("training", "train lora qlora checkpoint dataset sft reranker registry"),
    ("general", "hello help what can you do helix voice agent mcp tools"),
]
_VEC = [(intent, embed(text)) for intent, text in PROTOTYPES]

_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    ("backtest", re.compile(r"\b(backtest|sharpe|sma crossover|cagr)\b", re.I)),
    ("risk", re.compile(r"\b(var|cvar|shock|drop 12|risk if|scenario|preflight|kill.?switch)\b", re.I)),
    ("anomaly", re.compile(r"\b(anomal|inflow|dump|funding|z-score|zscore|stress window|open interest)\b", re.I)),
    ("regime", re.compile(r"\b(regime|bull|bear|high-vol|high vol|klines|mark price)\b", re.I)),
    ("eval", re.compile(r"\b(promote|ndcg|recall@|golden|evalforge|eval)\b", re.I)),
    ("training", re.compile(r"\b(lora|qlora|checkpoint|train|sft|registry)\b", re.I)),
    ("portfolio", re.compile(r"\b(portfolio|book|weights|copilot review|position|balance|account|futures)\b", re.I)),
    ("research", re.compile(r"\b(cite|evidence|why|research|staking|on-chain|onchain)\b", re.I)),
]


def classify_intent(query: str) -> Intent:
    for intent, pat in _RULES:
        if pat.search(query):
            return intent
    qv = embed(query)
    best: Intent = "general"
    score = -1.0
    for intent, vec in _VEC:
        s = cosine(qv, vec)
        if s > score:
            score, best = s, intent
    return "general" if score < 0.12 else best
