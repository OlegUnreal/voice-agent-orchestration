"""Pydantic contracts for the FastAPI gateway and MCP-style tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Ticker = Literal["BTC", "ETH", "SOL", "NVDA", "AAPL"]
Regime = Literal["bull-trend", "bear-trend", "range", "high-vol"]
Intent = Literal[
    "regime",
    "anomaly",
    "backtest",
    "risk",
    "research",
    "portfolio",
    "eval",
    "training",
    "memory",
    "general",
]
AgentId = Literal["supervisor", "quant", "research", "copilot", "eval", "training"]
PromotionStage = Literal["staging", "shadow", "canary", "production"]
DocType = Literal["research", "onchain", "strategy", "risk", "event"]
RetrieverKind = Literal["lexical", "embed", "lora"]


class Bar(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class OnChainPoint(BaseModel):
    t: int
    activeAddresses: int
    exchangeInflow: float
    fundingRate: float


class Anomaly(BaseModel):
    ticker: Ticker
    t: int
    kind: Literal["return", "volume", "funding", "inflow"]
    z: float
    note: str


class MarketSnapshot(BaseModel):
    ticker: Ticker
    price: float
    change1d: float
    change20d: float
    vol20d: float
    sma20: float
    sma50: float
    sma200: float
    regime: Regime
    lastAnomaly: Anomaly | None = None


class BacktestResult(BaseModel):
    ticker: Ticker
    strategy: str
    params: dict[str, float]
    totalReturn: float
    cagr: float
    sharpe: float
    maxDrawdown: float
    winRate: float
    trades: int
    equity: list[dict[str, float]]


class ScenarioPnl(BaseModel):
    name: str
    pnl: float
    pct: float


class WeightRow(BaseModel):
    ticker: Ticker
    weight: float
    value: float


class RiskResult(BaseModel):
    portfolioValue: float
    var95: float
    cvar95: float
    volAnn: float
    maxDrawdown: float
    scenarioPnl: list[ScenarioPnl]
    weights: list[WeightRow]
    parametricVar95: float | None = None
    monteCarloVar95: float | None = None


class KnowledgeDoc(BaseModel):
    id: str
    title: str
    body: str
    ticker: Ticker | None = None
    type: DocType
    ts: int


class RetrievedDoc(BaseModel):
    doc: KnowledgeDoc
    score: float
    cosine: float
    recency: float
    rerank: float | None = None
    snippet: str | None = None


class Citation(BaseModel):
    id: str
    title: str
    type: DocType
    ts: int


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    ms: float
    ok: bool
    summary: str


class AgentHop(BaseModel):
    agent: AgentId
    label: str
    ms: float


class EvalMetrics(BaseModel):
    recallAt5: float
    recallAt10: float
    mrr: float
    ndcgAt10: float
    intentAcc: float
    toolAcc: float
    groundedness: float
    citationCorrect: float
    structuredOk: float
    p50Ms: float
    p95Ms: float
    costUsd: float


class EvalCase(BaseModel):
    id: str
    query: str
    asOf: int
    relevant: list[str]
    expectedIntent: Intent
    expectedTools: list[str]


class Checkpoint(BaseModel):
    id: str
    name: str
    kind: Literal["baseline", "embed", "lora", "qlora"]
    rank: int | None = None
    createdAt: int
    metrics: EvalMetrics
    stage: PromotionStage
    lineage: list[str]


class Gate(BaseModel):
    name: str
    ok: bool
    value: float


class McpTool(BaseModel):
    name: str
    server: str
    description: str
    governed: Literal["auto", "approve"]
    schema_: dict[str, str] = Field(alias="schema")

    model_config = {"populate_by_name": True}


class Trace(BaseModel):
    id: str
    ts: int
    query: str
    intent: Intent
    tools: list[str]
    latencyMs: float
    tokens: int
    costUsd: float
    retrievalHit: bool
    feedback: Literal["up", "down"] | None = None


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=800)


class RetrieveRequest(BaseModel):
    query: str
    k: int = 6
    chronological: bool = True
    kind: RetrieverKind = "lora"


class TrainRequest(BaseModel):
    epochs_lora: int = 56
    epochs_qlora: int = 40


class PromoteRequest(BaseModel):
    checkpoint_id: str


class OrchestratorResult(BaseModel):
    intent: Intent
    hops: list[AgentHop]
    tools: list[ToolCall]
    citations: list[Citation]
    snapshot: MarketSnapshot | None = None
    anomalies: list[Anomaly] | None = None
    backtest: BacktestResult | None = None
    risk: RiskResult | None = None
    retrieved: list[RetrievedDoc] | None = None
    numbers: dict[str, Any]
    fallbackSpoken: str
    grounded: bool
    datasetVersion: str | None = None
    verified: bool = True
    verifyLeaks: list[str] = Field(default_factory=list)
    traceId: str | None = None
    memory: list[dict[str, Any]] = Field(default_factory=list)
