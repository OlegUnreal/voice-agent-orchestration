"""MCP-style tool registry: JSON schemas + auto vs human-approval governance."""

from __future__ import annotations

from helix.models import McpTool

MCP_TOOLS: list[McpTool] = [
    McpTool(name="get_market_snapshot", server="market", description="Last price, SMAs, 20d vol and regime for a ticker.", governed="auto", schema={"ticker": "BTC|ETH|SOL|NVDA|AAPL"}),
    McpTool(name="detect_regime", server="market", description="Classify bull-trend, bear-trend, range, or high-vol.", governed="auto", schema={"ticker": "string"}),
    McpTool(name="detect_anomalies", server="market", description="Return, volume, funding and inflow z-score events.", governed="auto", schema={"ticker": "string", "lookback": "number"}),
    McpTool(name="run_backtest", server="market", description="SMA crossover backtest with costs, Sharpe, max DD.", governed="auto", schema={"ticker": "string", "fast": "number", "slow": "number"}),
    McpTool(name="estimate_risk", server="market", description="VaR, CVaR, scenario P&L, parametric and Monte-Carlo VaR.", governed="auto", schema={"shockBtc": "number"}),
    McpTool(name="retrieve_evidence", server="retrieval", description="Time-aware hybrid RAG with logistic rerank.", governed="auto", schema={"query": "string", "k": "number"}),
    McpTool(name="get_eval_report", server="eval", description="Golden-set Recall@K, MRR, nDCG, tool and groundedness.", governed="auto", schema={}),
    McpTool(name="get_checkpoint_status", server="eval", description="Model registry stages and promotion gates.", governed="auto", schema={}),
    McpTool(name="propose_rebalance", server="market", description="Draft a portfolio tilt. Requires human approval.", governed="approve", schema={"ticker": "string", "deltaWeight": "number", "reason": "string"}),
    McpTool(name="memory_recall", server="memory", description="Read facts with source and timestamp. Auto.", governed="auto", schema={"query": "string"}),
    McpTool(name="memory_write", server="memory", description="Store a fact with provenance. Auto for explicit remember.", governed="auto", schema={"key": "string", "value": "string"}),
]


def tools_as_json() -> list[dict]:
    return [t.model_dump(by_alias=True) for t in MCP_TOOLS]
