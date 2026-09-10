"""Supervisor + specialist graph. LLM is not in this loop — tools own the numbers."""

from __future__ import annotations

import re
import time

from helix.evals import dataset_version, gate_check, run_eval_suite
from helix.market import (
    backtest_sma,
    detect_anomalies,
    detect_regime,
    estimate_risk,
    parse_ticker,
    snapshot,
)
from helix.models import (
    AgentHop,
    AgentId,
    Citation,
    Intent,
    OrchestratorResult,
    ToolCall,
)
from helix.ledger import log_turn, log_verify_fail
from helix.memory import list_facts, parse_remember, recall, remember
from helix.oktrader.live import allow_simulator, json_preview, try_live_turn
from helix.rag import retrieve
from helix.router import classify_intent
from helix.sanitize import scrub_obj
from helix.training import default_checkpoints
from helix.verifier import enforce

INTENT_AGENT: dict[Intent, AgentId] = {
    "regime": "quant",
    "anomaly": "quant",
    "backtest": "quant",
    "risk": "copilot",
    "research": "research",
    "portfolio": "copilot",
    "eval": "eval",
    "training": "training",
    "memory": "copilot",
    "general": "supervisor",
}

WRITE_Q = re.compile(
    r"\b(buy|sell|place (an )?order|market (buy|sell)|cancel all|kill.?switch|"
    r"execute( the)?|propose_rebalance|submit (an )?order)\b",
    re.I,
)
SECRET_Q = re.compile(r"\b(api key|secret key|doppler|bearer token|password|binance key)\b", re.I)


def _summary(name: str, value) -> str:
    if name == "get_market_snapshot":
        return f"{value.ticker} {value.price:.2f} regime={value.regime}"
    if name == "detect_regime":
        return str(value)
    if name == "detect_anomalies":
        return f"{len(value)} events"
    if name == "run_backtest":
        return f"Sharpe {value.sharpe:.2f} DD {value.maxDrawdown * 100:.1f}%"
    if name == "estimate_risk":
        return f"VaR95 {value.var95:.0f} vol {value.volAnn * 100:.1f}%"
    if name == "retrieve_evidence":
        return ", ".join(x.doc.id for x in value[:3])
    if name == "memory_recall":
        return f"{len(value)} facts"
    if name == "memory_write":
        return value.get("key") if isinstance(value, dict) else "stored"
    return "ok"


def _timed(name: str, args: dict, fn):
    t0 = time.perf_counter()
    value = fn()
    ms = (time.perf_counter() - t0) * 1000
    call = ToolCall(name=name, args=args, ms=ms, ok=True, summary=_summary(name, value))
    return value, call


def run_orchestrator(query: str) -> OrchestratorResult:
    t_start = time.perf_counter()
    hops: list[AgentHop] = []
    tools: list[ToolCall] = []
    intent = classify_intent(query)
    hops.append(
        AgentHop(
            agent="supervisor",
            label=f"route → {intent}",
            ms=(time.perf_counter() - t_start) * 1000,
        )
    )
    specialist = INTENT_AGENT[intent]
    ticker = parse_ticker(query) or "BTC"
    numbers: dict = {"intent": intent, "ticker": ticker}
    if WRITE_Q.search(query):
        numbers["writeBlocked"] = True
    if SECRET_Q.search(query):
        numbers["secretBlocked"] = True
    t_spec = time.perf_counter()

    snapshot_res = anomalies = backtest = risk = retrieved = None
    need_snapshot = intent in {"regime", "anomaly", "backtest", "portfolio", "general"}
    need_regime = intent in {"regime", "portfolio"}
    need_anom = intent in {"anomaly", "portfolio"}
    need_bt = intent == "backtest"
    need_risk = intent in {"risk", "portfolio"}
    need_rag = intent in {"research", "general", "anomaly", "regime", "portfolio", "eval", "training"}
    need_eval = intent == "eval"
    need_ckpt = intent in {"training", "eval"}

    live = try_live_turn(query, intent, ticker)
    if live and live.ok and live.tools:
        tools.extend(live.tools)
        numbers.update(live.numbers)
        compact = {k: json_preview(v)[:400] for k, v in live.payload.items()}
        numbers["livePayload"] = compact
        hops.append(
            AgentHop(
                agent="copilot",
                label="ok-trader mcp · " + " · ".join(t.name for t in live.tools),
                ms=float(live.numbers.get("liveMs") or 0),
            )
        )
        need_snapshot = need_regime = need_anom = need_bt = need_risk = False
        if intent != "research":
            need_rag = False
    elif live and not live.ok:
        numbers["liveError"] = live.error
        if not allow_simulator():
            need_snapshot = need_regime = need_anom = need_bt = need_risk = False
            numbers["simulator"] = False
    elif live and live.ok:
        numbers.update(live.numbers)

    if need_snapshot:
        snapshot_res, call = _timed("get_market_snapshot", {"ticker": ticker}, lambda: snapshot(ticker))
        tools.append(call)
        numbers.update(price=snapshot_res.price, change1d=snapshot_res.change1d, regime=snapshot_res.regime, vol20d=snapshot_res.vol20d)
    if need_regime:
        reg, call = _timed("detect_regime", {"ticker": ticker}, lambda: detect_regime(ticker))
        tools.append(call)
        numbers["regime"] = reg
    if need_anom:
        anomalies, call = _timed("detect_anomalies", {"ticker": ticker, "lookback": 60}, lambda: detect_anomalies(ticker))
        tools.append(call)
        numbers["anomalyCount"] = len(anomalies)
    if need_bt:
        backtest, call = _timed("run_backtest", {"ticker": ticker, "fast": 10, "slow": 30}, lambda: backtest_sma(ticker))
        tools.append(call)
        numbers.update(sharpe=backtest.sharpe, totalReturn=backtest.totalReturn, maxDrawdown=backtest.maxDrawdown, trades=backtest.trades)
    if need_risk:
        risk, call = _timed("estimate_risk", {"shockBtc": -0.12}, lambda: estimate_risk(shock_btc=-0.12))
        tools.append(call)
        numbers.update(var95=risk.var95, volAnn=risk.volAnn, scenario0=risk.scenarioPnl[0].pct)
    if need_rag:
        retrieved, call = _timed("retrieve_evidence", {"query": query, "k": 6}, lambda: retrieve(query, k=6))
        tools.append(call)
    if need_eval:
        metrics, call = _timed("get_eval_report", {}, lambda: run_eval_suite("lora"))
        tools.append(call)
        numbers.update(ndcgAt10=metrics.ndcgAt10, recallAt5=metrics.recallAt5, mrr=metrics.mrr, groundedness=metrics.groundedness)
    if need_ckpt:
        ckpts, call = _timed("get_checkpoint_status", {}, default_checkpoints)
        tools.append(call)
        prod = next((c for c in ckpts if c.stage == "production"), None)
        if prod:
            numbers["production"] = prod.name
    parsed = parse_remember(query)
    if parsed:
        key, value = parsed
        stored, call = _timed(
            "memory_write",
            {"key": key, "source": "user"},
            lambda: remember(key, value, source="user"),
        )
        tools.append(call)
        numbers["memoryStored"] = stored["id"]
    mem_hits, call = _timed("memory_recall", {"query": query, "k": 5}, lambda: recall(query, k=5))
    if not mem_hits and intent == "memory":
        mem_hits = list_facts(8)
    tools.append(call)
    numbers["memoryCount"] = len(mem_hits)

    hops.append(
        AgentHop(
            agent=specialist,
            label=" · ".join(t.name for t in tools) or specialist,
            ms=(time.perf_counter() - t_spec) * 1000,
        )
    )
    citations = [
        Citation(id=d.doc.id, title=d.doc.title, type=d.doc.type, ts=d.doc.ts)
        for d in (retrieved or [])[:4]
    ]
    spoken = _speak(intent, ticker, snapshot_res, anomalies, backtest, risk, retrieved, numbers)
    payload_for_verify = {
        "numbers": numbers,
        "tools": [t.summary for t in tools],
        "citations": [c.model_dump() for c in citations],
        "snapshot": snapshot_res.model_dump() if snapshot_res else None,
        "backtest": backtest.model_dump() if backtest else None,
        "risk": risk.model_dump() if risk else None,
        "memory": mem_hits,
    }
    checked = enforce(spoken, payload_for_verify, spoken)
    if not checked["ok"]:
        log_verify_fail({"query": query, "spoken": spoken, "leaks": checked["leaks"]})
        spoken = checked["spoken"]
    grounded = (
        bool(citations)
        or bool(snapshot_res or backtest or risk)
        or bool(numbers.get("live"))
        or bool(mem_hits)
        or bool(numbers.get("memoryStored"))
    ) and checked["ok"]
    result = OrchestratorResult(
        intent=intent,
        hops=hops,
        tools=tools,
        citations=citations,
        snapshot=snapshot_res,
        anomalies=anomalies,
        backtest=backtest,
        risk=risk,
        retrieved=retrieved,
        numbers=numbers,
        fallbackSpoken=spoken,
        grounded=grounded,
        datasetVersion=dataset_version(),
        verified=checked["ok"],
        verifyLeaks=checked["leaks"],
        memory=mem_hits,
    )
    result.traceId = log_turn(
        scrub_obj(
            {
                "query": query,
                "intent": intent,
                "tools": [t.name for t in tools],
                "spoken": spoken,
                "verified": result.verified,
                "leaks": result.verifyLeaks,
                "grounded": grounded,
                "latencyMs": (time.perf_counter() - t_start) * 1000,
                "retrievalHit": bool(retrieved),
                "live": bool(numbers.get("live")),
                "writeBlocked": bool(numbers.get("writeBlocked")),
            }
        )
    )
    return result


def grok_tool_payload(result: OrchestratorResult) -> dict:
    return {
        "intent": result.intent,
        "tools": [{"name": t.name, "summary": t.summary, "args": t.args} for t in result.tools],
        "numbers": result.numbers,
        "citations": [c.model_dump() for c in result.citations],
        "evidence": [
            {"id": r.doc.id, "title": r.doc.title, "body": r.doc.body, "score": round(r.score, 3)}
            for r in (result.retrieved or [])[:4]
        ],
        "snapshot": result.snapshot.model_dump() if result.snapshot else None,
        "anomalies": [a.model_dump() for a in (result.anomalies or [])[:3]],
        "backtest": (
            {
                "ticker": result.backtest.ticker,
                "totalReturn": result.backtest.totalReturn,
                "sharpe": result.backtest.sharpe,
                "maxDrawdown": result.backtest.maxDrawdown,
                "trades": result.backtest.trades,
            }
            if result.backtest
            else None
        ),
        "risk": (
            {
                "var95": result.risk.var95,
                "cvar95": result.risk.cvar95,
                "volAnn": result.risk.volAnn,
                "scenarios": [s.model_dump() for s in result.risk.scenarioPnl],
            }
            if result.risk
            else None
        ),
        "gates": gate_check(run_eval_suite("lora")),
        "datasetVersion": result.datasetVersion,
        "live": bool((result.numbers or {}).get("live")),
        "memory": result.memory,
        "verified": result.verified,
    }


def _speak(intent, ticker, snap, anomalies, backtest, risk, retrieved, numbers) -> str:
    if numbers.get("secretBlocked"):
        return "I do not have API keys. Secrets stay in Doppler and OK-Trader. I will not print them."
    if numbers.get("writeBlocked"):
        return (
            "I will not submit an order. Write tools need human approval. "
            "I can run a snapshot or a preflight instead."
        )
    if numbers.get("liveError") and not numbers.get("live"):
        return (
            "Project Hub MCP is unreachable, so I will not invent a live price. "
            f"{numbers.get('liveError')}. "
            "Run Helix on the Tailscale tailnet or next to the Hub container."
        )
    if numbers.get("live") and numbers.get("livePayload"):
        parts = [f"{k}: {v}" for k, v in list(numbers["livePayload"].items())[:3]]
        joined = " ".join(parts)[:420]
        return (
            f"Live OK-Trader via Project Hub MCP ({int(numbers.get('liveToolCount') or 0)} tools). "
            f"{joined} "
            "Write tools (orders, cancel, kill-switch) are listed and require approval — I did not call them."
        )
    if numbers.get("live") and numbers.get("liveToolsAvailable"):
        names = ", ".join(str(n) for n in numbers["liveToolsAvailable"][:8])
        return f"MCP is up. Available tools: {names}. Ask for a price, position, or risk preflight."
    if intent == "memory":
        if numbers.get("memoryStored"):
            return "Stored with source=user. I will recall it through the memory tool, not as a model guess."
        facts = numbers.get("memoryCount") or 0
        return f"Memory holds {facts} matching facts. Ask me to remember a mandate, a risk cap, or a symbol constraint."
    if intent == "regime" and snap:
        return (
            f"{ticker} is in a {snap.regime.replace('-', ' ')} regime. Last {snap.price:.2f}, "
            f"1-day {snap.change1d * 100:.1f} percent, 20-day vol {snap.vol20d * 100:.0f} percent annualized. "
            f"50-day SMA {snap.sma50:.2f} versus 200-day {snap.sma200:.2f}."
        )
    if intent == "anomaly":
        if not anomalies:
            return f"No z-score events above threshold on {ticker} in the lookback."
        a = anomalies[0]
        return f"Latest {ticker} anomaly: {a.note}. {len(anomalies)} events in the window. Evidence is grounded in the return, volume and on-chain z-scores, not a language-model guess."
    if intent == "backtest" and backtest:
        b = backtest
        return (
            f"{ticker} 10/30 SMA crossover: total return {b.totalReturn * 100:.1f} percent, "
            f"Sharpe {b.sharpe:.2f}, max drawdown {b.maxDrawdown * 100:.1f} percent over {b.trades} trades after 10 bps costs."
        )
    if intent in {"risk", "portfolio"} and risk:
        sc = risk.scenarioPnl[0]
        return (
            f"On a ${risk.portfolioValue / 1e6:.1f}M book, 1-day 95 percent VaR is ${risk.var95 / 1000:.0f}k. "
            f"Annualized vol {risk.volAnn * 100:.1f} percent. A 12 percent BTC shock maps to {sc.pct * 100:.1f} percent P and L via empirical betas, not a flat 12."
        )
    if intent == "eval":
        return (
            f"EvalForge production reranker: nDCG@10 {float(numbers.get('ndcgAt10', 0)) * 100:.0f} percent, "
            f"Recall@5 {float(numbers.get('recallAt5', 0)) * 100:.0f} percent. Promotion still requires groundedness, p95 latency and cost gates."
        )
    if intent == "training":
        return f"Registry production checkpoint is {numbers.get('production', 'rerank-lora-r8')}. LoRA rank-8 is a logistic probe on retrieval features — the same promotion loop you would use for a real adapter."
    if retrieved:
        cite = retrieved[0]
        return f"{cite.doc.title}. {cite.doc.body[:220]}"
    if snap:
        return f"{ticker} last {snap.price:.2f}, regime {snap.regime}. Ask for a regime read, anomalies, a backtest, or portfolio risk."
    return "Helix is the voice layer over market tools, RAG, evals and the model registry. Ask a market question."
