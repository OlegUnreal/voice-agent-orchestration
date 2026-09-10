"""Map Helix intents onto discovered Project Hub / OK-Trader MCP tools."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from helix.models import ToolCall
from helix.oktrader.mcp_client import configured_url, get_client, is_write_tool

INTENT_NEEDLES: dict[str, tuple[str, ...]] = {
    "regime": ("price", "ticker", "klines", "candle", "mark", "index", "premium"),
    "anomaly": ("funding", "open_interest", "long_short", "liquidation", "oi"),
    "backtest": ("klines", "trades", "history", "candle"),
    "risk": ("risk", "preflight", "policy", "margin", "leverage", "notional"),
    "portfolio": ("position", "balance", "account", "portfolio", "wallet", "income"),
    "research": ("context", "decision", "portfolio_project", "project_"),
    "general": ("price", "ticker", "account", "position", "list_portfolio"),
}

# Never auto-invoked. Helix may name them in spoken text as "needs approval".
WRITE_ALWAYS = ("order", "cancel", "buy", "sell", "execute", "kill", "redeem")


@dataclass
class LiveTurn:
    ok: bool
    tools: list[ToolCall] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def allow_simulator() -> bool:
    return os.environ.get("HELIX_ALLOW_SIMULATOR", "true").lower() in {"1", "true", "yes"}


def _needles(intent: str) -> tuple[str, ...]:
    return INTENT_NEEDLES.get(intent, INTENT_NEEDLES["general"])


def _score_tool(name: str, description: str, intent: str) -> int:
    blob = f"{name} {description}".lower()
    if is_write_tool(name):
        return -100
    score = 0
    for n in _needles(intent):
        if n in blob:
            score += 3
    if "future" in blob:
        score += 1
    return score


def pick_tools(catalog: list[dict[str, Any]], intent: str, k: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        catalog,
        key=lambda t: _score_tool(str(t.get("name") or ""), str(t.get("description") or ""), intent),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for t in ranked:
        name = str(t.get("name") or "")
        if not name or is_write_tool(name):
            continue
        if _score_tool(name, str(t.get("description") or ""), intent) <= 0:
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out


def _args_for(tool: dict[str, Any], ticker: str, query: str) -> dict[str, Any]:
    schema = tool.get("inputSchema") or tool.get("schema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    args: dict[str, Any] = {}
    if not isinstance(props, dict):
        # Spring AI often uses camelCase symbol even without a published schema.
        name = str(tool.get("name") or "").lower()
        if any(k in name for k in ("price", "ticker", "klines", "depth", "book", "funding", "position")):
            args["symbol"] = _symbol(ticker)
        return args
    keys = {k.lower(): k for k in props}
    if "symbol" in keys:
        args[keys["symbol"]] = _symbol(ticker)
    if "query" in keys:
        args[keys["query"]] = query
    if "projectid" in keys:
        args[keys["projectid"]] = "ok-trader"
    if "limit" in keys:
        args[keys["limit"]] = 50
    if "interval" in keys:
        args[keys["interval"]] = "4h"
    return args


def _symbol(ticker: str) -> str:
    t = ticker.upper().replace("/", "")
    if t.endswith("USDT"):
        return t
    if t in {"BTC", "ETH", "SOL"}:
        return t + "USDT"
    return t


def _summarize(name: str, result: Any) -> str:
    text = json_preview(result)
    return f"{name}: {text[:180]}"


def json_preview(result: Any) -> str:
    if result is None:
        return "null"
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "content" in result and isinstance(result["content"], list):
            bits = []
            for c in result["content"]:
                if isinstance(c, dict) and c.get("type") == "text":
                    bits.append(str(c.get("text") or ""))
            if bits:
                return " ".join(bits)
        return str({k: result[k] for k in list(result)[:6]})
    return str(result)[:180]


def live_status() -> dict[str, Any]:
    url = configured_url()
    if not url:
        return {"configured": False, "ok": False, "tools": [], "writeTools": []}
    try:
        client = get_client()
        assert client is not None
        tools = client.list_tools()
        names = [str(t.get("name")) for t in tools]
        return {
            "configured": True,
            "ok": True,
            "urlHost": url.split("/")[2] if "://" in url else url,
            "server": client.server_info,
            "tools": names,
            "writeTools": [n for n in names if is_write_tool(n)],
            "count": len(names),
        }
    except Exception as exc:  # noqa: BLE001 — surface MCP reachability
        return {"configured": True, "ok": False, "error": str(exc)[:240], "tools": []}


def try_live_turn(query: str, intent: str, ticker: str) -> LiveTurn | None:
    if not configured_url():
        return None
    t0 = time.perf_counter()
    try:
        client = get_client()
        assert client is not None
        catalog = client.list_tools()
        chosen = pick_tools(catalog, intent)
        if not chosen:
            return LiveTurn(
                ok=True,
                numbers={"live": True, "liveToolsAvailable": [t.get("name") for t in catalog]},
                payload={"catalog": [t.get("name") for t in catalog]},
            )
        calls: list[ToolCall] = []
        blob: dict[str, Any] = {}
        for tool in chosen:
            name = str(tool["name"])
            args = _args_for(tool, ticker, query)
            c0 = time.perf_counter()
            result = client.call_tool(name, args)
            ms = (time.perf_counter() - c0) * 1000
            calls.append(
                ToolCall(
                    name=name,
                    args=args,
                    ms=ms,
                    ok=True,
                    summary=_summarize(name, result),
                )
            )
            blob[name] = result
        return LiveTurn(
            ok=True,
            tools=calls,
            numbers={
                "live": True,
                "liveSource": "project-hub-mcp",
                "liveToolCount": len(calls),
                "liveMs": (time.perf_counter() - t0) * 1000,
            },
            payload=blob,
        )
    except Exception as exc:  # noqa: BLE001
        return LiveTurn(ok=False, error=str(exc)[:300], numbers={"live": False})
