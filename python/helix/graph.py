"""HITL graph: same tools as the supervisor, interrupt on write. Not ReAct — LangGraph for durable approve."""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from helix.orchestrator import run_orchestrator
from helix.store import append_jsonl, read_jsonl

PENDING = "graph_pending.jsonl"


class HelixState(TypedDict, total=False):
    query: str
    thread_id: str
    write_blocked: bool
    status: str
    result: dict[str, Any]


def _compile():
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    def tools_node(state: HelixState) -> HelixState:
        result = run_orchestrator(state["query"])
        blocked = bool((result.numbers or {}).get("writeBlocked") or (result.numbers or {}).get("secretBlocked"))
        return {
            "result": result.model_dump(),
            "write_blocked": blocked,
            "status": "awaiting_approval" if blocked else "ok",
        }

    g = StateGraph(HelixState)
    g.add_node("tools", tools_node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile(checkpointer=MemorySaver())


_APP = None


def graph_app():
    global _APP
    if _APP is None:
        _APP = _compile()
    return _APP


def _pending_put(thread_id: str, payload: dict[str, Any]) -> None:
    append_jsonl(PENDING, {"thread_id": thread_id, "ts": int(time.time() * 1000), **payload})


def _pending_get(thread_id: str) -> dict[str, Any] | None:
    rows = [r for r in read_jsonl(PENDING) if r.get("thread_id") == thread_id]
    return rows[-1] if rows else None


def run_graph(query: str, thread_id: str | None = None) -> dict[str, Any]:
    tid = thread_id or f"th-{uuid.uuid4().hex[:10]}"
    app = graph_app()
    if app is not None:
        out = app.invoke({"query": query, "thread_id": tid}, config={"configurable": {"thread_id": tid}})
        body = dict(out)
    else:
        result = run_orchestrator(query)
        blocked = bool((result.numbers or {}).get("writeBlocked") or (result.numbers or {}).get("secretBlocked"))
        body = {
            "query": query,
            "thread_id": tid,
            "result": result.model_dump(),
            "write_blocked": blocked,
            "status": "awaiting_approval" if blocked else "ok",
        }
    body["thread_id"] = tid
    body["backend"] = "langgraph" if app is not None else "inline"
    if body.get("status") == "awaiting_approval":
        _pending_put(tid, {"query": query, "result": body.get("result")})
    result = body.get("result") or {}
    if isinstance(result, dict) and "grokPayload" not in result:
        # orchestrator dump already; payload for UI
        pass
    return body


def resume_graph(thread_id: str, approved: bool) -> dict[str, Any]:
    pending = _pending_get(thread_id)
    if not pending:
        return {"ok": False, "error": "unknown thread"}
    # Helix still never fires write tools. Approve only releases the refusal speech.
    result = pending.get("result") or {}
    spoken = (
        result.get("fallbackSpoken")
        if not approved
        else result.get("fallbackSpoken")
    )
    return {
        "ok": True,
        "approved": approved,
        "thread_id": thread_id,
        "status": "ok",
        "executed": False,
        "reason": "write tools stay approve-only even after HITL resume",
        "spoken": spoken,
        "result": result,
    }


def grok_from_graph(body: dict[str, Any]) -> dict[str, Any] | None:
    raw = body.get("result")
    if not isinstance(raw, dict):
        return None
    return raw.get("grokPayload") or raw
