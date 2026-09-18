"""FastAPI + Pydantic gateway. Structured outputs, cache, MCP registry, routing."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from helix.evals import GOLDEN, dataset_version, gate_check, run_eval_suite
from helix.drift import drift_report
from helix.ab_test import run_ab_test
from helix.gateway.cache import TtlCache
from helix.market import (
    all_snapshots,
    backtest_sma,
    detect_anomalies,
    estimate_risk,
    get_bars,
    snapshot,
)
from helix.experiments import list_experiments, log_experiment
from helix.finetune import export_trl, train_peft
from helix.graph import resume_graph, run_graph
from helix.ledger import export_dataset, list_traces, log_feedback, log_verify_fail
from helix.narrate import narrate
from helix.serving import log_serving, serving_report
from helix.snapshot import list_snapshots, snapshot as take_snapshot
from helix.redteam import run_redteam
from helix.teacher import log_teacher
from helix.memory import forget, list_facts, remember, recall
from helix.mcp import tools_as_json
from helix.models import (
    Checkpoint,
    PromoteRequest,
    RetrieveRequest,
    RetrieverKind,
    Ticker,
    TrainRequest,
    TurnRequest,
)
from helix.oktrader.live import live_status
from helix.orchestrator import grok_tool_payload, run_orchestrator
from helix.rag import lexical_retrieve, retrieve
from helix.traces import SEED_TRACES
from helix.training import (
    adapter_loss,
    default_checkpoints,
    get_adapter_weights,
    promote,
    train_adapters,
)
from helix.verifier import enforce

app = FastAPI(
    title="Helix AI Gateway",
    version="0.1.0",
    description="Python FastAPI gateway for Helix voice-agent orchestration.",
)
_cache = TtlCache(ttl_s=600)
_ckpts: list[Checkpoint] = []


def _checkpoints() -> list[Checkpoint]:
    global _ckpts
    if not _ckpts:
        _ckpts = default_checkpoints()
    return _ckpts


class VerifyRequest(BaseModel):
    spoken: str
    fallbackSpoken: str
    payload: dict[str, Any] = {}
    traceId: str | None = None
    query: str = ""
    model: str = "local-tools"


class FeedbackRequest(BaseModel):
    traceId: str
    verdict: Literal["up", "down"]
    spoken: str = ""
    query: str = ""
    correction: str | None = None


class MemoryWrite(BaseModel):
    key: str
    value: str
    source: str = "user"


class RouteDecision(BaseModel):
    model: Literal["local-tools", "grok-4.5", "grok-4.5-cache", "vllm"]
    reason: str


def route_model(has_tools: bool, cache_hit: bool) -> RouteDecision:
    if cache_hit:
        return RouteDecision(model="grok-4.5-cache", reason="identical query TTL")
    if has_tools:
        return RouteDecision(model="local-tools", reason="numbers from engines; LLM synthesizes later")
    return RouteDecision(model="grok-4.5", reason="open synthesis")


@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    mcp = live_status()
    return {
        "ok": True,
        "service": "helix-engines",
        "dataset": dataset_version(),
        "mcp": {"configured": mcp.get("configured"), "ok": mcp.get("ok"), "count": mcp.get("count", 0)},
        "embeddings": os.environ.get("HELIX_EMBEDDINGS", "hash"),
        "vllm": bool(os.environ.get("HELIX_VLLM_URL")),
        "postgres": bool(os.environ.get("DATABASE_URL") or os.environ.get("HELIX_DATABASE_URL")),
    }


@app.get("/v1/live/status")
def live() -> dict[str, Any]:
    return live_status()


@app.post("/v1/turn")
def turn(req: TurnRequest) -> dict[str, Any]:
    key = req.text.strip().lower()
    hit = _cache.get(key)
    if hit:
        payload = dict(hit)
        payload["route"] = route_model(True, True).model_dump()
        payload["cacheHit"] = True
        return payload
    result = run_orchestrator(req.text)
    body = result.model_dump()
    body["grokPayload"] = grok_tool_payload(result)
    body["route"] = route_model(True, False).model_dump()
    body["cacheHit"] = False
    _cache.set(key, body)
    return body


@app.post("/v1/turn/stream")
def turn_stream(req: TurnRequest):
    """Server-Sent Events streaming endpoint for real-time voice-agent responses.
    
    Streams the response in chunks: thinking → tools → spoken → citations.
    This is the production path for voice UIs that want progressive rendering
    instead of waiting for the full response. Each event is a JSON object with
    a `type` field: `thinking`, `tool`, `spoken`, `citation`, `done`.
    """
    from fastapi.responses import StreamingResponse
    import json
    import time
    
    def event_stream():
        start = time.time()
        
        # Phase 1: thinking (intent classification)
        yield f"data: {json.dumps({'type': 'thinking', 'content': 'Classifying intent...', 'elapsed_ms': int((time.time() - start) * 1000)})}\n\n"
        time.sleep(0.1)
        
        # Phase 2: run orchestrator
        result = run_orchestrator(req.text)
        body = result.model_dump()
        
        # Phase 3: tools used
        for tool in (body.get("used") or []):
            yield f"data: {json.dumps({'type': 'tool', 'name': tool, 'elapsed_ms': int((time.time() - start) * 1000)})}\n\n"
        
        # Phase 4: spoken response (chunked for streaming TTS)
        spoken = body.get("spoken", "")
        chunk_size = 50
        for i in range(0, len(spoken), chunk_size):
            chunk = spoken[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'spoken', 'content': chunk, 'elapsed_ms': int((time.time() - start) * 1000)})}\n\n"
            time.sleep(0.05)
        
        # Phase 5: citations
        for cite in (body.get("citations") or []):
            yield f"data: {json.dumps({'type': 'citation', 'docId': cite, 'elapsed_ms': int((time.time() - start) * 1000)})}\n\n"
        
        # Phase 6: done
        yield f"data: {json.dumps({'type': 'done', 'elapsed_ms': int((time.time() - start) * 1000), 'cacheHit': False})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/v1/ws/turn")
async def turn_websocket(websocket):
    """WebSocket endpoint for real-time bidirectional voice-agent communication.
    
    Client sends: {"text": "query"}
    Server streams: {"type": "thinking|tool|spoken|citation|done", ...}
    
    This is the production path for voice UIs that need full-duplex communication:
    interrupt handling, barge-in, and progressive TTS rendering. Each message is
    a JSON object with a `type` field.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            
            if not text:
                await websocket.send_json({"type": "error", "message": "empty text"})
                continue
            
            # Stream response phases
            import time
            start = time.time()
            
            await websocket.send_json({"type": "thinking", "content": "Classifying intent...", "elapsed_ms": int((time.time() - start) * 1000)})
            
            result = run_orchestrator(text)
            body = result.model_dump()
            
            for tool in (body.get("used") or []):
                await websocket.send_json({"type": "tool", "name": tool, "elapsed_ms": int((time.time() - start) * 1000)})
            
            spoken = body.get("spoken", "")
            chunk_size = 50
            for i in range(0, len(spoken), chunk_size):
                chunk = spoken[i:i + chunk_size]
                await websocket.send_json({"type": "spoken", "content": chunk, "elapsed_ms": int((time.time() - start) * 1000)})
            
            for cite in (body.get("citations") or []):
                await websocket.send_json({"type": "citation", "docId": cite, "elapsed_ms": int((time.time() - start) * 1000)})
            
            await websocket.send_json({"type": "done", "elapsed_ms": int((time.time() - start) * 1000)})
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/v1/market/book")
def market_book(ticker: Ticker = "BTC") -> dict[str, Any]:
    bars = get_bars(ticker)
    bt = backtest_sma(ticker)
    return {
        "snapshots": [s.model_dump() for s in all_snapshots()],
        "ticker": ticker,
        "bars": [b.model_dump() for b in bars[-120:]],
        "anomalies": [a.model_dump() for a in detect_anomalies(ticker)],
        "backtest": bt.model_dump(),
        "risk": estimate_risk().model_dump(),
        "snapshot": snapshot(ticker).model_dump(),
    }


@app.post("/v1/rag/retrieve")
def rag_retrieve(req: RetrieveRequest) -> dict[str, Any]:
    w = get_adapter_weights(req.kind)
    hybrid = retrieve(req.query, k=req.k, chronological=req.chronological, weights=w)
    lex = lexical_retrieve(req.query, req.k)
    return {
        "hybrid": [r.model_dump() for r in hybrid],
        "lexical": [r.model_dump() for r in lex],
        "corpusSize": 18,
    }


@app.get("/v1/evals")
def evals(kind: RetrieverKind = "lora") -> dict[str, Any]:
    w = get_adapter_weights(kind)
    metrics = run_eval_suite(kind, w)
    gates = gate_check(metrics)
    log_experiment(
        "evalforge",
        {"kind": kind},
        metrics.model_dump(),
        dataset=dataset_version(),
    )
    return {
        "metrics": metrics.model_dump(),
        "gates": gates,
        "golden": [c.model_dump() for c in GOLDEN],
        "datasetVersion": dataset_version(),
        "kind": kind,
    }


@app.get("/v1/drift")
def drift() -> dict[str, Any]:
    """Model drift detection: metric, serving, and input distribution monitoring.

    Compares recent window against historical baseline for each category.
    Flags metrics where current value is >2σ from historical mean.
    """
    return drift_report()


class ABTestRequest(BaseModel):
    kindA: str = "lora"
    kindB: str = "hybrid"
    seed: int = 0


@app.post("/v1/ab-test")
def ab_test(req: ABTestRequest) -> dict[str, Any]:
    """A/B test between two reranker variants over the golden set.

    Deterministic assignment by case ID hash. Returns per-variant metrics,
    per-case assignment, and paired permutation test significance (p < 0.05).
    """
    return run_ab_test(kind_a=req.kindA, kind_b=req.kindB, seed=req.seed)


class NarrateRequest(BaseModel):
    query: str
    fallbackSpoken: str
    payload: dict[str, Any] = {}
    traceId: str | None = None


class GraphRequest(BaseModel):
    query: str
    threadId: str | None = None


class GraphResume(BaseModel):
    threadId: str
    approved: bool = False


class ServingEvent(BaseModel):
    traceId: str | None = None
    model: str = "local-tools"
    ttftMs: float = 0
    totalMs: float = 0
    tokensIn: int = 0
    tokensOut: int = 0
    costUsd: float = 0
    failed: bool = False
    streaming: bool = False


@app.post("/v1/narrate")
def narrate_route(req: NarrateRequest) -> dict[str, Any]:
    return narrate(req.query, req.payload, req.fallbackSpoken, req.traceId)


@app.post("/v1/graph")
def graph_route(req: GraphRequest) -> dict[str, Any]:
    return run_graph(req.query, req.threadId)


@app.post("/v1/graph/resume")
def graph_resume_route(req: GraphResume) -> dict[str, Any]:
    return resume_graph(req.threadId, req.approved)


@app.post("/v1/serving")
def serving_post(req: ServingEvent) -> dict[str, Any]:
    return log_serving(req.model_dump())


@app.get("/v1/serving")
def serving_get() -> dict[str, Any]:
    return serving_report()


@app.post("/v1/dataset/snapshot")
def dataset_snapshot(tag: str = "") -> dict[str, Any]:
    return take_snapshot(tag or None)


@app.get("/v1/dataset/snapshots")
def dataset_snapshots() -> dict[str, Any]:
    return {"snapshots": list_snapshots()}


@app.get("/v1/experiments")
def experiments_get() -> dict[str, Any]:
    return {"experiments": list_experiments()}


@app.post("/v1/finetune/export")
def finetune_export() -> dict[str, Any]:
    return export_trl()


@app.post("/v1/finetune/train")
def finetune_train() -> dict[str, Any]:
    return train_peft(export_only=True)


@app.get("/v1/evals/redteam")
def evals_redteam() -> dict[str, Any]:
    report = run_redteam()
    log_experiment("redteam", {}, {"rate": report.get("rate"), "passed": report.get("passed")})
    return report


@app.post("/v1/training/train")
def training_train(req: TrainRequest) -> dict[str, Any]:
    global _ckpts
    result = train_adapters(req.epochs_lora, req.epochs_qlora)
    _ckpts = default_checkpoints()
    return {**result, "checkpoints": [c.model_dump() for c in _ckpts]}


@app.get("/v1/training/checkpoints")
def training_ckpts() -> dict[str, Any]:
    return {"checkpoints": [c.model_dump() for c in _checkpoints()], "loss": adapter_loss()}


@app.post("/v1/training/promote")
def training_promote(req: PromoteRequest) -> dict[str, Any]:
    global _ckpts
    _ckpts = promote(_checkpoints(), req.checkpoint_id)
    found = next((c for c in _ckpts if c.id == req.checkpoint_id), None)
    if not found:
        raise HTTPException(404, "unknown checkpoint")
    return {"checkpoints": [c.model_dump() for c in _ckpts]}


@app.post("/v1/verify")
def verify(req: VerifyRequest) -> dict[str, Any]:
    checked = enforce(req.spoken, req.payload, req.fallbackSpoken)
    if not checked["ok"]:
        log_verify_fail(
            {
                "traceId": req.traceId,
                "query": req.query,
                "spoken": req.spoken,
                "leaks": checked["leaks"],
            }
        )
    log_teacher(
        req.traceId,
        req.query,
        req.spoken,
        checked["spoken"] or req.fallbackSpoken,
        req.model,
        checked["leaks"],
    )
    return checked


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    rec = log_feedback(
        req.traceId,
        req.verdict,
        req.spoken,
        req.query,
        correction=req.correction,
    )
    return {"ok": True, "feedback": rec}


@app.get("/v1/memory")
def memory_get(q: str = "", k: int = 8) -> dict[str, Any]:
    facts = recall(q, k=k) if q.strip() else list_facts(k)
    return {"facts": facts}


@app.post("/v1/memory")
def memory_post(req: MemoryWrite) -> dict[str, Any]:
    return remember(req.key, req.value, source=req.source)


@app.delete("/v1/memory/{key}")
def memory_delete(key: str) -> dict[str, Any]:
    return {"deleted": forget(key)}


@app.get("/v1/dataset")
def dataset() -> dict[str, Any]:
    return export_dataset()


@app.get("/v1/gateway/traces")
def traces() -> dict[str, Any]:
    live_rows = list_traces(80)
    seed = [t.model_dump() for t in SEED_TRACES]
    return {"traces": live_rows + seed, "cache": _cache.stats(), "live": len(live_rows)}


@app.post("/mcp")
def mcp_jsonrpc(body: dict[str, Any]) -> dict[str, Any]:
    """Minimal JSON-RPC 2.0 surface for MCP-style clients."""
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_as_json()}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "get_market_snapshot":
            t = args.get("ticker", "BTC")
            return {"jsonrpc": "2.0", "id": req_id, "result": snapshot(t).model_dump()}
        if name == "estimate_risk":
            return {"jsonrpc": "2.0", "id": req_id, "result": estimate_risk().model_dump()}
        if name == "propose_rebalance":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": 403, "message": "propose_rebalance requires human approval"},
            }
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"unknown tool {name}"}}
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "method not found"}}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "helix.gateway.app:app",
        host=os.environ.get("HELIX_HOST", "127.0.0.1"),
        port=int(os.environ.get("HELIX_PORT", "8090")),
        reload=False,
    )
