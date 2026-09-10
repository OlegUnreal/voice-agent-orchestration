"""Model routing for speech: vLLM → xAI → tool fallback. JSON envelope = structured output."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from helix.serving import estimate_cost, estimate_tokens, log_serving
from helix.verifier import enforce

SYSTEM = (
    "You are Helix, a voice financial copilot. TOOL RESULTS are the only source of numbers. "
    "Never invent prices, Sharpe, VaR, or citations. Reply with JSON only: "
    '{"spoken":"2-4 plain sentences then a recap","citations":["id"],"used":["BTC"]} '
    "No markdown. If a number is missing from tools, say you do not have it."
)


class SpokenEnvelope(BaseModel):
    spoken: str = Field(min_length=1, max_length=1200)
    citations: list[str] = Field(default_factory=list)
    used: list[str] = Field(default_factory=list)


def _parse_envelope(raw: str, fallback: str) -> SpokenEnvelope:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return SpokenEnvelope.model_validate_json(text)
    except ValidationError:
        pass
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return SpokenEnvelope.model_validate_json(text[start : end + 1])
    except ValidationError:
        pass
    return SpokenEnvelope(spoken=fallback or text[:800])


def _stream_chat(url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[str, float, bool]:
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    with httpx.Client(timeout=float(os.environ.get("HELIX_NARRATE_TIMEOUT", "8"))) as client:
        with client.stream("POST", url, headers=headers, json={**body, "stream": True}) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"narrate {resp.status_code}")
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        parts.append(piece)
                except (KeyError, json.JSONDecodeError, IndexError):
                    continue
    total_wait = (time.perf_counter() - t0) * 1000
    return "".join(parts), (ttft if ttft is not None else total_wait), True


def _post_chat(url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[str, float, bool]:
    t0 = time.perf_counter()
    with httpx.Client(timeout=float(os.environ.get("HELIX_NARRATE_TIMEOUT", "8"))) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    return raw or "", (time.perf_counter() - t0) * 1000, False


def _complete(url: str, token: str | None, model: str, user: str) -> tuple[str, float, bool, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 420,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    try:
        raw, ttft, streamed = _stream_chat(url, headers, body)
        if raw.strip():
            return raw, ttft, streamed, model
    except Exception:
        pass
    raw, ttft, streamed = _post_chat(url, headers, body)
    return raw, ttft, streamed, model


def narrate(query: str, payload: dict[str, Any], fallback: str, trace_id: str | None = None) -> dict[str, Any]:
    user = f"User said: {query}\n\nTOOL RESULTS (authoritative JSON):\n{json.dumps(payload, default=str)[:8000]}"
    t0 = time.perf_counter()
    raw = ""
    model = "local-tools"
    ttft = 0.0
    streamed = False
    vllm = (os.environ.get("HELIX_VLLM_URL") or "").rstrip("/")
    # Ambient XAI_API_KEY is for the voice UI. Engines only call xAI when explicitly opted in,
    # otherwise every /v1/narrate hangs the turn if the vendor is slow or blocked.
    xai = os.environ.get("HELIX_XAI_API_KEY") or (
        os.environ.get("XAI_API_KEY") if os.environ.get("HELIX_NARRATE_XAI", "").lower() in {"1", "true", "yes"} else None
    )
    try:
        if vllm:
            raw, ttft, streamed, model = _complete(
                f"{vllm}/chat/completions" if not vllm.endswith("completions") else vllm,
                os.environ.get("HELIX_VLLM_TOKEN"),
                os.environ.get("HELIX_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                user,
            )
            model = f"vllm:{os.environ.get('HELIX_VLLM_MODEL', 'local')}"
        elif xai:
            raw, ttft, streamed, model = _complete(
                "https://api.x.ai/v1/chat/completions",
                xai,
                os.environ.get("HELIX_XAI_MODEL", "grok-4.5"),
                user,
            )
    except Exception:
        raw = ""
        model = "local-tools"
    envelope = _parse_envelope(raw, fallback) if raw else SpokenEnvelope(spoken=fallback)
    checked = enforce(envelope.spoken, payload, fallback)
    spoken = checked["spoken"] or fallback
    total = (time.perf_counter() - t0) * 1000
    tok_in = estimate_tokens(user)
    tok_out = estimate_tokens(spoken)
    serving = log_serving(
        {
            "traceId": trace_id,
            "model": model if checked["ok"] else "verified-fallback",
            "ttftMs": ttft,
            "totalMs": total,
            "tokensIn": tok_in,
            "tokensOut": tok_out,
            "costUsd": estimate_cost(tok_in, tok_out, model),
            "failed": not checked["ok"] or not raw,
            "streaming": streamed,
        }
    )
    return {
        "spoken": spoken,
        "draftSpoken": envelope.spoken,
        "model": serving["model"],
        "verified": checked["ok"],
        "leaks": checked["leaks"],
        "structured": envelope.model_dump(),
        "serving": serving,
        "route": "vllm" if vllm and raw else ("xai" if xai and raw else "local-tools"),
    }
