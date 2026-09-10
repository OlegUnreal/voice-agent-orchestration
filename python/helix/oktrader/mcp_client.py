"""Streamable HTTP MCP client (protocol 2025-06-18).

Talks to Project Hub the same way a manual curl session does:
initialize → Mcp-Session-Id → tools/list → tools/call.
Never logs tokens. Write tools are the caller's problem to gate.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any
from urllib.parse import urlparse

import httpx

PROTOCOL = os.environ.get("HELIX_MCP_PROTOCOL") or os.environ.get("OK_TRADER_MCP_PROTOCOL", "2025-06-18")
DEFAULT_TIMEOUT = float(os.environ.get("HELIX_MCP_TIMEOUT") or os.environ.get("OK_TRADER_MCP_TIMEOUT", "8"))

WRITE_HINTS = (
    "order",
    "cancel",
    "buy",
    "sell",
    "place",
    "execute",
    "kill",
    "redeem",
    "leverage",
    "margin",
    "bind",
    "enable",
    "activate",
    "deactivate",
    "close_position",
    "set_",
)


def is_write_tool(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in WRITE_HINTS)


def _parse_sse(text: str) -> dict[str, Any] | None:
    payload: list[str] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload.append(line[5:].lstrip())
        elif line.strip() == "" and payload:
            break
    if not payload:
        return None
    raw = "\n".join(payload).strip()
    if not raw:
        return None
    return json.loads(raw)


def _decode_body(content_type: str, text: str) -> dict[str, Any]:
    ct = (content_type or "").lower()
    if "text/event-stream" in ct or text.lstrip().startswith("event:"):
        parsed = _parse_sse(text)
        if parsed is None:
            raise RuntimeError("empty SSE body from MCP")
        return parsed
    return json.loads(text)


class McpClient:
    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.session_id: str | None = None
        self.server_info: dict[str, Any] = {}
        self._n = 0
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL,
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _rpc(self, method: str, params: dict[str, Any] | None = None, notification: bool = False) -> dict[str, Any]:
        with self._lock:
            self._n += 1
            n = self._n
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            body["id"] = n
        if params is not None:
            body["params"] = params
        res = self._http.post(self.url, headers=self._headers(), json=body)
        sid = res.headers.get("mcp-session-id") or res.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        if notification:
            return {}
        if res.status_code >= 400:
            raise RuntimeError(f"MCP HTTP {res.status_code}: {res.text[:300]}")
        if not res.text.strip():
            return {}
        payload = _decode_body(res.headers.get("content-type", ""), res.text)
        if "error" in payload:
            err = payload["error"]
            raise RuntimeError(f"MCP {method} error {err.get('code')}: {err.get('message')}")
        return payload.get("result") or {}

    def initialize(self) -> dict[str, Any]:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "helix", "version": "0.1.0"},
            },
        )
        self.server_info = result.get("serverInfo") or {}
        self._rpc("notifications/initialized", notification=True)
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        if not self.session_id:
            self.initialize()
        result = self._rpc("tools/list", {})
        return list(result.get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self.session_id:
            self.initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if isinstance(result, dict) and result.get("isError"):
            raise RuntimeError(f"tool {name} returned isError")
        return result


_client: McpClient | None = None
_client_lock = threading.Lock()


def configured_url() -> str | None:
    url = (os.environ.get("HELIX_MCP_URL") or os.environ.get("OK_TRADER_MCP_URL") or "").strip()
    return url or None


def get_client() -> McpClient | None:
    url = configured_url()
    if not url:
        return None
    host = urlparse(url).hostname or ""
    global _client
    with _client_lock:
        if _client is None or _client.url != url.rstrip("/"):
            if _client is not None:
                _client.close()
            _client = McpClient(
                url,
                token=os.environ.get("HELIX_MCP_TOKEN") or os.environ.get("OK_TRADER_MCP_TOKEN") or None,
            )
        return _client
