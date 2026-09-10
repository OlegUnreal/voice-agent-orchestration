import json

import httpx

from helix.oktrader.live import pick_tools
from helix.oktrader.mcp_client import McpClient, _parse_sse, is_write_tool


def test_parse_sse():
    body = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
    assert _parse_sse(body)["result"]["ok"] is True


def test_write_hints():
    assert is_write_tool("futures_place_order")
    assert is_write_tool("cancel_all")
    assert not is_write_tool("futures_mark_price")
    assert not is_write_tool("list_portfolio_projects")


def test_pick_skips_write_tools():
    catalog = [
        {"name": "futures_place_order", "description": "submit order"},
        {"name": "futures_mark_price", "description": "mark price ticker"},
        {"name": "futures_position_risk", "description": "open positions"},
    ]
    picked = pick_tools(catalog, "regime")
    names = [t["name"] for t in picked]
    assert "futures_place_order" not in names
    assert "futures_mark_price" in names


def test_client_initialize_and_call():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        calls.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "sess-1", "Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"protocolVersion": "2025-06-18", "serverInfo": {"name": "hub"}},
                },
            )
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202, headers={"Mcp-Session-Id": "sess-1"})
        if payload["method"] == "tools/list":
            assert request.headers.get("mcp-session-id") == "sess-1"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {"name": "futures_mark_price", "description": "mark price"},
                            {"name": "futures_place_order", "description": "place order"},
                        ]
                    },
                },
            )
        if payload["method"] == "tools/call":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"content": [{"type": "text", "text": "BTCUSDT 110250"}]},
                },
            )
        return httpx.Response(400, json={"error": "unknown"})

    transport = httpx.MockTransport(handler)
    client = McpClient("https://hub.example/mcp", transport=transport)
    tools = client.list_tools()
    assert [t["name"] for t in tools] == ["futures_mark_price", "futures_place_order"]
    out = client.call_tool("futures_mark_price", {"symbol": "BTCUSDT"})
    assert "110250" in out["content"][0]["text"]
    assert "initialize" in calls
    client.close()
