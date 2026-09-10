from fastapi.testclient import TestClient

from helix.gateway.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_turn_and_cache():
    body = {"text": "What is BTC's current regime?"}
    a = client.post("/v1/turn", json=body)
    b = client.post("/v1/turn", json=body)
    assert a.status_code == 200
    assert a.json()["intent"] == "regime"
    assert b.json()["cacheHit"] is True


def test_mcp_list_and_approve_gate():
    listed = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert listed.status_code == 200
    names = [t["name"] for t in listed.json()["result"]["tools"]]
    assert "propose_rebalance" in names
    blocked = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "propose_rebalance", "arguments": {"ticker": "BTC"}},
        },
    )
    assert blocked.json()["error"]["code"] == 403


def test_verify_and_feedback():
    turn = client.post("/v1/turn", json={"text": "What is BTC's current regime?"})
    assert turn.status_code == 200
    body = turn.json()
    assert body["verified"] is True
    tid = body["traceId"]
    bad = client.post(
        "/v1/verify",
        json={
            "spoken": "BTC last traded at 888888.25 dollars",
            "fallbackSpoken": body["fallbackSpoken"],
            "payload": body["grokPayload"],
            "traceId": tid,
            "query": "price",
        },
    )
    assert bad.status_code == 200
    assert bad.json()["ok"] is False
    fb = client.post(
        "/v1/feedback",
        json={"traceId": tid, "verdict": "up", "spoken": body["fallbackSpoken"], "query": "regime"},
    )
    assert fb.json()["ok"] is True
    mem = client.post("/v1/memory", json={"key": "cap", "value": "1 USDT"})
    assert mem.status_code == 200
    listed = client.get("/v1/memory")
    assert listed.json()["facts"]
