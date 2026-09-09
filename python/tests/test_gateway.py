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


def test_evals_endpoint():
    r = client.get("/v1/evals", params={"kind": "embed"})
    assert r.status_code == 200
    assert "metrics" in r.json()
