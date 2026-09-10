from helix.finetune import export_trl, train_peft
from helix.graph import resume_graph, run_graph
from helix.ledger import log_feedback, log_turn
from helix.narrate import narrate
from helix.serving import log_serving, serving_report
from helix.snapshot import snapshot
from helix.experiments import list_experiments, log_experiment


def test_serving_report():
    log_serving(
        {
            "traceId": "tr-1",
            "model": "grok-4.5",
            "ttftMs": 120,
            "totalMs": 400,
            "tokensIn": 80,
            "tokensOut": 60,
            "costUsd": 0.001,
        }
    )
    rep = serving_report()
    assert rep["n"] >= 1
    assert "ttftP95Ms" in rep


def test_snapshot_and_experiment():
    log_turn({"query": "q", "intent": "general", "tools": [], "spoken": "ok"})
    meta = snapshot("test")
    assert meta["id"].startswith("ds-live-")
    rec = log_experiment("evalforge", {"kind": "embed"}, {"ndcgAt10": 0.8}, dataset=meta["id"])
    assert rec["name"] == "evalforge"
    assert list_experiments()


def test_graph_blocks_writes_for_hitl():
    body = run_graph("Place a market buy of 2 BTC now")
    assert body["status"] == "awaiting_approval"
    assert body["write_blocked"] is True
    tid = body["thread_id"]
    resumed = resume_graph(tid, approved=True)
    assert resumed["executed"] is False
    assert resumed["ok"] is True


def test_narrate_falls_back_without_keys(monkeypatch):
    monkeypatch.delenv("HELIX_VLLM_URL", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("HELIX_XAI_API_KEY", raising=False)
    out = narrate("regime?", {"numbers": {"price": 1.0}, "ticker": "BTC"}, "BTC last 1.0 from tools.")
    assert out["route"] == "local-tools"
    assert "1.0" in out["spoken"]


def test_trl_export_from_feedback():
    tid = log_turn({"query": "hi", "intent": "general", "tools": [], "spoken": "hello"})
    log_feedback(tid, "up", "hello", "hi")
    log_feedback(tid, "down", "bad", "hi", correction="BTC last 1 from tools.")
    dumped = export_trl()
    assert dumped["sftPairs"] >= 1
    assert dumped["dpoPairs"] >= 1
    peft = train_peft(export_only=True)
    assert peft["trained"] is False
