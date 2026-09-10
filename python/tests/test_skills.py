from helix.ledger import export_dataset, log_feedback, log_turn
from helix.memory import parse_remember, recall, remember
from helix.orchestrator import run_orchestrator
from helix.redteam import run_redteam
from helix.router import classify_intent
from helix.sanitize import scrub_text
from helix.teacher import log_teacher
from helix.verifier import enforce, verify_spoken


def test_verifier_allows_tool_numbers():
    spoken = "BTC last 70123.45 in a high-vol regime."
    payload = {"numbers": {"price": 70123.45, "regime": "high-vol"}, "ticker": "BTC"}
    assert verify_spoken(spoken, payload)["ok"] is True


def test_verifier_blocks_invented_price():
    spoken = "BTC last 999999.0 today."
    payload = {"numbers": {"price": 70123.45}, "ticker": "BTC"}
    check = enforce(spoken, payload, "BTC last 70123.45 from tools.")
    assert check["ok"] is False
    assert "999999" in "".join(check["leaks"])
    assert "999999" not in check["spoken"]


def test_memory_roundtrip():
    remember("risk cap", "max net risk 1 USDT isolated 1x", source="user")
    hits = recall("what is my risk cap")
    assert hits
    assert "1 USDT" in hits[0]["value"]
    assert hits[0]["source"] == "user"


def test_remember_intent_and_parse():
    assert classify_intent("Remember that isolated 1x only") == "memory"
    assert parse_remember("remember that isolated 1x only")[1].startswith("isolated")


def test_turn_logs_trace_and_memory_tool():
    r = run_orchestrator("What is BTC's current regime?")
    assert r.traceId
    assert r.verified
    assert any(t.name == "memory_recall" for t in r.tools)
    assert export_dataset()["traces"] >= 1


def test_feedback_writes_sft():
    tid = log_turn({"query": "hi", "intent": "general", "tools": [], "spoken": "hello"})
    log_feedback(tid, "up", "hello", "hi")
    ds = export_dataset()
    assert ds["sftPairs"] >= 1


def test_dpo_needs_correction():
    tid = log_turn({"query": "price", "intent": "regime", "tools": [], "spoken": "bad"})
    log_feedback(tid, "down", "bad", "price", correction="BTC last 70123.45 from tools.")
    assert export_dataset()["dpoPairs"] >= 1


def test_scrub_secrets():
    assert "[redacted]" in scrub_text("token Bearer abcdefghijklmnop and a@b.co")
    assert "BTC" in scrub_text("BTC last 70123")


def test_teacher_distill():
    log_teacher("tr-1", "price?", "BTC is 888888.25", "BTC last 70123.45 from tools.", "grok-4.5", ["888888.25"])
    ds = export_dataset()
    assert ds["teachers"] >= 1
    assert ds["distillPairs"] >= 1


def test_redteam_blocks_writes_and_inventions():
    report = run_redteam()
    failed = [c for c in report["cases"] if not c["ok"]]
    assert report["total"] == 12
    assert report["rate"] >= 0.8, failed
    assert all(c["ok"] for c in report["cases"] if c["kind"] == "order"), failed
