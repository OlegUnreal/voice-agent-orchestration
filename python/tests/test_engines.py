from helix.evals import dataset_version, gate_check, run_eval_suite
from helix.market import backtest_sma, detect_regime, estimate_risk, snapshot
from helix.orchestrator import run_orchestrator
from helix.rag import retrieve
from helix.router import classify_intent
from helix.training import train_adapters


def test_snapshot_and_regime():
    s = snapshot("BTC")
    assert s.price > 0
    assert s.regime in {"bull-trend", "bear-trend", "range", "high-vol"}
    assert detect_regime("BTC") == s.regime


def test_backtest_and_risk():
    bt = backtest_sma("ETH")
    assert bt.trades >= 0
    assert 0 <= bt.maxDrawdown <= 1
    risk = estimate_risk()
    assert risk.var95 > 0
    assert risk.parametricVar95 is not None
    assert len(risk.scenarioPnl) == 3


def test_intent_and_rag():
    assert classify_intent("What is BTC's current regime?") == "regime"
    assert classify_intent("Portfolio risk if BTC drops 12 percent") == "risk"
    hits = retrieve("BTC exchange inflow anomalies", k=5)
    assert hits
    assert any("BTC" in h.doc.id or h.doc.ticker == "BTC" for h in hits)


def test_orchestrator_turn():
    r = run_orchestrator("What is BTC's current regime?")
    assert r.intent == "regime"
    assert r.tools
    assert r.grounded
    assert r.fallbackSpoken


def test_eval_suite_and_dataset_version():
    m = run_eval_suite("embed")
    assert 0 <= m.ndcgAt10 <= 1
    assert 0 <= m.recallAt5 <= 1
    assert dataset_version().startswith("ds-golden-")
    gates = gate_check(m)
    assert len(gates) == 5


def test_train_adapters():
    out = train_adapters(epochs_lora=8, epochs_qlora=6)
    assert out["pairs"] > 0
    assert out["loraLoss"]
