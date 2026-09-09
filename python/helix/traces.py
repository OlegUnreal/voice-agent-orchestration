"""Gateway traces — production hops become the next labeled dataset."""

from __future__ import annotations

from helix.models import Trace
from helix.settings import AS_OF

H = 3_600_000

SEED_TRACES: list[Trace] = [
    Trace(id="tr-10491", ts=AS_OF - 2 * H, query="BTC regime and last inflow anomaly", intent="regime", tools=["get_market_snapshot", "detect_regime", "retrieve_evidence"], latencyMs=412, tokens=640, costUsd=0.0041, retrievalHit=True, feedback="up"),
    Trace(id="tr-10488", ts=AS_OF - 5 * H, query="SOL sma backtest after the vol event", intent="backtest", tools=["run_backtest", "retrieve_evidence"], latencyMs=388, tokens=510, costUsd=0.0033, retrievalHit=True),
    Trace(id="tr-10481", ts=AS_OF - 9 * H, query="Why did the copilot cite NVDA on a BTC question", intent="research", tools=["retrieve_evidence"], latencyMs=290, tokens=420, costUsd=0.0026, retrievalHit=False, feedback="down"),
    Trace(id="tr-10476", ts=AS_OF - 14 * H, query="Book VaR and 12 percent BTC shock", intent="risk", tools=["estimate_risk"], latencyMs=176, tokens=380, costUsd=0.0021, retrievalHit=True, feedback="up"),
    Trace(id="tr-10470", ts=AS_OF - 20 * H, query="ETH staking flows", intent="research", tools=["retrieve_evidence"], latencyMs=254, tokens=460, costUsd=0.0029, retrievalHit=True),
    Trace(id="tr-10461", ts=AS_OF - 28 * H, query="Promote qlora if nDCG holds", intent="eval", tools=["get_eval_report", "get_checkpoint_status"], latencyMs=198, tokens=290, costUsd=0.0018, retrievalHit=True),
]
