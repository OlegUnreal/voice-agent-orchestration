"""ChainLens-style market intelligence: GBM tape, regimes, anomalies, SMA backtest, risk.

All math is pandas/NumPy. Prices are a seeded simulator, not a live venue — the
point is a deterministic, testable quant layer behind the voice agent.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from helix.models import (
    Anomaly,
    BacktestResult,
    Bar,
    MarketSnapshot,
    OnChainPoint,
    Regime,
    RiskResult,
    ScenarioPnl,
    Ticker,
    WeightRow,
)
from helix.prng import TICKER_SEEDS, box_muller, mulberry32
from helix.settings import (
    AS_OF,
    DEFAULT_BOOK,
    PORTFOLIO_VALUE,
    SERIES_LEN,
    START,
    TICKERS,
    DAY_MS,
)

SPECS: dict[str, dict] = {
    "BTC": {"start": 64_200, "mu": 0.00055, "sigma": 0.028, "crypto": True},
    "ETH": {"start": 3_420, "mu": 0.00048, "sigma": 0.032, "crypto": True},
    "SOL": {"start": 148, "mu": 0.0007, "sigma": 0.045, "crypto": True},
    "NVDA": {"start": 118, "mu": 0.0009, "sigma": 0.022, "crypto": False},
    "AAPL": {"start": 212, "mu": 0.00025, "sigma": 0.014, "crypto": False},
}


def _series(ticker: Ticker) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPECS[ticker]
    rand = mulberry32(TICKER_SEEDS[ticker])
    rows: list[dict] = []
    chain: list[dict] = []
    price = float(spec["start"])
    addr = 800_000 + rand() * 200_000
    for i in range(SERIES_LEN):
        t = START + i * DAY_MS
        mu, sigma = spec["mu"], spec["sigma"]
        if 80 < i < 140:
            mu = -spec["mu"] * 1.4
            sigma *= 1.35
        if 250 < i < 310:
            mu *= 2.1
            sigma *= 0.8
        if 365 < i < 378:
            sigma *= 2.4
            mu = -0.012
        ret = mu + sigma * box_muller(rand)
        o = price
        c = max(0.5, price * (1 + ret))
        wiggle = sigma * price * 0.6
        h = max(o, c) + abs(rand() * wiggle)
        low = max(0.4, min(o, c) - abs(rand() * wiggle))
        vol_base = 28_000 if ticker == "BTC" else 12_000
        v = (0.6 + rand() * 0.8) * vol_base * (3.2 if 365 < i < 378 else 1)
        rows.append({"t": t, "o": o, "h": h, "l": low, "c": c, "v": v})
        price = c
        if spec["crypto"]:
            addr = max(100_000, addr * (1 + (rand() - 0.48) * 0.02))
            inflow_mult = 4 if 368 < i < 374 else 1
            chain.append(
                {
                    "t": t,
                    "activeAddresses": int(round(addr)),
                    "exchangeInflow": v * (0.08 + rand() * 0.12) * inflow_mult,
                    "fundingRate": (rand() - 0.5) * 0.0012
                    + (-0.0018 if 365 < i < 378 else 0.0002),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(chain)


_CACHE: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def bars_df(ticker: Ticker) -> pd.DataFrame:
    if ticker not in _CACHE:
        _CACHE[ticker] = _series(ticker)
    return _CACHE[ticker][0]


def chain_df(ticker: Ticker) -> pd.DataFrame:
    if ticker not in _CACHE:
        _CACHE[ticker] = _series(ticker)
    return _CACHE[ticker][1]


def get_bars(ticker: Ticker) -> list[Bar]:
    return [Bar.model_validate(r) for r in bars_df(ticker).to_dict("records")]


def get_chain(ticker: Ticker) -> list[OnChainPoint]:
    df = chain_df(ticker)
    if df.empty:
        return []
    return [OnChainPoint.model_validate(r) for r in df.to_dict("records")]


def _sma(close: np.ndarray, n: int, idx: int) -> float:
    start = max(0, idx - n + 1)
    window = close[start : idx + 1]
    return float(window.mean()) if len(window) else 0.0


def _vol_ann(close: np.ndarray, n: int, idx: int) -> float:
    start = max(1, idx - n + 1)
    rets = np.diff(np.log(close[start - 1 : idx + 1]))
    if rets.size == 0:
        return 0.0
    return float(np.sqrt(rets.var() * 365))


def detect_regime(ticker: Ticker, at_idx: int | None = None) -> Regime:
    close = bars_df(ticker)["c"].to_numpy()
    idx = len(close) - 1 if at_idx is None else min(at_idx, len(close) - 1)
    vol = _vol_ann(close, 20, idx)
    s50 = _sma(close, 50, idx)
    s200 = _sma(close, 200, idx)
    if vol > 0.85:
        return "high-vol"
    if s50 > s200 * 1.03:
        return "bull-trend"
    if s50 < s200 * 0.97:
        return "bear-trend"
    return "range"


def detect_anomalies(ticker: Ticker, lookback: int = 60) -> list[Anomaly]:
    df = bars_df(ticker)
    close = df["c"].to_numpy()
    vol = df["v"].to_numpy()
    ts = df["t"].to_numpy()
    end = len(df) - 1
    start = max(21, end - lookback)
    logret = np.diff(np.log(close))
    window = logret[start - 1 : end]
    mean, sd = window.mean(), window.std(ddof=1) if window.size > 1 else 1.0
    v_win = vol[start : end + 1]
    v_mean, v_sd = v_win.mean(), v_win.std(ddof=1) if v_win.size > 1 else 1.0
    out: list[Anomaly] = []
    for i in range(start, end + 1):
        r = float(np.log(close[i] / close[i - 1]))
        z = (r - mean) / sd if sd else 0.0
        if abs(z) >= 2.6:
            out.append(
                Anomaly(
                    ticker=ticker,
                    t=int(ts[i]),
                    kind="return",
                    z=float(z),
                    note=f"{ticker} {'spike' if z > 0 else 'dump'} {r * 100:.1f}% (z={z:.2f})",
                )
            )
        vz = (vol[i] - v_mean) / v_sd if v_sd else 0.0
        if vz >= 3:
            out.append(
                Anomaly(
                    ticker=ticker,
                    t=int(ts[i]),
                    kind="volume",
                    z=float(vz),
                    note=f"{ticker} volume z={vz:.2f}",
                )
            )
    chain = chain_df(ticker)
    if not chain.empty:
        sl = chain.iloc[-lookback:]
        funds = sl["fundingRate"].to_numpy()
        inflows = sl["exchangeInflow"].to_numpy()
        f_mean, f_sd = funds.mean(), funds.std(ddof=1) if funds.size > 1 else 1.0
        i_mean, i_sd = inflows.mean(), inflows.std(ddof=1) if inflows.size > 1 else 1.0
        for _, p in sl.iterrows():
            fz = (p.fundingRate - f_mean) / f_sd if f_sd else 0.0
            if abs(fz) >= 2.8:
                out.append(
                    Anomaly(
                        ticker=ticker,
                        t=int(p.t),
                        kind="funding",
                        z=float(fz),
                        note=f"{ticker} funding {p.fundingRate * 100:.3f}% (z={fz:.2f})",
                    )
                )
            iz = (p.exchangeInflow - i_mean) / i_sd if i_sd else 0.0
            if iz >= 3:
                out.append(
                    Anomaly(
                        ticker=ticker,
                        t=int(p.t),
                        kind="inflow",
                        z=float(iz),
                        note=f"{ticker} exchange inflow z={iz:.2f} — possible distribution",
                    )
                )
    out.sort(key=lambda a: a.t, reverse=True)
    return out[:8]


def snapshot(ticker: Ticker) -> MarketSnapshot:
    df = bars_df(ticker)
    close = df["c"].to_numpy()
    i = len(df) - 1
    last, prev = float(close[i]), float(close[i - 1])
    ago20 = float(close[i - 20] if i >= 20 else close[0])
    anomalies = detect_anomalies(ticker, 30)
    return MarketSnapshot(
        ticker=ticker,
        price=last,
        change1d=last / prev - 1,
        change20d=last / ago20 - 1,
        vol20d=_vol_ann(close, 20, i),
        sma20=_sma(close, 20, i),
        sma50=_sma(close, 50, i),
        sma200=_sma(close, 200, i),
        regime=detect_regime(ticker),
        lastAnomaly=anomalies[0] if anomalies else None,
    )


def all_snapshots() -> list[MarketSnapshot]:
    return [snapshot(t) for t in TICKERS]


def backtest_sma(ticker: Ticker, fast: int = 10, slow: int = 30) -> BacktestResult:
    df = bars_df(ticker)
    close = df["c"].to_numpy()
    ts = df["t"].to_numpy()
    cash, pos, entry = 1.0, 0, 0.0
    wins = trades = 0
    peak, max_dd = 1.0, 0.0
    equity: list[dict[str, float]] = []
    for i in range(slow, len(close)):
        f, s, px = _sma(close, fast, i), _sma(close, slow, i), float(close[i])
        want = 1 if f > s else 0
        if want != pos:
            if pos == 1:
                ret = px / entry - 1 - 0.001
                cash *= 1 + ret
                wins += int(ret > 0)
                trades += 1
            if want == 1:
                entry = px
            pos = want
        marked = cash * (px / entry) if pos == 1 else cash
        peak = max(peak, marked)
        max_dd = max(max_dd, 1 - marked / peak)
        equity.append({"t": float(ts[i]), "v": marked})
    if pos == 1:
        cash *= float(close[-1]) / entry
        trades += 1
    days = len(close) - slow
    total = cash - 1
    cagr = (1 + total) ** (365 / days) - 1 if days else 0.0
    eq = np.array([p["v"] for p in equity], dtype=float)
    rets = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([0.0])
    m, sd = float(rets.mean()) if rets.size else 0.0, float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    sharpe = (m / sd) * np.sqrt(365) if sd else 0.0
    return BacktestResult(
        ticker=ticker,
        strategy="SMA crossover",
        params={"fast": float(fast), "slow": float(slow)},
        totalReturn=float(total),
        cagr=float(cagr),
        sharpe=float(sharpe),
        maxDrawdown=float(max_dd),
        winRate=(wins / trades) if trades else 0.0,
        trades=trades,
        equity=equity,
    )


def _log_returns(ticker: Ticker) -> np.ndarray:
    c = bars_df(ticker)["c"].to_numpy()
    return np.diff(np.log(c))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def estimate_risk(
    weights: dict[str, float] | None = None,
    value: float = PORTFOLIO_VALUE,
    shock_btc: float = -0.12,
) -> RiskResult:
    w = {**DEFAULT_BOOK, **(weights or {})}
    series = {t: _log_returns(t) for t in TICKERS}
    n = series["BTC"].size
    port = np.zeros(n)
    for t in TICKERS:
        port += w[t] * series[t]
    sorted_p = np.sort(port)
    i95 = max(0, int(0.05 * len(sorted_p)))
    var95 = -float(sorted_p[i95])
    cvar95 = -float(sorted_p[: i95 + 1].mean())
    vol_ann = float(port.std(ddof=1) * np.sqrt(365))
    eq = np.cumprod(1 + port)
    peak = np.maximum.accumulate(eq)
    max_dd = float((1 - eq / peak).max()) if eq.size else 0.0
    mu, sigma = float(port.mean()), float(port.std(ddof=1))
    parametric = abs(mu + sigma * 1.64485)
    rng = np.random.default_rng(7)
    mc = np.percentile(rng.normal(mu, sigma, 20_000), 5)
    monte_carlo = abs(float(mc))

    def beta(to: Ticker, other: Ticker) -> float:
        return 1.0 if to == other else _pearson(series[to], series[other])

    btc_move = sum(w[t] * shock_btc * beta("BTC", t) for t in TICKERS)
    eth_move = sum(w[t] * -0.18 * (1.0 if t == "ETH" else beta("ETH", t) * 0.7) for t in TICKERS)
    return RiskResult(
        portfolioValue=value,
        var95=var95 * value,
        cvar95=cvar95 * value,
        volAnn=vol_ann,
        maxDrawdown=max_dd,
        parametricVar95=parametric * value,
        monteCarloVar95=monte_carlo * value,
        scenarioPnl=[
            ScenarioPnl(name=f"BTC {round(shock_btc * 100)}%", pnl=btc_move * value, pct=btc_move),
            ScenarioPnl(name="ETH −18% risk-off", pnl=eth_move * value, pct=eth_move),
            ScenarioPnl(name="Vol spike (2σ)", pnl=-2 * sigma * value, pct=-2 * sigma),
        ],
        weights=[WeightRow(ticker=t, weight=w[t], value=w[t] * value) for t in TICKERS],
    )


def parse_ticker(query: str) -> Ticker | None:
    u = query.upper()
    aliases = {"BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL"}
    for k, v in aliases.items():
        if k in u:
            return v  # type: ignore[return-value]
    for t in TICKERS:
        if t in u:
            return t  # type: ignore[return-value]
    return None


def timed(name: str, args: dict, fn):
    t0 = time.perf_counter()
    value = fn()
    ms = (time.perf_counter() - t0) * 1000
    return value, ms
