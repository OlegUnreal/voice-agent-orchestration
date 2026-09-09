import { mulberry32 } from "./prng";
import type {
  Anomaly,
  BacktestResult,
  Bar,
  MarketSnapshot,
  OnChainPoint,
  Regime,
  RiskResult,
  Ticker,
} from "./types";
import { TICKERS } from "./types";

const DAY = 86_400_000;
export const SERIES_LEN = 420;
export const AS_OF = Date.UTC(2026, 8, 9);
export const START = AS_OF - (SERIES_LEN - 1) * DAY;

const SPECS: Record<
  Ticker,
  { start: number; mu: number; sigma: number; crypto: boolean }
> = {
  BTC: { start: 64_200, mu: 0.00055, sigma: 0.028, crypto: true },
  ETH: { start: 3_420, mu: 0.00048, sigma: 0.032, crypto: true },
  SOL: { start: 148, mu: 0.0007, sigma: 0.045, crypto: true },
  NVDA: { start: 118, mu: 0.0009, sigma: 0.022, crypto: false },
  AAPL: { start: 212, mu: 0.00025, sigma: 0.014, crypto: false },
};

function seriesFor(ticker: Ticker): { bars: Bar[]; chain: OnChainPoint[] } {
  const spec = SPECS[ticker];
  const rand = mulberry32(hashTicker(ticker));
  const bars: Bar[] = [];
  const chain: OnChainPoint[] = [];
  let price = spec.start;
  let addr = 800_000 + rand() * 200_000;
  for (let i = 0; i < SERIES_LEN; i++) {
    const t = START + i * DAY;
    let mu = spec.mu;
    let sigma = spec.sigma;
    if (i > 80 && i < 140) {
      mu = -spec.mu * 1.4;
      sigma *= 1.35;
    }
    if (i > 250 && i < 310) {
      mu *= 2.1;
      sigma *= 0.8;
    }
    if (i > 365 && i < 378) {
      sigma *= 2.4;
      mu = -0.012;
    }
    const z = boxMuller(rand);
    const ret = mu + sigma * z;
    const o = price;
    const c = Math.max(0.5, price * (1 + ret));
    const wiggle = sigma * price * 0.6;
    const h = Math.max(o, c) + Math.abs(rand() * wiggle);
    const l = Math.max(0.4, Math.min(o, c) - Math.abs(rand() * wiggle));
    const v = (0.6 + rand() * 0.8) * (ticker === "BTC" ? 28_000 : 12_000) * (i > 365 && i < 378 ? 3.2 : 1);
    bars.push({ t, o, h, l, c, v });
    price = c;
    if (spec.crypto) {
      addr = Math.max(100_000, addr * (1 + (rand() - 0.48) * 0.02));
      chain.push({
        t,
        activeAddresses: Math.round(addr),
        exchangeInflow: v * (0.08 + rand() * 0.12) * (i > 368 && i < 374 ? 4 : 1),
        fundingRate: (rand() - 0.5) * 0.0012 + (i > 365 && i < 378 ? -0.0018 : 0.0002),
      });
    }
  }
  return { bars, chain };
}

function hashTicker(t: Ticker) {
  return { BTC: 11, ETH: 23, SOL: 41, NVDA: 59, AAPL: 73 }[t];
}

function boxMuller(rand: () => number) {
  const u = Math.max(1e-9, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const CACHE: Record<Ticker, { bars: Bar[]; chain: OnChainPoint[] }> = {
  BTC: seriesFor("BTC"),
  ETH: seriesFor("ETH"),
  SOL: seriesFor("SOL"),
  NVDA: seriesFor("NVDA"),
  AAPL: seriesFor("AAPL"),
};

export function getBars(ticker: Ticker): Bar[] {
  return CACHE[ticker].bars;
}

export function getChain(ticker: Ticker): OnChainPoint[] {
  return CACHE[ticker].chain;
}

function sma(bars: Bar[], n: number, idx: number) {
  const from = Math.max(0, idx - n + 1);
  let s = 0;
  for (let i = from; i <= idx; i++) s += bars[i]!.c;
  return s / (idx - from + 1);
}

function vol(bars: Bar[], n: number, idx: number) {
  const from = Math.max(1, idx - n + 1);
  const rets: number[] = [];
  for (let i = from; i <= idx; i++) {
    rets.push(Math.log(bars[i]!.c / bars[i - 1]!.c));
  }
  if (!rets.length) return 0;
  const m = rets.reduce((a, b) => a + b, 0) / rets.length;
  const v = rets.reduce((a, b) => a + (b - m) ** 2, 0) / rets.length;
  return Math.sqrt(v * 365);
}

export function detectRegime(ticker: Ticker, atIdx = SERIES_LEN - 1): Regime {
  const bars = getBars(ticker);
  const idx = Math.min(atIdx, bars.length - 1);
  const v = vol(bars, 20, idx);
  const s50 = sma(bars, 50, idx);
  const s200 = sma(bars, 200, idx);
  if (v > 0.85) return "high-vol";
  if (s50 > s200 * 1.03) return "bull-trend";
  if (s50 < s200 * 0.97) return "bear-trend";
  return "range";
}

export function detectAnomalies(ticker: Ticker, lookback = 60): Anomaly[] {
  const bars = getBars(ticker);
  const chain = getChain(ticker);
  const out: Anomaly[] = [];
  const end = bars.length - 1;
  const start = Math.max(21, end - lookback);
  const rets: number[] = [];
  const vols: number[] = [];
  for (let i = start; i <= end; i++) {
    rets.push(Math.log(bars[i]!.c / bars[i - 1]!.c));
    vols.push(bars[i]!.v);
  }
  const mean = avg(rets);
  const sd = std(rets, mean);
  const vMean = avg(vols);
  const vSd = std(vols, vMean);
  for (let i = start; i <= end; i++) {
    const r = Math.log(bars[i]!.c / bars[i - 1]!.c);
    const z = sd ? (r - mean) / sd : 0;
    if (Math.abs(z) >= 2.6) {
      out.push({
        ticker,
        t: bars[i]!.t,
        kind: "return",
        z,
        note: `${ticker} ${z > 0 ? "spike" : "dump"} ${(r * 100).toFixed(1)}% (z=${z.toFixed(2)})`,
      });
    }
    const vz = vSd ? (bars[i]!.v - vMean) / vSd : 0;
    if (vz >= 3) {
      out.push({
        ticker,
        t: bars[i]!.t,
        kind: "volume",
        z: vz,
        note: `${ticker} volume z=${vz.toFixed(2)}`,
      });
    }
  }
  if (chain.length) {
    const funds = chain.slice(-lookback).map((p) => p.fundingRate);
    const fMean = avg(funds);
    const fSd = std(funds, fMean);
    const inflows = chain.slice(-lookback).map((p) => p.exchangeInflow);
    const iMean = avg(inflows);
    const iSd = std(inflows, iMean);
    for (const p of chain.slice(-lookback)) {
      const fz = fSd ? (p.fundingRate - fMean) / fSd : 0;
      if (Math.abs(fz) >= 2.8) {
        out.push({
          ticker,
          t: p.t,
          kind: "funding",
          z: fz,
          note: `${ticker} funding ${(p.fundingRate * 100).toFixed(3)}% (z=${fz.toFixed(2)})`,
        });
      }
      const iz = iSd ? (p.exchangeInflow - iMean) / iSd : 0;
      if (iz >= 3) {
        out.push({
          ticker,
          t: p.t,
          kind: "inflow",
          z: iz,
          note: `${ticker} exchange inflow z=${iz.toFixed(2)} — possible distribution`,
        });
      }
    }
  }
  return out.sort((a, b) => b.t - a.t).slice(0, 8);
}

export function snapshot(ticker: Ticker): MarketSnapshot {
  const bars = getBars(ticker);
  const i = bars.length - 1;
  const last = bars[i]!;
  const prev = bars[i - 1]!;
  const ago20 = bars[i - 20] ?? bars[0]!;
  const anomalies = detectAnomalies(ticker, 30);
  return {
    ticker,
    price: last.c,
    change1d: last.c / prev.c - 1,
    change20d: last.c / ago20.c - 1,
    vol20d: vol(bars, 20, i),
    sma20: sma(bars, 20, i),
    sma50: sma(bars, 50, i),
    sma200: sma(bars, 200, i),
    regime: detectRegime(ticker),
    lastAnomaly: anomalies[0],
  };
}

export function allSnapshots(): MarketSnapshot[] {
  return TICKERS.map(snapshot);
}

export function backtestSma(
  ticker: Ticker,
  fast = 10,
  slow = 30,
): BacktestResult {
  const bars = getBars(ticker);
  const equity: { t: number; v: number }[] = [];
  let cash = 1;
  let pos = 0;
  let entry = 0;
  let wins = 0;
  let trades = 0;
  let peak = 1;
  let maxDd = 0;
  for (let i = slow; i < bars.length; i++) {
    const f = sma(bars, fast, i);
    const s = sma(bars, slow, i);
    const px = bars[i]!.c;
    const want = f > s ? 1 : 0;
    if (want !== pos) {
      if (pos === 1) {
        const ret = px / entry - 1 - 0.001;
        cash *= 1 + ret;
        if (ret > 0) wins++;
        trades++;
      }
      if (want === 1) entry = px;
      pos = want;
    }
    const marked = pos === 1 ? cash * (px / entry) : cash;
    peak = Math.max(peak, marked);
    maxDd = Math.max(maxDd, 1 - marked / peak);
    equity.push({ t: bars[i]!.t, v: marked });
  }
  if (pos === 1) {
    const px = bars[bars.length - 1]!.c;
    cash *= px / entry;
    trades++;
  }
  const days = bars.length - slow;
  const totalReturn = cash - 1;
  const cagr = (1 + totalReturn) ** (365 / days) - 1;
  const rets: number[] = [];
  for (let i = 1; i < equity.length; i++) {
    rets.push(equity[i]!.v / equity[i - 1]!.v - 1);
  }
  const m = avg(rets);
  const sd = std(rets, m);
  const sharpe = sd ? (m / sd) * Math.sqrt(365) : 0;
  return {
    ticker,
    strategy: "SMA crossover",
    params: { fast, slow },
    totalReturn,
    cagr,
    sharpe,
    maxDrawdown: maxDd,
    winRate: trades ? wins / trades : 0,
    trades,
    equity,
  };
}

const DEFAULT_WEIGHTS: Record<Ticker, number> = {
  BTC: 0.4,
  ETH: 0.25,
  SOL: 0.15,
  NVDA: 0.1,
  AAPL: 0.1,
};

export function estimateRisk(
  weights: Partial<Record<Ticker, number>> = DEFAULT_WEIGHTS,
  value = 1_000_000,
  shockBtc = -0.12,
): RiskResult {
  const w = { ...DEFAULT_WEIGHTS, ...weights };
  const series = TICKERS.map((t) => {
    const bars = getBars(t);
    return bars.slice(1).map((b, i) => Math.log(b.c / bars[i]!.c));
  });
  const n = series[0]!.length;
  const port: number[] = [];
  for (let i = 0; i < n; i++) {
    let r = 0;
    TICKERS.forEach((t, k) => {
      r += w[t]! * series[k]![i]!;
    });
    port.push(r);
  }
  const sorted = [...port].sort((a, b) => a - b);
  const i95 = Math.floor(0.05 * sorted.length);
  const var95 = -sorted[i95]!;
  const cvar95 = -avg(sorted.slice(0, i95 + 1));
  const m = avg(port);
  const volAnn = std(port, m) * Math.sqrt(365);
  let eq = 1;
  let peak = 1;
  let maxDd = 0;
  for (const r of port) {
    eq *= 1 + r;
    peak = Math.max(peak, eq);
    maxDd = Math.max(maxDd, 1 - eq / peak);
  }
  const corr = (a: Ticker, b: Ticker) => {
    const ia = TICKERS.indexOf(a);
    const ib = TICKERS.indexOf(b);
    return pearson(series[ia]!, series[ib]!);
  };
  const btcShock = shockBtc;
  const scenarioMove = TICKERS.reduce((s, t) => {
    const beta = t === "BTC" ? 1 : corr("BTC", t);
    return s + w[t]! * btcShock * beta;
  }, 0);
  const ethDump = TICKERS.reduce((s, t) => {
    const beta = t === "ETH" ? 1 : corr("ETH", t) * 0.7;
    return s + w[t]! * -0.18 * beta;
  }, 0);
  return {
    portfolioValue: value,
    var95: var95 * value,
    cvar95: cvar95 * value,
    volAnn,
    maxDrawdown: maxDd,
    scenarioPnl: [
      { name: `BTC ${Math.round(shockBtc * 100)}%`, pnl: scenarioMove * value, pct: scenarioMove },
      { name: "ETH −18% risk-off", pnl: ethDump * value, pct: ethDump },
      { name: "Vol spike (2σ)", pnl: -2 * std(port, m) * value, pct: -2 * std(port, m) },
    ],
    weights: TICKERS.map((t) => ({
      ticker: t,
      weight: w[t]!,
      value: w[t]! * value,
    })),
  };
}

function avg(xs: number[]) {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}
function std(xs: number[], mean: number) {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1));
}
function pearson(a: number[], b: number[]) {
  const ma = avg(a);
  const mb = avg(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]! - ma;
    const y = b[i]! - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

export function parseTicker(q: string): Ticker | undefined {
  const u = q.toUpperCase();
  return TICKERS.find((t) => u.includes(t) || (t === "BTC" && /BITCOIN/.test(u)) || (t === "ETH" && /ETHEREUM/.test(u)) || (t === "SOL" && /SOLANA/.test(u)));
}
