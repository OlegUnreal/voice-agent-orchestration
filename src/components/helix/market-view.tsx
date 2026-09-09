import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import {
  allSnapshots,
  backtestSma,
  detectAnomalies,
  estimateRisk,
  getBars,
} from "@/lib/helix/market";
import type { Ticker } from "@/lib/helix/types";
import { cn, formatNum, formatPct, formatUsd } from "@/lib/utils";
import { useMemo, useState } from "react";

export function MarketView() {
  const [ticker, setTicker] = useState<Ticker>("BTC");
  const snaps = useMemo(() => allSnapshots(), []);
  const bars = getBars(ticker);
  const chart = bars.slice(-120).map((b) => ({
    t: new Date(b.t).toISOString().slice(5, 10),
    c: Number(b.c.toFixed(2)),
    v: b.v,
  }));
  const snap = snaps.find((s) => s.ticker === ticker)!;
  const anomalies = detectAnomalies(ticker);
  const bt = backtestSma(ticker);
  const risk = estimateRisk();
  const eq = bt.equity.filter((_, i) => i % 3 === 0).map((p) => ({
    t: new Date(p.t).toISOString().slice(5, 10),
    v: Number((p.v * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-5">
      <header>
        <h1 className="font-display text-3xl tracking-tight">Market intelligence</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Deterministic prices, regime labels, z-score anomalies and backtests. Numbers never come from the language model.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {snaps.map((s) => (
          <button
            key={s.ticker}
            type="button"
            onClick={() => setTicker(s.ticker)}
            className={cn(
              "min-w-[108px] rounded-2xl border px-3 py-3 text-left",
              s.ticker === ticker ? "border-fg/30 bg-elevated" : "border-border bg-surface",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-muted">{s.ticker}</span>
              <Badge tone={s.change1d >= 0 ? "gain" : "loss"}>{formatPct(s.change1d)}</Badge>
            </div>
            <p className="mt-1 font-mono text-sm tabular-nums">{formatNum(s.price, 2)}</p>
            <p className="mt-1 text-[11px] text-subtle">{s.regime}</p>
          </button>
        ))}
      </div>

      <section className="rounded-3xl border border-border bg-surface p-4 sm:p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-xl">{ticker}</h2>
            <p className="text-sm text-muted">
              20d vol {(snap.vol20d * 100).toFixed(0)}% · SMA50 {formatNum(snap.sma50, 1)} · SMA200{" "}
              {formatNum(snap.sma200, 1)}
            </p>
          </div>
          <Badge>{snap.regime}</Badge>
        </div>
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chart}>
              <defs>
                <linearGradient id="px" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#d4d4d8" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="#d4d4d8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#27272a" vertical={false} />
              <XAxis dataKey="t" tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={24} />
              <YAxis
                tick={{ fill: "#71717a", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                width={56}
                domain={["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{ background: "#121214", border: "1px solid #27272a", borderRadius: 12 }}
                labelStyle={{ color: "#a1a1aa" }}
              />
              <Area type="monotone" dataKey="c" stroke="#d4d4d8" fill="url(#px)" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-3xl border border-border bg-surface p-5">
          <h2 className="font-display text-lg">Anomalies</h2>
          <ul className="mt-3 space-y-3">
            {anomalies.length === 0 && <li className="text-sm text-muted">None above threshold.</li>}
            {anomalies.slice(0, 5).map((a, i) => (
              <li key={i} className="flex items-start justify-between gap-3 text-sm">
                <div>
                  <p>{a.note}</p>
                  <p className="text-xs text-subtle">{new Date(a.t).toISOString().slice(0, 10)}</p>
                </div>
                <span className="font-mono text-xs tabular-nums text-muted">z {a.z.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </section>
        <section className="rounded-3xl border border-border bg-surface p-5">
          <h2 className="font-display text-lg">SMA 10/30 backtest</h2>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <Stat k="Return" v={formatPct(bt.totalReturn)} pos={bt.totalReturn >= 0} />
            <Stat k="Sharpe" v={bt.sharpe.toFixed(2)} />
            <Stat k="Max DD" v={formatPct(-bt.maxDrawdown)} pos={false} />
            <Stat k="Win rate" v={formatPct(bt.winRate)} />
          </dl>
          <div className="mt-4 h-28">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={eq}>
                <XAxis dataKey="t" hide />
                <YAxis hide domain={["auto", "auto"]} />
                <Area type="monotone" dataKey="v" stroke="#6d8f78" fill="#6d8f7822" strokeWidth={1.4} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="rounded-3xl border border-border bg-surface p-5">
        <h2 className="font-display text-lg">Book risk</h2>
        <p className="mt-1 text-sm text-muted">Default 40/25/15/10/10 · $1M notional · empirical betas</p>
        <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat k="95% VaR" v={formatUsd(risk.var95)} pos={false} />
          <Stat k="CVaR" v={formatUsd(risk.cvar95)} pos={false} />
          <Stat k="Ann. vol" v={formatPct(risk.volAnn)} />
          <Stat k="Max DD" v={formatPct(-risk.maxDrawdown)} pos={false} />
        </dl>
        <ul className="mt-4 space-y-2">
          {risk.scenarioPnl.map((s) => (
            <li key={s.name} className="flex justify-between text-sm">
              <span className="text-muted">{s.name}</span>
              <span className={s.pnl >= 0 ? "text-gain" : "text-loss"}>
                {formatUsd(s.pnl)} ({formatPct(s.pct)})
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function Stat({ k, v, pos }: { k: string; v: string; pos?: boolean }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wider text-subtle">{k}</dt>
      <dd
        className={cn(
          "mt-1 font-mono text-sm tabular-nums",
          pos === true && "text-gain",
          pos === false && "text-loss",
        )}
      >
        {v}
      </dd>
    </div>
  );
}

