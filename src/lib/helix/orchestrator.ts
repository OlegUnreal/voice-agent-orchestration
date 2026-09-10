import {
  allSnapshots,
  backtestSma,
  detectAnomalies,
  detectRegime,
  estimateRisk,
  parseTicker,
  snapshot,
} from "./market";
import { retrieve } from "./rag";
import { classifyIntent } from "./router";
import { defaultCheckpoints, gateCheck } from "./training";
import { runEvalSuite } from "./evals";
import type {
  AgentHop,
  AgentId,
  Intent,
  OrchestratorResult,
  Ticker,
  ToolCall,
} from "./types";

function timed<T>(name: string, args: Record<string, string | number | boolean>, fn: () => T): { value: T; call: ToolCall } {
  const t0 = performance.now();
  const value = fn();
  const ms = performance.now() - t0;
  return {
    value,
    call: {
      name,
      args,
      ms,
      ok: true,
      summary: summarize(name, value),
    },
  };
}

function summarize(name: string, value: unknown): string {
  if (name === "get_market_snapshot") {
    const s = value as ReturnType<typeof snapshot>;
    return `${s.ticker} ${s.price.toFixed(2)} regime=${s.regime}`;
  }
  if (name === "detect_regime") return String(value);
  if (name === "detect_anomalies") {
    const a = value as ReturnType<typeof detectAnomalies>;
    return `${a.length} events`;
  }
  if (name === "run_backtest") {
    const b = value as ReturnType<typeof backtestSma>;
    return `Sharpe ${b.sharpe.toFixed(2)} DD ${(b.maxDrawdown * 100).toFixed(1)}%`;
  }
  if (name === "estimate_risk") {
    const r = value as ReturnType<typeof estimateRisk>;
    return `VaR95 ${r.var95.toFixed(0)} vol ${(r.volAnn * 100).toFixed(1)}%`;
  }
  if (name === "retrieve_evidence") {
    const d = value as ReturnType<typeof retrieve>;
    return d.slice(0, 3).map((x) => x.doc.id).join(", ");
  }
  return "ok";
}

const INTENT_AGENT: Record<Intent, AgentId> = {
  regime: "quant",
  anomaly: "quant",
  backtest: "quant",
  risk: "copilot",
  research: "research",
  portfolio: "copilot",
  eval: "eval",
  training: "training",
  memory: "copilot",
  general: "supervisor",
};

export function runOrchestrator(query: string): OrchestratorResult {
  const tStart = performance.now();
  const hops: AgentHop[] = [];
  const tools: ToolCall[] = [];
  const intent = classifyIntent(query);
  hops.push({
    agent: "supervisor",
    label: `route → ${intent}`,
    ms: performance.now() - tStart,
  });
  const specialist = INTENT_AGENT[intent];
  const ticker: Ticker = parseTicker(query) ?? "BTC";
  const numbers: Record<string, number | string> = { intent, ticker };
  const tSpec = performance.now();

  let snapshotRes: ReturnType<typeof snapshot> | undefined;
  let anomalies: ReturnType<typeof detectAnomalies> | undefined;
  let backtest: ReturnType<typeof backtestSma> | undefined;
  let risk: ReturnType<typeof estimateRisk> | undefined;
  let retrieved: ReturnType<typeof retrieve> | undefined;

  const needSnapshot = ["regime", "anomaly", "backtest", "portfolio", "general"].includes(intent);
  const needRegime = intent === "regime" || intent === "portfolio";
  const needAnom = intent === "anomaly" || intent === "portfolio";
  const needBt = intent === "backtest";
  const needRisk = intent === "risk" || intent === "portfolio";
  const needRag = ["research", "general", "anomaly", "regime", "portfolio", "eval", "training"].includes(intent);
  const needEval = intent === "eval";
  const needCkpt = intent === "training" || intent === "eval";

  if (needSnapshot) {
    const r = timed("get_market_snapshot", { ticker }, () => snapshot(ticker));
    tools.push(r.call);
    snapshotRes = r.value;
    numbers.price = r.value.price;
    numbers.change1d = r.value.change1d;
    numbers.regime = r.value.regime;
    numbers.vol20d = r.value.vol20d;
  }
  if (needRegime) {
    const r = timed("detect_regime", { ticker }, () => detectRegime(ticker));
    tools.push(r.call);
    numbers.regime = r.value;
  }
  if (needAnom) {
    const r = timed("detect_anomalies", { ticker, lookback: 60 }, () => detectAnomalies(ticker));
    tools.push(r.call);
    anomalies = r.value;
    numbers.anomalyCount = r.value.length;
  }
  if (needBt) {
    const r = timed("run_backtest", { ticker, fast: 10, slow: 30 }, () => backtestSma(ticker));
    tools.push(r.call);
    backtest = r.value;
    numbers.sharpe = r.value.sharpe;
    numbers.totalReturn = r.value.totalReturn;
    numbers.maxDrawdown = r.value.maxDrawdown;
    numbers.trades = r.value.trades;
  }
  if (needRisk) {
    const shock = /12/.test(query) ? -0.12 : -0.12;
    const r = timed("estimate_risk", { shockBtc: shock }, () => estimateRisk(undefined, 1_000_000, shock));
    tools.push(r.call);
    risk = r.value;
    numbers.var95 = r.value.var95;
    numbers.volAnn = r.value.volAnn;
    numbers.scenario0 = r.value.scenarioPnl[0]!.pct;
  }
  if (needRag) {
    const r = timed("retrieve_evidence", { query, k: 6 }, () => retrieve(query, { k: 6 }));
    tools.push(r.call);
    retrieved = r.value;
  }
  if (needEval) {
    const r = timed("get_eval_report", {}, () => runEvalSuite("lora"));
    tools.push(r.call);
    const m = r.value;
    numbers.ndcgAt10 = m.ndcgAt10;
    numbers.recallAt5 = m.recallAt5;
    numbers.mrr = m.mrr;
    numbers.groundedness = m.groundedness;
  }
  if (needCkpt) {
    const r = timed("get_checkpoint_status", {}, () => defaultCheckpoints());
    tools.push(r.call);
    const prod = r.value.find((c) => c.stage === "production");
    if (prod) numbers.production = prod.name;
  }

  hops.push({
    agent: specialist,
    label: tools.map((t) => t.name).join(" · ") || specialist,
    ms: performance.now() - tSpec,
  });

  const citations = (retrieved ?? []).slice(0, 4).map((d) => ({
    id: d.doc.id,
    title: d.doc.title,
    type: d.doc.type,
    ts: d.doc.ts,
  }));

  const fallbackSpoken = speak(intent, ticker, {
    snapshotRes,
    anomalies,
    backtest,
    risk,
    retrieved,
    numbers,
  });

  return {
    intent,
    hops,
    tools,
    citations,
    snapshot: snapshotRes,
    anomalies,
    backtest,
    risk,
    retrieved,
    numbers,
    fallbackSpoken,
    grounded: citations.length > 0 || Boolean(snapshotRes || backtest || risk),
  };
}

export function bookOverview() {
  return {
    snapshots: allSnapshots(),
    risk: estimateRisk(),
  };
}

function speak(
  intent: Intent,
  ticker: Ticker,
  ctx: {
    snapshotRes?: ReturnType<typeof snapshot>;
    anomalies?: ReturnType<typeof detectAnomalies>;
    backtest?: ReturnType<typeof backtestSma>;
    risk?: ReturnType<typeof estimateRisk>;
    retrieved?: ReturnType<typeof retrieve>;
    numbers: Record<string, number | string>;
  },
): string {
  const s = ctx.snapshotRes;
  if (intent === "regime" && s) {
    return `${ticker} is in a ${s.regime.replace("-", " ")} regime. Last ${s.price.toFixed(2)}, 1-day ${(s.change1d * 100).toFixed(1)} percent, 20-day vol ${(s.vol20d * 100).toFixed(0)} percent annualized. 50-day SMA ${s.sma50.toFixed(2)} versus 200-day ${s.sma200.toFixed(2)}.`;
  }
  if (intent === "anomaly") {
    const a = ctx.anomalies?.[0];
    if (!a) return `No z-score events above threshold on ${ticker} in the lookback.`;
    return `Latest ${ticker} anomaly: ${a.note}. ${ctx.anomalies!.length} events in the window. Evidence is grounded in the return, volume and on-chain z-scores, not a language-model guess.`;
  }
  if (intent === "backtest" && ctx.backtest) {
    const b = ctx.backtest;
    return `${ticker} 10/30 SMA crossover: total return ${(b.totalReturn * 100).toFixed(1)} percent, Sharpe ${b.sharpe.toFixed(2)}, max drawdown ${(b.maxDrawdown * 100).toFixed(1)} percent over ${b.trades} trades after 10 bps costs.`;
  }
  if ((intent === "risk" || intent === "portfolio") && ctx.risk) {
    const r = ctx.risk;
    const sc = r.scenarioPnl[0]!;
    return `On a $${(r.portfolioValue / 1e6).toFixed(1)}M book, 1-day 95 percent VaR is $${(r.var95 / 1000).toFixed(0)}k. Annualized vol ${(r.volAnn * 100).toFixed(1)} percent. A 12 percent BTC shock maps to ${(sc.pct * 100).toFixed(1)} percent P and L via empirical betas, not a flat 12.`;
  }
  if (intent === "eval") {
    return `EvalForge production reranker: nDCG@10 ${(Number(ctx.numbers.ndcgAt10) * 100).toFixed(0)} percent, Recall@5 ${(Number(ctx.numbers.recallAt5) * 100).toFixed(0)} percent. Promotion still requires groundedness, p95 latency and cost gates.`;
  }
  if (intent === "training") {
    return `Registry production checkpoint is ${ctx.numbers.production ?? "rerank-lora-r8"}. LoRA rank-8 is a logistic probe on retrieval features — the same promotion loop you would use for a real adapter.`;
  }
  const cite = ctx.retrieved?.[0];
  if (cite) {
    return `${cite.doc.title}. ${cite.doc.body.slice(0, 220)}`;
  }
  if (s) {
    return `${ticker} last ${s.price.toFixed(2)}, regime ${s.regime}. Ask for a regime read, anomalies, a backtest, or portfolio risk.`;
  }
  return "Helix is the voice layer over market tools, RAG, evals and the model registry. Ask a market question.";
}

export function grokToolPayload(result: OrchestratorResult) {
  return {
    intent: result.intent,
    tools: result.tools.map((t) => ({ name: t.name, summary: t.summary, args: t.args })),
    numbers: result.numbers,
    citations: result.citations,
    evidence: (result.retrieved ?? []).slice(0, 4).map((r) => ({
      id: r.doc.id,
      title: r.doc.title,
      body: r.doc.body,
      score: Number(r.score.toFixed(3)),
    })),
    snapshot: result.snapshot,
    anomalies: result.anomalies?.slice(0, 3),
    backtest: result.backtest
      ? {
          ticker: result.backtest.ticker,
          totalReturn: result.backtest.totalReturn,
          sharpe: result.backtest.sharpe,
          maxDrawdown: result.backtest.maxDrawdown,
          trades: result.backtest.trades,
        }
      : undefined,
    risk: result.risk
      ? {
          var95: result.risk.var95,
          cvar95: result.risk.cvar95,
          volAnn: result.risk.volAnn,
          scenarios: result.risk.scenarioPnl,
        }
      : undefined,
    gates: gateCheck(runEvalSuite("lora")),
  };
}
