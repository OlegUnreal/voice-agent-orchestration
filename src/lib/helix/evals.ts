import { AS_OF } from "./market";
import { lexicalRetrieve, retrieve } from "./rag";
import { BASELINE_WEIGHTS, type RerankWeights } from "./reranker";
import { classifyIntent } from "./router";
import type { EvalCase, EvalMetrics, Intent } from "./types";

const D = 86_400_000;

export const GOLDEN: EvalCase[] = [
  {
    id: "g1",
    query: "What is BTC's current regime?",
    asOf: AS_OF,
    relevant: ["R-BTC-REGIME"],
    expectedIntent: "regime",
    expectedTools: ["detect_regime", "get_market_snapshot"],
  },
  {
    id: "g2",
    query: "Any BTC exchange inflow anomalies?",
    asOf: AS_OF,
    relevant: ["R-BTC-ONCHAIN", "R-ANOM-WINDOW"],
    expectedIntent: "anomaly",
    expectedTools: ["detect_anomalies"],
  },
  {
    id: "g3",
    query: "ETH staking flows and liquid staking",
    asOf: AS_OF,
    relevant: ["R-ETH-STAKING"],
    expectedIntent: "research",
    expectedTools: ["retrieve_evidence"],
  },
  {
    id: "g4",
    query: "Backtest SMA crossover on SOL",
    asOf: AS_OF,
    relevant: ["R-SOL-MOM", "R-STRAT-SMA"],
    expectedIntent: "backtest",
    expectedTools: ["run_backtest"],
  },
  {
    id: "g5",
    query: "Portfolio risk if BTC drops 12 percent",
    asOf: AS_OF,
    relevant: ["R-PORT-VAR", "R-PORT-REVIEW"],
    expectedIntent: "risk",
    expectedTools: ["estimate_risk"],
  },
  {
    id: "g6",
    query: "Cite evidence for the SOL momentum call",
    asOf: AS_OF,
    relevant: ["R-SOL-MOM", "R-SOL-ONCHAIN"],
    expectedIntent: "research",
    expectedTools: ["retrieve_evidence"],
  },
  {
    id: "g7",
    query: "How do promotion gates work for rerankers?",
    asOf: AS_OF,
    relevant: ["R-PROMOTE", "R-GOLDEN"],
    expectedIntent: "eval",
    expectedTools: ["get_eval_report"],
  },
  {
    id: "g8",
    query: "NVDA implied vol after earnings",
    asOf: AS_OF,
    relevant: ["R-NVDA-EARN"],
    expectedIntent: "research",
    expectedTools: ["retrieve_evidence"],
  },
  {
    id: "g9",
    query: "Is AAPL useful ballast in a crypto book?",
    asOf: AS_OF,
    relevant: ["R-AAPL-DEF", "R-PORT-VAR"],
    expectedIntent: "portfolio",
    expectedTools: ["estimate_risk", "retrieve_evidence"],
  },
  {
    id: "g10",
    query: "What happened in the late-sample stress window?",
    asOf: AS_OF,
    relevant: ["R-ANOM-WINDOW", "R-BTC-STRAT", "R-BTC-ONCHAIN"],
    expectedIntent: "anomaly",
    expectedTools: ["detect_anomalies", "retrieve_evidence"],
  },
  {
    id: "g11",
    query: "ETH regime — trend or high vol?",
    asOf: AS_OF,
    relevant: ["R-ETH-REGIME"],
    expectedIntent: "regime",
    expectedTools: ["detect_regime"],
  },
  {
    id: "g12",
    query: "Explain the groundedness contract for spoken answers",
    asOf: AS_OF,
    relevant: ["R-CITATION"],
    expectedIntent: "research",
    expectedTools: ["retrieve_evidence"],
  },
  {
    id: "g13",
    query: "Which MCP tools need approval?",
    asOf: AS_OF,
    relevant: ["R-MCP"],
    expectedIntent: "general",
    expectedTools: ["retrieve_evidence"],
  },
  {
    id: "g14",
    query: "SMA playbook and trading costs",
    asOf: AS_OF,
    relevant: ["R-STRAT-SMA", "R-BTC-STRAT"],
    expectedIntent: "backtest",
    expectedTools: ["run_backtest", "retrieve_evidence"],
  },
  {
    id: "g15",
    query: "Review the default 40 25 15 10 10 book",
    asOf: AS_OF,
    relevant: ["R-PORT-VAR", "R-PORT-REVIEW"],
    expectedIntent: "portfolio",
    expectedTools: ["estimate_risk"],
  },
  {
    id: "g16",
    query: "Retrieval policy recency and no leakage",
    asOf: AS_OF - D,
    relevant: ["R-RAG-POLICY"],
    expectedIntent: "research",
    expectedTools: ["retrieve_evidence"],
  },
];

const INTENT_TOOLS: Record<Intent, string[]> = {
  regime: ["detect_regime", "get_market_snapshot"],
  anomaly: ["detect_anomalies"],
  backtest: ["run_backtest"],
  risk: ["estimate_risk"],
  research: ["retrieve_evidence"],
  portfolio: ["estimate_risk", "retrieve_evidence"],
  eval: ["get_eval_report"],
  training: ["get_checkpoint_status"],
  general: ["retrieve_evidence"],
};

function dcg(rels: number[]) {
  return rels.reduce((s, r, i) => s + ((2 ** r - 1) / Math.log2(i + 2)), 0);
}

export function retrievalMetrics(
  rankedIds: string[],
  relevant: string[],
): { recall5: number; recall10: number; mrr: number; ndcg10: number } {
  const rel = new Set(relevant);
  const hits5 = rankedIds.slice(0, 5).filter((id) => rel.has(id)).length;
  const hits10 = rankedIds.slice(0, 10).filter((id) => rel.has(id)).length;
  let mrr = 0;
  for (let i = 0; i < rankedIds.length; i++) {
    if (rel.has(rankedIds[i]!)) {
      mrr = 1 / (i + 1);
      break;
    }
  }
  const gains = rankedIds.slice(0, 10).map((id) => (rel.has(id) ? 1 : 0));
  const ideal = [...gains].sort((a, b) => b - a);
  const idcg = dcg(relevant.slice(0, 10).map(() => 1));
  const ndcg10 = idcg ? dcg(gains) / Math.max(idcg, dcg(ideal)) : 0;
  return {
    recall5: relevant.length ? hits5 / relevant.length : 0,
    recall10: relevant.length ? hits10 / relevant.length : 0,
    mrr,
    ndcg10,
  };
}

export type RetrieverKind = "lexical" | "embed" | "lora";

export function runEvalSuite(kind: RetrieverKind, weights?: RerankWeights): EvalMetrics {
  const t0 = performance.now();
  const rows = GOLDEN.map((c) => {
    const t1 = performance.now();
    const ranked =
      kind === "lexical"
        ? lexicalRetrieve(c.query, 10, c.asOf)
        : retrieve(c.query, {
            k: 10,
            asOf: c.asOf,
            weights: kind === "embed" ? undefined : weights,
          });
    const ids = ranked.map((r) => r.doc.id);
    const ir = retrievalMetrics(ids, c.relevant);
    const intent = classifyIntent(c.query);
    const tools = INTENT_TOOLS[intent];
    const toolHit =
      c.expectedTools.filter((t) => tools.includes(t)).length / c.expectedTools.length;
    const citeOk = ids.slice(0, 5).some((id) => c.relevant.includes(id)) ? 1 : 0;
    const needsCite = c.expectedTools.includes("retrieve_evidence");
    const grounded = needsCite ? citeOk : 1;
    const structured = intent === c.expectedIntent ? 1 : 0;
    return {
      ir,
      intentOk: intent === c.expectedIntent ? 1 : 0,
      toolHit,
      citeOk,
      grounded,
      structured,
      ms: performance.now() - t1,
    };
  });
  const n = rows.length;
  const pick = (fn: (r: (typeof rows)[0]) => number) => rows.reduce((s, r) => s + fn(r), 0) / n;
  const times = rows.map((r) => r.ms).sort((a, b) => a - b);
  const p50 = times[Math.floor(n * 0.5)] ?? 0;
  const p95 = times[Math.floor(n * 0.95)] ?? times[n - 1] ?? 0;
  void t0;
  return {
    recallAt5: pick((r) => r.ir.recall5),
    recallAt10: pick((r) => r.ir.recall10),
    mrr: pick((r) => r.ir.mrr),
    ndcgAt10: pick((r) => r.ir.ndcg10),
    intentAcc: pick((r) => r.intentOk),
    toolAcc: pick((r) => r.toolHit),
    groundedness: pick((r) => r.grounded),
    citationCorrect: pick((r) => r.citeOk),
    structuredOk: pick((r) => r.structured),
    p50Ms: p50,
    p95Ms: p95,
    costUsd: kind === "lexical" ? 0 : 0.0012,
  };
}

export function lexicalEval() {
  return runEvalSuite("lexical", BASELINE_WEIGHTS);
}
