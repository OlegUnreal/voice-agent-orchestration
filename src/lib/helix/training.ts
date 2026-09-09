import { GOLDEN, runEvalSuite, type RetrieverKind } from "./evals";
import { retrieve } from "./rag";
import {
  BASELINE_WEIGHTS,
  EMBED_WEIGHTS,
  trainLogistic,
  type RerankWeights,
  type TrainPoint,
} from "./reranker";
import type { Checkpoint, EvalMetrics, PromotionStage } from "./types";

const GATES = {
  ndcgAt10: 0.72,
  recallAt5: 0.8,
  groundedness: 0.85,
  p95Ms: 800,
  costUsd: 0.02,
};

function buildPairs(): TrainPoint[] {
  const data: TrainPoint[] = [];
  for (const c of GOLDEN) {
    const ranked = retrieve(c.query, { k: 12, asOf: c.asOf, weights: EMBED_WEIGHTS });
    const rel = new Set(c.relevant);
    for (const r of ranked) {
      data.push({
        features: {
          cosine: r.cosine,
          recency: r.recency,
          tickerHit: r.doc.ticker && c.query.toUpperCase().includes(r.doc.ticker) ? 1 : 0,
          titleHit: c.query.toLowerCase().split(/\s+/).some((w) =>
            r.doc.title.toLowerCase().includes(w),
          )
            ? 1
            : 0,
        },
        y: rel.has(r.doc.id) ? 1 : 0,
      });
    }
  }
  return data;
}

let loraWeights: RerankWeights = [...EMBED_WEIGHTS];
let qloraWeights: RerankWeights = [...EMBED_WEIGHTS];
let loraLoss: number[] = [];
let qloraLoss: number[] = [];
let lastTrainMs = 0;

export function getAdapterWeights(kind: RetrieverKind): RerankWeights | undefined {
  if (kind === "lora") return loraWeights;
  if (kind === "embed") return EMBED_WEIGHTS;
  return BASELINE_WEIGHTS;
}

export function trainAdapters() {
  const data = buildPairs();
  const t0 = performance.now();
  const lora = trainLogistic(data, 56, 0.4, 0.008, 0);
  const qlora = trainLogistic(data, 40, 0.45, 0.02, 1);
  loraWeights = lora.weights;
  qloraWeights = qlora.weights;
  loraLoss = lora.loss;
  qloraLoss = qlora.loss;
  lastTrainMs = performance.now() - t0;
  return { loraLoss, qloraLoss, lastTrainMs, pairs: data.length };
}

export function adapterLoss() {
  return { loraLoss, qloraLoss, lastTrainMs };
}

function metricsFor(kind: RetrieverKind, w?: RerankWeights): EvalMetrics {
  return runEvalSuite(kind, w);
}

export function defaultCheckpoints(): Checkpoint[] {
  const lexical = metricsFor("lexical");
  const embed = metricsFor("embed");
  const lora = metricsFor("lora", loraWeights);
  const qlora = runEvalSuite("lora", qloraWeights);
  return [
    {
      id: "ckpt-lex-01",
      name: "lexical-bm25",
      kind: "baseline",
      createdAt: Date.UTC(2026, 6, 12),
      metrics: lexical,
      stage: "staging",
      lineage: [],
    },
    {
      id: "ckpt-emb-02",
      name: "embed-hash64",
      kind: "embed",
      createdAt: Date.UTC(2026, 7, 2),
      metrics: embed,
      stage: "shadow",
      lineage: ["ckpt-lex-01"],
    },
    {
      id: "ckpt-qlora-r4",
      name: "rerank-qlora-r4",
      kind: "qlora",
      rank: 4,
      createdAt: Date.UTC(2026, 7, 22),
      metrics: qlora,
      stage: "canary",
      lineage: ["ckpt-lex-01", "ckpt-emb-02"],
    },
    {
      id: "ckpt-lora-r8",
      name: "rerank-lora-r8",
      kind: "lora",
      rank: 8,
      createdAt: Date.UTC(2026, 8, 4),
      metrics: lora,
      stage: "production",
      lineage: ["ckpt-lex-01", "ckpt-emb-02", "ckpt-qlora-r4"],
    },
  ];
}

export function gateCheck(m: EvalMetrics) {
  return [
    { name: "nDCG@10 ≥ 0.72", ok: m.ndcgAt10 >= GATES.ndcgAt10, value: m.ndcgAt10 },
    { name: "Recall@5 ≥ 0.80", ok: m.recallAt5 >= GATES.recallAt5, value: m.recallAt5 },
    { name: "Groundedness ≥ 0.85", ok: m.groundedness >= GATES.groundedness, value: m.groundedness },
    { name: "p95 < 800ms", ok: m.p95Ms < GATES.p95Ms, value: m.p95Ms },
    { name: "Cost/query < $0.02", ok: m.costUsd < GATES.costUsd, value: m.costUsd },
  ];
}

export function nextStage(s: PromotionStage): PromotionStage | null {
  if (s === "staging") return "shadow";
  if (s === "shadow") return "canary";
  if (s === "canary") return "production";
  return null;
}

export function canPromote(m: EvalMetrics) {
  return gateCheck(m).every((g) => g.ok);
}
