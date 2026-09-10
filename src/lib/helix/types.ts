export const TICKERS = ["BTC", "ETH", "SOL", "NVDA", "AAPL"] as const;
export type Ticker = (typeof TICKERS)[number];

export type Regime = "bull-trend" | "bear-trend" | "range" | "high-vol";

export type Intent =
  | "regime"
  | "anomaly"
  | "backtest"
  | "risk"
  | "research"
  | "portfolio"
  | "eval"
  | "training"
  | "memory"
  | "general";

export type AgentId =
  | "supervisor"
  | "quant"
  | "research"
  | "copilot"
  | "eval"
  | "training";

export type PromotionStage = "staging" | "shadow" | "canary" | "production";

export type DocType = "research" | "onchain" | "strategy" | "risk" | "event";

export interface Bar {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export interface OnChainPoint {
  t: number;
  activeAddresses: number;
  exchangeInflow: number;
  fundingRate: number;
}

export interface MarketSnapshot {
  ticker: Ticker;
  price: number;
  change1d: number;
  change20d: number;
  vol20d: number;
  sma20: number;
  sma50: number;
  sma200: number;
  regime: Regime;
  lastAnomaly?: Anomaly;
}

export interface Anomaly {
  ticker: Ticker;
  t: number;
  kind: "return" | "volume" | "funding" | "inflow";
  z: number;
  note: string;
}

export interface BacktestResult {
  ticker: Ticker;
  strategy: string;
  params: Record<string, number>;
  totalReturn: number;
  cagr: number;
  sharpe: number;
  maxDrawdown: number;
  winRate: number;
  trades: number;
  equity: { t: number; v: number }[];
}

export interface RiskResult {
  portfolioValue: number;
  var95: number;
  cvar95: number;
  volAnn: number;
  maxDrawdown: number;
  scenarioPnl: { name: string; pnl: number; pct: number }[];
  weights: { ticker: Ticker; weight: number; value: number }[];
}

export interface KnowledgeDoc {
  id: string;
  title: string;
  body: string;
  ticker?: Ticker;
  type: DocType;
  ts: number;
}

export interface RetrievedDoc {
  doc: KnowledgeDoc;
  score: number;
  cosine: number;
  recency: number;
  rerank?: number;
}

export interface Citation {
  id: string;
  title: string;
  type: DocType;
  ts: number;
}

export interface ToolCall {
  name: string;
  args: Record<string, string | number | boolean>;
  ms: number;
  ok: boolean;
  summary: string;
}

export interface AgentHop {
  agent: AgentId;
  label: string;
  ms: number;
}

export interface OrchestratorResult {
  intent: Intent;
  hops: AgentHop[];
  tools: ToolCall[];
  citations: Citation[];
  snapshot?: MarketSnapshot;
  anomalies?: Anomaly[];
  backtest?: BacktestResult;
  risk?: RiskResult;
  retrieved?: RetrievedDoc[];
  numbers: Record<string, number | string>;
  fallbackSpoken: string;
  grounded: boolean;
  verified?: boolean;
  verifyLeaks?: string[];
  traceId?: string;
  memory?: { id: string; key: string; value: string; source: string }[];
}

export interface EvalCase {
  id: string;
  query: string;
  asOf: number;
  relevant: string[];
  expectedIntent: Intent;
  expectedTools: string[];
}

export interface EvalMetrics {
  recallAt5: number;
  recallAt10: number;
  mrr: number;
  ndcgAt10: number;
  intentAcc: number;
  toolAcc: number;
  groundedness: number;
  citationCorrect: number;
  structuredOk: number;
  p50Ms: number;
  p95Ms: number;
  costUsd: number;
}

export interface Checkpoint {
  id: string;
  name: string;
  kind: "baseline" | "embed" | "lora" | "qlora";
  rank?: number;
  createdAt: number;
  metrics: EvalMetrics;
  stage: PromotionStage;
  lineage: string[];
}

export interface Trace {
  id: string;
  ts: number;
  query: string;
  intent: Intent;
  tools: string[];
  latencyMs: number;
  tokens: number;
  costUsd: number;
  retrievalHit: boolean;
  feedback?: "up" | "down";
}

export interface McpTool {
  name: string;
  server: string;
  description: string;
  governed: "auto" | "approve";
  schema: Record<string, string>;
}
