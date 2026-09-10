import { cosine, embed } from "./embeddings";
import type { Intent } from "./types";

const PROTOTYPES: { intent: Intent; text: string }[] = [
  { intent: "regime", text: "regime trend bull bear high vol range sma market state" },
  { intent: "anomaly", text: "anomaly dump spike inflow funding zscore unusual stress" },
  { intent: "backtest", text: "backtest sma crossover strategy sharpe drawdown trades" },
  { intent: "risk", text: "risk var cvar shock scenario pnl portfolio drop crash" },
  { intent: "research", text: "cite evidence research note on-chain why explain grounded" },
  { intent: "portfolio", text: "portfolio review book weights allocation ballast copilot" },
  { intent: "eval", text: "eval metrics recall ndcg promotion gates golden regression" },
  { intent: "training", text: "train lora qlora checkpoint dataset sft reranker registry" },
  { intent: "memory", text: "remember forget recall my mandate risk cap note fact" },
  { intent: "general", text: "hello help what can you do helix voice agent mcp tools" },
];

const VEC = PROTOTYPES.map((p) => ({ intent: p.intent, v: embed(p.text) }));

export function classifyIntent(query: string): Intent {
  const q = query.toLowerCase();
  if (/\b(backtest|sharpe|sma crossover|cagr)\b/.test(q)) return "backtest";
  if (/\b(var|cvar|shock|drop 12|risk if|scenario)\b/.test(q)) return "risk";
  if (/\b(anomal|inflow|dump|funding|z-score|zscore|stress window)\b/.test(q)) return "anomaly";
  if (/\b(regime|bull|bear|high-vol|high vol)\b/.test(q)) return "regime";
  if (/\b(promote|ndcg|recall@|golden|evalforge|eval)\b/.test(q)) return "eval";
  if (/\b(lora|qlora|checkpoint|train|sft|registry)\b/.test(q)) return "training";
  if (/\b(remember|forget|what do you remember)\b/.test(q)) return "memory";
  if (/\b(portfolio|book|weights|copilot review)\b/.test(q)) return "portfolio";
  if (/\b(cite|evidence|why|research|staking|on-chain|onchain)\b/.test(q)) return "research";
  const qv = embed(query);
  let best: Intent = "general";
  let score = -1;
  for (const p of VEC) {
    const s = cosine(qv, p.v);
    if (s > score) {
      score = s;
      best = p.intent;
    }
  }
  return score < 0.12 ? "general" : best;
}
