import { CORPUS } from "./corpus";
import { cosine, embed, tokenize } from "./embeddings";
import { AS_OF } from "./market";
import { scoreRerank, type RerankWeights } from "./reranker";
import type { KnowledgeDoc, RetrievedDoc } from "./types";

const HALF_LIFE = 14 * 86_400_000;
const INDEX = CORPUS.map((doc) => ({
  doc,
  vec: embed(`${doc.title} ${doc.body} ${doc.ticker ?? ""} ${doc.type}`),
}));

export function recencyScore(ts: number, asOf = AS_OF) {
  const age = Math.max(0, asOf - ts);
  return Math.exp(-Math.LN2 * age / HALF_LIFE);
}

export function retrieve(query: string, opts?: {
  k?: number;
  asOf?: number;
  chronological?: boolean;
  weights?: RerankWeights;
}): RetrievedDoc[] {
  const k = opts?.k ?? 8;
  const asOf = opts?.asOf ?? AS_OF;
  const chronological = opts?.chronological ?? true;
  const qv = embed(query);
  const qTokens = new Set(tokenize(query));
  const scored: RetrievedDoc[] = [];
  for (const row of INDEX) {
    if (chronological && row.doc.ts > asOf) continue;
    const cos = cosine(qv, row.vec);
    const rec = recencyScore(row.doc.ts, asOf);
    const tickerHit = row.doc.ticker && query.toUpperCase().includes(row.doc.ticker) ? 1 : 0;
    const titleHit = tokenize(row.doc.title).some((t) => qTokens.has(t)) ? 1 : 0;
    const docToks = tokenize(`${row.doc.title} ${row.doc.body} ${row.doc.id}`);
    let hit = 0;
    for (const t of docToks) if (qTokens.has(t)) hit++;
    const overlap = hit / Math.sqrt((qTokens.size || 1) * (docToks.length || 1));
    const rerank = scoreRerank({ cosine: cos, recency: rec, tickerHit, titleHit }, opts?.weights);
    const score = 0.4 * rerank + 0.4 * overlap + 0.12 * tickerHit + 0.08 * rec;
    scored.push({
      doc: row.doc,
      cosine: cos,
      recency: rec,
      rerank,
      score,
    });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

export function lexicalRetrieve(query: string, k = 8, asOf = AS_OF): RetrievedDoc[] {
  const q = new Set(tokenize(query));
  const scored: RetrievedDoc[] = [];
  for (const row of INDEX) {
    if (row.doc.ts > asOf) continue;
    const toks = tokenize(`${row.doc.title} ${row.doc.body}`);
    let hit = 0;
    for (const t of toks) if (q.has(t)) hit++;
    const score = hit / Math.sqrt(toks.length || 1);
    scored.push({
      doc: row.doc,
      cosine: score,
      recency: recencyScore(row.doc.ts, asOf),
      score,
    });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

export function docById(id: string): KnowledgeDoc | undefined {
  return CORPUS.find((d) => d.id === id);
}
