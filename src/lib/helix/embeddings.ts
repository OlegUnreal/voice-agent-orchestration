import { hash32 } from "./prng";

export const EMBED_DIM = 64;

const STOP = new Set([
  "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "as",
  "at", "by", "from", "that", "this", "be", "are", "was", "it", "into",
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9%\-]+/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t));
}

export function embed(text: string): Float32Array {
  const v = new Float32Array(EMBED_DIM);
  const tokens = tokenize(text);
  const grams: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    grams.push(tokens[i]!);
    if (i + 1 < tokens.length) grams.push(tokens[i] + "_" + tokens[i + 1]);
  }
  for (const g of grams) {
    for (let k = 0; k < 3; k++) {
      const h = hash32(g, k + 1) % EMBED_DIM;
      const sign = hash32(g, k + 17) % 2 === 0 ? 1 : -1;
      v[h] += sign;
    }
  }
  let n = 0;
  for (let i = 0; i < EMBED_DIM; i++) n += v[i]! * v[i]!;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < EMBED_DIM; i++) v[i]! /= n;
  return v;
}

export function cosine(a: Float32Array, b: Float32Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}
