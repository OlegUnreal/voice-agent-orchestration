export interface RerankFeatures {
  cosine: number;
  recency: number;
  tickerHit: number;
  titleHit: number;
}

export type RerankWeights = [number, number, number, number, number];

export const BASELINE_WEIGHTS: RerankWeights = [0, 1.4, 0.2, 0.4, 0.5];
export const EMBED_WEIGHTS: RerankWeights = [-0.2, 2.2, 0.6, 0.8, 0.4];

function feat(f: RerankFeatures): number[] {
  return [1, f.cosine, f.recency, f.tickerHit, f.titleHit];
}

export function scoreRerank(f: RerankFeatures, w: RerankWeights = EMBED_WEIGHTS) {
  const x = feat(f);
  let z = 0;
  for (let i = 0; i < w.length; i++) z += w[i]! * x[i]!;
  return 1 / (1 + Math.exp(-z));
}

export interface TrainPoint {
  features: RerankFeatures;
  y: 0 | 1;
}

export function trainLogistic(
  data: TrainPoint[],
  epochs = 48,
  lr = 0.35,
  l2 = 0.01,
  rankDrop = 0,
): { weights: RerankWeights; loss: number[] } {
  const w: RerankWeights = [0, 0.4, 0.1, 0.1, 0.1];
  const loss: number[] = [];
  const dim = rankDrop > 0 ? 5 - rankDrop : 5;
  for (let e = 0; e < epochs; e++) {
    let l = 0;
    const g = [0, 0, 0, 0, 0];
    for (const p of data) {
      const x = feat(p.features);
      let z = 0;
      for (let i = 0; i < dim; i++) z += w[i]! * x[i]!;
      const pred = 1 / (1 + Math.exp(-z));
      const err = pred - p.y;
      l += -(p.y * Math.log(pred + 1e-9) + (1 - p.y) * Math.log(1 - pred + 1e-9));
      for (let i = 0; i < dim; i++) g[i]! += err * x[i]!;
    }
    const n = data.length || 1;
    for (let i = 0; i < dim; i++) {
      w[i] = w[i]! - lr * (g[i]! / n + l2 * w[i]!);
    }
    loss.push(l / n);
  }
  return { weights: w, loss };
}
