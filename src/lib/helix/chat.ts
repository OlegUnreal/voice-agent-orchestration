import { createServerFn } from "@tanstack/react-start";
import { pythonTurn, pythonVerify } from "./engine";
import { grokToolPayload } from "./orchestrator";
import type { OrchestratorResult } from "./types";

const cache = new Map<string, { spoken: string; at: number }>();
const TTL = 10 * 60_000;

function cacheKey(text: string) {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

async function synthesize(result: OrchestratorResult & { grokPayload?: unknown }, query: string) {
  const apiKey = process.env.XAI_API_KEY;
  if (!apiKey) {
    return { spoken: result.fallbackSpoken, ai: false as const, model: "local-tools" };
  }
  const key = cacheKey(query);
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < TTL) {
    return { spoken: hit.spoken, ai: true as const, model: "grok-4.5-cache" };
  }
  const payload =
    "grokPayload" in result && result.grokPayload
      ? result.grokPayload
      : grokToolPayload(result);
  const res = await fetch("https://api.x.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "grok-4.5",
      max_tokens: 420,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content:
            "You are Helix, a voice financial copilot. TOOL RESULTS are the only source of numbers. Never invent prices, Sharpe, VaR, or citations. Speak 2–4 sentences of plain text for voice (no markdown, no bullets). Then one short recap line. Mention citation ids when present. If a number is missing from tools, say you do not have it.",
        },
        {
          role: "user",
          content: `User said: ${query}\n\nTOOL RESULTS (authoritative JSON):\n${JSON.stringify(payload)}`,
        },
      ],
    }),
  });
  if (!res.ok) {
    return { spoken: result.fallbackSpoken, ai: false as const, model: `xai-${res.status}` };
  }
  const body = (await res.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  const spoken = body.choices?.[0]?.message?.content?.trim() || result.fallbackSpoken;
  return { spoken, ai: true as const, model: "grok-4.5" };
}

export const runHelixTurn = createServerFn({ method: "POST" })
  .validator((input: { text: string }) => input)
  .handler(async ({ data }) => {
    const text = data.text.slice(0, 800);
    const local = await pythonTurn(text);
    const t0 = Date.now();
    const syn = await synthesize(local, text);
    const payload =
      "grokPayload" in local && local.grokPayload
        ? local.grokPayload
        : grokToolPayload(local);
    const checked = await pythonVerify({
      spoken: syn.spoken,
      fallbackSpoken: local.fallbackSpoken,
      payload,
      traceId: local.traceId,
      query: text,
      model: syn.model,
    });
    const spoken = checked.spoken || local.fallbackSpoken;
    if (syn.ai && checked.ok) {
      cache.set(cacheKey(text), { spoken, at: Date.now() });
    }
    return {
      intent: local.intent,
      hops: local.hops,
      tools: local.tools,
      citations: local.citations,
      snapshot: local.snapshot,
      anomalies: local.anomalies?.slice(0, 5),
      backtest: local.backtest
        ? { ...local.backtest, equity: local.backtest.equity.filter((_, i) => i % 4 === 0) }
        : undefined,
      risk: local.risk,
      retrieved: (local.retrieved ?? []).slice(0, 5).map((r) => ({
        id: r.doc.id,
        title: r.doc.title,
        type: r.doc.type,
        ts: r.doc.ts,
        score: r.score,
        cosine: r.cosine,
        recency: r.recency,
        snippet: r.doc.body.slice(0, 180),
      })),
      numbers: local.numbers,
      spoken,
      draftSpoken: syn.spoken,
      fallbackSpoken: local.fallbackSpoken,
      grounded: local.grounded && checked.ok,
      verified: checked.ok,
      verifyLeaks: checked.leaks,
      traceId: local.traceId,
      ai: syn.ai && checked.ok,
      model: checked.ok ? syn.model : "verified-fallback",
      totalMs: Date.now() - t0 + local.hops.reduce((s, h) => s + h.ms, 0),
    };
  });

export const speakHelix = createServerFn({ method: "POST" })
  .validator((input: { text: string }) => input)
  .handler(async ({ data }) => {
    const apiKey = process.env.XAI_API_KEY;
    if (!apiKey) return { ok: false as const, error: "Voice is not available" };
    const text = data.text.slice(0, 900);
    const res = await fetch("https://api.x.ai/v1/audio/speech", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "grok-voice-latest",
        voice: "orion",
        input: text,
      }),
    });
    if (!res.ok) {
      const retry = await fetch("https://api.x.ai/v1/tts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({ text, voice_id: "orion" }),
      });
      if (!retry.ok) return { ok: false as const, error: `TTS ${res.status}` };
      const buf = Buffer.from(await retry.arrayBuffer());
      return { ok: true as const, mime: "audio/mpeg", audio: buf.toString("base64") };
    }
    const buf = Buffer.from(await res.arrayBuffer());
    return {
      ok: true as const,
      mime: res.headers.get("content-type") ?? "audio/mpeg",
      audio: buf.toString("base64"),
    };
  });
