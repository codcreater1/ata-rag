export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function ask(question, language = "auto", history = []) {
  const res = await fetch(`${API_URL}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, language, history }),
  });
  if (!res.ok) throw new Error("The assistant is unavailable right now.");
  return res.json();
}

/** Get an answer, streamed when the browser can, in one piece when it can't.
 *  Calls onSources(list), onToken(text) and onDone(meta) as they arrive. */
export async function askStream(
  question,
  language = "auto",
  history = [],
  { onSources, onToken, onDone } = {},
) {
  // Track whether any token reached the UI: if streaming fails *after* output
  // has started, falling back would duplicate the answer, so only an untouched
  // failure falls through to the non-streaming request.
  let emitted = false;

  try {
    const res = await fetch(`${API_URL}/chat/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language, history }),
    });
    if (!res.ok) throw new Error("bad status");
    // iOS Safari's fetch often has no readable body (streaming unsupported);
    // treat that as "cannot stream" and fall back rather than erroring.
    if (!res.body || typeof res.body.getReader !== "function") {
      throw new Error("no stream body");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; keep any partial tail.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const event = frame.match(/^event: (.+)$/m)?.[1];
        const raw = frame.match(/^data: (.+)$/m)?.[1];
        if (!event || !raw) continue;
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch {
          continue;
        }
        if (event === "sources") onSources?.(payload.sources || []);
        else if (event === "token") { emitted = true; onToken?.(payload.text || ""); }
        else if (event === "done") onDone?.(payload);
      }
    }
    return;
  } catch (err) {
    // Streaming isn't available here (typically iOS Safari, which rejects the
    // streamed fetch with "Load failed"). Fall back to the plain JSON endpoint
    // so the answer still arrives — just all at once instead of token by token.
    if (emitted) throw err;
  }

  const data = await ask(question, language, history);
  onSources?.(data.sources || []);
  onToken?.(data.answer || "");
  onDone?.({
    answered: data.answered,
    confidence: data.confidence,
    latency_ms: data.latency_ms,
    query_id: data.query_id,
    sources: data.sources || [],
    cached: data.cached,
  });
}

export async function getSuggestions() {
  const res = await fetch(`${API_URL}/chat/suggestions`);
  if (!res.ok) return { pl: [], en: [] };
  return res.json();
}

export async function sendFeedback(queryId, helpful) {
  if (queryId == null) return;
  await fetch(`${API_URL}/chat/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_id: queryId, helpful }),
  }).catch(() => {});
}

// Dashboard
export async function trackSourceClick(url, title, queryId) {
  // Fire-and-forget: analytics must never delay opening the link.
  fetch(`${API_URL}/chat/source-click`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, title, query_id: queryId ?? null }),
    keepalive: true,
  }).catch(() => {});
}

export async function getClickedSources() {
  const res = await fetch(`${API_URL}/dashboard/clicked-sources`);
  if (!res.ok) return { sources: [] };
  return res.json();
}

export async function getStats() {
  const res = await fetch(`${API_URL}/dashboard/stats`);
  if (!res.ok) throw new Error("stats unavailable");
  return res.json();
}

export async function getGaps() {
  const res = await fetch(`${API_URL}/dashboard/gaps`);
  if (!res.ok) return { gaps: [] };
  return res.json();
}

export async function getTopQuestions() {
  const res = await fetch(`${API_URL}/dashboard/top-questions`);
  if (!res.ok) return { questions: [] };
  return res.json();
}
