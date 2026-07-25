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

/** Stream an answer as Server-Sent Events.
 *  Calls onSources(list), onToken(text) and onDone(meta) as they arrive. */
export async function askStream(
  question,
  language = "auto",
  history = [],
  { onSources, onToken, onDone } = {},
) {
  const res = await fetch(`${API_URL}/chat/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, language, history }),
  });
  if (!res.ok || !res.body) throw new Error("The assistant is unavailable right now.");

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
      else if (event === "token") onToken?.(payload.text || "");
      else if (event === "done") onDone?.(payload);
    }
  }
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
