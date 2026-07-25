export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function ask(question, language = "auto") {
  const res = await fetch(`${API_URL}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, language }),
  });
  if (!res.ok) throw new Error("The assistant is unavailable right now.");
  return res.json();
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
