import { useEffect, useRef, useState } from "react";
import { ExternalLink, Send, ThumbsDown, ThumbsUp } from "lucide-react";

import { ask, getSuggestions, sendFeedback } from "./api";

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="sources">
      <span className="sourcesLabel">Sources</span>
      {sources.map((s) => (
        <a key={s.n} href={s.url} target="_blank" rel="noreferrer" className="source">
          <ExternalLink size={12} /> {s.title}
        </a>
      ))}
    </div>
  );
}

function Feedback({ queryId }) {
  const [sent, setSent] = useState(null);
  if (queryId == null) return null;

  return (
    <div className="feedback">
      {sent ? (
        <span className="feedbackDone">Thanks for the feedback</span>
      ) : (
        <>
          <button onClick={() => { sendFeedback(queryId, true); setSent("up"); }}>
            <ThumbsUp size={13} /> Helpful
          </button>
          <button onClick={() => { sendFeedback(queryId, false); setSent("down"); }}>
            <ThumbsDown size={13} /> Not helpful
          </button>
        </>
      )}
    </div>
  );
}

export default function Chat({ language = "auto" }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    getSuggestions().then((s) => setSuggestions([...(s.en || []), ...(s.pl || [])]));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function submit(question) {
    const q = (question ?? input).trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const res = await ask(q, language);
      setMessages((m) => [...m, { role: "assistant", ...res }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", answer: err.message, sources: [], answered: false },
      ]);
    } finally {
      setBusy(false);
    }
  }

  const empty = messages.length === 0;

  return (
    <div className="chat">
      <div className="thread">
        {empty && (
          <div className="welcome">
            <img className="welcomeLogo" src="/ata-bear.png" alt="" width="72" height="72" />
            <h2>Ask about studying at ATA</h2>
            <p>
              Tuition, admissions, programmes, student services — answered from the
              university website, with sources.
            </p>
            <div className="suggestions">
              {suggestions.map((s, i) => (
                <button key={i} className="suggestion" onClick={() => submit(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg user">
              <div className="bubble">{m.text}</div>
            </div>
          ) : (
            <div key={i} className="msg assistant">
              <div className="bubble">
                <p className="answer">{m.answer}</p>
                <Sources sources={m.sources} />
                {m.confidence != null && (
                  <div className="metaRow">
                    <span className={`badge ${m.answered ? "ok" : "warn"}`}>
                      {m.answered ? "answered" : "not found"}
                    </span>
                    {m.confidence != null && (
                      <span className="dim">match {(m.confidence * 100).toFixed(0)}%</span>
                    )}
                    {m.latency_ms != null && (
                      <span className="dim">{m.latency_ms} ms</span>
                    )}
                    <Feedback queryId={m.query_id} />
                  </div>
                )}
              </div>
            </div>
          ),
        )}

        {busy && (
          <div className="msg assistant">
            <div className="bubble">
              <span className="typing">
                <i></i><i></i><i></i>
              </span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about tuition, admissions, programmes…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
