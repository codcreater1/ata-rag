import { useEffect, useRef, useState } from "react";
import { ExternalLink, Send, ThumbsDown, ThumbsUp } from "lucide-react";

import { askStream, getSuggestions, sendFeedback, trackSourceClick } from "./api";

function Sources({ sources, queryId }) {
  if (!sources?.length) return null;
  return (
    <div className="sources">
      <span className="sourcesLabel">Sources</span>
      {sources.map((s) => (
        <a
          key={s.n}
          href={s.url}
          target="_blank"
          rel="noreferrer"
          className="source"
          onClick={() => trackSourceClick(s.url, s.title, queryId)}
        >
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
    // Send the recent exchange so follow-ups ("and in Wrocław?") resolve.
    const history = messages
      .slice(-6)
      .map((m) => ({
        role: m.role,
        content: m.role === "user" ? m.text : m.answer,
      }))
      .filter((m) => m.content);

    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);

    // The assistant bubble is created empty and filled as tokens arrive.
    const index = messages.length + 1;
    setMessages((m) => [...m, { role: "assistant", answer: "", sources: [], streaming: true }]);

    const patch = (fields) =>
      setMessages((m) => m.map((msg, i) => (i === index ? { ...msg, ...fields } : msg)));

    try {
      let text = "";
      await askStream(q, language, history, {
        onSources: (sources) => patch({ sources }),
        onToken: (piece) => {
          text += piece;
          patch({ answer: text });
        },
        onDone: (meta) => patch({ ...meta, streaming: false }),
      });
    } catch (err) {
      patch({ answer: err.message, sources: [], answered: false, streaming: false });
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
                {m.streaming && !m.answer ? (
                  <span className="typing"><i></i><i></i><i></i></span>
                ) : (
                  <p className="answer">
                    {m.answer}
                    {m.streaming && <span className="caret" />}
                  </p>
                )}
                <Sources sources={m.sources} queryId={m.query_id} />
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
