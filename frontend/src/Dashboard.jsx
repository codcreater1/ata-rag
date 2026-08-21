import { useEffect, useState } from "react";
import { AlertTriangle, ClipboardList, Coins, Database, Download, FileText, Gauge, MessageSquare, MousePointerClick, ThumbsDown, ThumbsUp, Zap } from "lucide-react";

import { getActionItems, getClickedSources, getGaps, getStats, getTopQuestions } from "./api";

// The university acts on these off the dashboard, so make them portable. A CSV
// downloads cleanly into a spreadsheet; the browser sandbox permits it here (a
// real app, not an embedded artifact).
function downloadActionItemsCsv(unanswered, disliked) {
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = [["type", "question", "count", "detail", "last_asked"]];
  for (const g of unanswered)
    rows.push(["missing answer", g.question, g.times_asked,
      `match ${Math.round((g.avg_similarity || 0) * 100)}%`, g.last_asked]);
  for (const d of disliked)
    rows.push(["unhelpful answer", d.question, d.not_helpful,
      `${d.not_helpful} of ${d.times_asked} 👎`, d.last_asked]);
  const csv = rows.map((r) => r.map(esc).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `ata-content-gaps-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function Card({ icon, label, value, sub }) {
  return (
    <div className="statCard">
      <div className="statIcon">{icon}</div>
      <div>
        <div className="statValue">{value}</div>
        <div className="statLabel">{label}</div>
        {sub && <div className="statSub">{sub}</div>}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [gaps, setGaps] = useState([]);
  const [top, setTop] = useState([]);
  const [clicked, setClicked] = useState([]);
  const [actions, setActions] = useState({ unanswered: [], disliked: [] });
  const [error, setError] = useState("");

  useEffect(() => {
    getStats().then(setStats).catch(() => setError("Dashboard data unavailable."));
    getGaps().then((d) => setGaps(d.gaps || []));
    getTopQuestions().then((d) => setTop(d.questions || []));
    getClickedSources().then((d) => setClicked(d.sources || []));
    getActionItems().then((d) => setActions({ unanswered: d.unanswered || [], disliked: d.disliked || [] }));
  }, []);

  if (error) return <div className="dashError">{error}</div>;
  if (!stats) return <div className="dashLoading">Loading analytics…</div>;

  const { index, usage, cache } = stats;
  const helpfulRate =
    usage.helpful + usage.not_helpful > 0
      ? Math.round((usage.helpful / (usage.helpful + usage.not_helpful)) * 100)
      : null;

  return (
    <div className="dashboard">
      <div className="statGrid">
        <Card icon={<FileText size={20} />} label="Documents indexed" value={index.documents ?? 0} />
        <Card icon={<Database size={20} />} label="Chunks" value={index.chunks ?? 0} />
        <Card icon={<MessageSquare size={20} />} label="Questions asked" value={usage.questions ?? 0} />
        <Card
          icon={<AlertTriangle size={20} />}
          label="Unanswered"
          value={usage.unanswered ?? 0}
          sub="knowledge gaps"
        />
        <Card
          icon={<Gauge size={20} />}
          label="Avg. match"
          value={usage.avg_similarity != null ? `${Math.round(usage.avg_similarity * 100)}%` : "—"}
          sub={usage.avg_latency_ms ? `${usage.avg_latency_ms} ms avg` : ""}
        />
        <Card
          icon={<Coins size={20} />}
          label="Tokens used"
          value={
            usage.total_tokens
              ? usage.total_tokens >= 1000
                ? `${(usage.total_tokens / 1000).toFixed(1)}k`
                : usage.total_tokens
              : "—"
          }
          sub={`${usage.prompt_tokens || 0} in / ${usage.completion_tokens || 0} out`}
        />
        <Card
          icon={<ThumbsUp size={20} />}
          label="Helpful rate"
          value={helpfulRate != null ? `${helpfulRate}%` : "—"}
          sub={`${usage.helpful || 0}👍 / ${usage.not_helpful || 0}👎`}
        />
        <Card
          icon={<Zap size={20} />}
          label="Answers reused"
          value={cache?.answers_reused ?? 0}
          sub={`${cache?.cached_answers ?? 0} cached · saved model calls`}
        />
      </div>

      {(actions.unanswered.length > 0 || actions.disliked.length > 0) && (
        <section className="dashPanel">
          <div className="panelHead">
            <h3><ClipboardList size={16} /> Action items — content to add or fix</h3>
            <button
              className="exportBtn"
              onClick={() => downloadActionItemsCsv(actions.unanswered, actions.disliked)}
              title="Download as CSV"
            >
              <Download size={14} /> Export CSV
            </button>
          </div>
          <p className="panelHint">
            What to publish on the website next: questions the assistant couldn&apos;t
            answer, and answers visitors marked unhelpful.
          </p>
          <div className="dashCols">
            <div>
              <div className="actionLabel"><AlertTriangle size={13} /> Missing answers</div>
              {actions.unanswered.length === 0 ? (
                <div className="dashEmpty">None.</div>
              ) : (
                <ul className="qList">
                  {actions.unanswered.map((g, i) => (
                    <li key={i}>
                      <span className="qText">{g.question}</span>
                      <span className="qMeta">×{g.times_asked}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <div className="actionLabel"><ThumbsDown size={13} /> Unhelpful answers</div>
              {actions.disliked.length === 0 ? (
                <div className="dashEmpty">None — every rated answer was helpful.</div>
              ) : (
                <ul className="qList">
                  {actions.disliked.map((d, i) => (
                    <li key={i}>
                      <span className="qText">{d.question}</span>
                      <span className="qMeta">{d.not_helpful} 👎</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </section>
      )}

      <div className="dashCols">
        <section className="dashPanel">
          <h3><AlertTriangle size={16} /> Unanswered questions</h3>
          <p className="panelHint">Where the knowledge base — or the website — has gaps.</p>
          {gaps.length === 0 ? (
            <div className="dashEmpty">No unanswered questions yet.</div>
          ) : (
            <ul className="qList">
              {gaps.map((g, i) => (
                <li key={i}>
                  <span className="qText">{g.question}</span>
                  <span className="qMeta">
                    ×{g.times_asked} · match {Math.round((g.avg_similarity || 0) * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="dashPanel">
          <h3><MessageSquare size={16} /> Most asked</h3>
          <p className="panelHint">Popular questions and whether they were answered.</p>
          {top.length === 0 ? (
            <div className="dashEmpty">No questions yet.</div>
          ) : (
            <ul className="qList">
              {top.map((q, i) => (
                <li key={i}>
                  <span className="qText">{q.question}</span>
                  <span className="qMeta">
                    ×{q.times_asked}{" "}
                    <span className={q.ever_answered ? "dot ok" : "dot warn"} />
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="dashPanel">
        <h3><MousePointerClick size={16} /> Most clicked sources</h3>
        <p className="panelHint">Which cited pages visitors actually opened.</p>
        {clicked.length === 0 ? (
          <div className="dashEmpty">No source clicks recorded yet.</div>
        ) : (
          <ul className="qList">
            {clicked.map((s, i) => (
              <li key={i}>
                <a className="qText srcLink" href={s.url} target="_blank" rel="noreferrer">
                  {s.title || s.url}
                </a>
                <span className="qMeta">{s.clicks} clicks</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
