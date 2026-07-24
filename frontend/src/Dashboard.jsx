import { useEffect, useState } from "react";
import { AlertTriangle, Database, FileText, Gauge, MessageSquare, ThumbsUp } from "lucide-react";

import { getGaps, getStats, getTopQuestions } from "./api";

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
  const [error, setError] = useState("");

  useEffect(() => {
    getStats().then(setStats).catch(() => setError("Dashboard data unavailable."));
    getGaps().then((d) => setGaps(d.gaps || []));
    getTopQuestions().then((d) => setTop(d.questions || []));
  }, []);

  if (error) return <div className="dashError">{error}</div>;
  if (!stats) return <div className="dashLoading">Loading analytics…</div>;

  const { index, usage } = stats;
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
          icon={<ThumbsUp size={20} />}
          label="Helpful rate"
          value={helpfulRate != null ? `${helpfulRate}%` : "—"}
          sub={`${usage.helpful || 0}👍 / ${usage.not_helpful || 0}👎`}
        />
      </div>

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
    </div>
  );
}
