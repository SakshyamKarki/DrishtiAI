import { useNavigate } from "react-router-dom";
import "../styles/detection.css";
import { useSelector } from "react-redux";
import { selectUser } from "../features/auth/authSlice";
import { useEffect, useState } from "react";
import { api } from "../api/axiosInstance";

// ── Risk color helper ──────────────────────────────────────────────────────
const verdictColors = {
  FAKE: { main: "#f85149", bg: "rgba(248,81,73,0.08)", border: "rgba(248,81,73,0.2)" },
  REAL: { main: "#56d364", bg: "rgba(86,211,100,0.08)", border: "rgba(86,211,100,0.2)" },
  SUSPICIOUS: { main: "#e3b341", bg: "rgba(227,179,65,0.08)", border: "rgba(227,179,65,0.2)" },
};

// ── Detection Card ─────────────────────────────────────────────────────────
const DetectionCard = ({ item }) => {
  const verdict = item.verdict || (item.is_fake ? "FAKE" : "REAL");
  const c = verdictColors[verdict] || verdictColors.SUSPICIOUS;

  const formatDate = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) +
      " · " + d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="det-card" style={{ border: `1px solid ${c.border}` }}>
      <div className="det-img" style={{ background: c.bg }}>
        {item.image ? (
          <img src={item.image} alt="detection" className="det-img-tag" />
        ) : (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", fontSize: "32px", opacity: 0.3 }}>
            👤
          </div>
        )}
        <span className="det-badge" style={{ background: c.bg, color: c.main, border: `0.5px solid ${c.border}` }}>
          {verdict}
        </span>
      </div>
      <div className="det-body">
        <div className="det-filename">{item.filename || `Detection #${item.id}`}</div>
        <div className="det-meta">
          <span className="det-mode">
            {verdict === "FAKE" ? "AI Generated" : verdict === "REAL" ? "Authentic" : "Uncertain"}
          </span>
          <span className="det-conf" style={{ color: c.main }}>
            {item.confidence_score ? `${Math.round(item.confidence_score)}%` : "—"}
          </span>
        </div>
        <div className="det-date">{formatDate(item.created_at)}</div>
      </div>
    </div>
  );
};

// ── Stat Card ──────────────────────────────────────────────────────────────
const StatCard = ({ label, value, unit, sub, subColor }) => (
  <div className="stat-card">
    <div className="stat-label">{label}</div>
    <div className="stat-value">{value}<span>{unit}</span></div>
    <div className="stat-trend" style={{ color: subColor || "#56d364" }}>{sub}</div>
  </div>
);

// ── Ring Chart (fake/real/suspicious) ─────────────────────────────────────
const RingChart = ({ fakeRate, realRate }) => {
  const suspicious = Math.max(0, 100 - fakeRate - realRate);
  const r = 32;
  const circ = 2 * Math.PI * r;

  const segments = [
    { pct: fakeRate, color: "#f85149" },
    { pct: realRate, color: "#56d364" },
    { pct: suspicious, color: "#e3b341" },
  ];

  let offset = 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
      <svg width="80" height="80" viewBox="0 0 80 80">
        {segments.map(({ pct, color }, i) => {
          const dash = (pct / 100) * circ;
          const gap = circ - dash;
          const seg = (
            <circle key={i} cx="40" cy="40" r={r} fill="none" stroke={color}
              strokeWidth="7" strokeDasharray={`${dash} ${gap}`}
              strokeDashoffset={-offset} transform="rotate(-90 40 40)" />
          );
          offset += dash;
          return seg;
        })}
        <circle cx="40" cy="40" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="7" />
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
        {[
          ["Fake", Math.round(fakeRate) + "%", "#f85149"],
          ["Real", Math.round(realRate) + "%", "#56d364"],
          ["Suspicious", Math.round(suspicious) + "%", "#e3b341"],
        ].map(([label, val, color]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: color, flexShrink: 0 }} />
            <span style={{ fontSize: "10px", color: "#8b949e" }}>{label}</span>
            <span style={{ fontSize: "10px", color, marginLeft: "auto", fontWeight: 600 }}>{val}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Dashboard ──────────────────────────────────────────────────────────────
function Dashboard() {
  const navigate = useNavigate();
  const user = useSelector(selectUser);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loadingStats, setLoadingStats] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, historyRes] = await Promise.all([
          api.get("/detection/stats/").catch(() => null),
          api.get("/detection/").catch(() => null),
        ]);
        if (statsRes?.data) setStats(statsRes.data);
        if (historyRes?.data) setHistory(historyRes.data.slice(0, 6));
      } catch {
        // fallback: leave null
      } finally {
        setLoadingStats(false);
      }
    };
    fetchData();
  }, []);

  const fakeRate = stats?.fake_rate || 0;
  const realRate = stats ? (100 - fakeRate - Math.max(0, 100 - (stats.fake_count + stats.real_count) / stats.total_checked * 100)) : 0;

  return (
    <div className="dash-page">
      <div className="mb-7">
        <h1 className="dash-greeting">
          Welcome back, {user?.username ?? "there"} 👁
        </h1>
        <p className="dash-sub">Fake profile photo detection · 14-signal hybrid pipeline</p>
      </div>

      {/* Stats row */}
      {loadingStats ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "28px" }}>
          {[1, 2, 3].map(i => (
            <div key={i} className="stat-card" style={{ opacity: 0.4, minHeight: "80px" }} />
          ))}
        </div>
      ) : stats ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "28px" }}>
          <StatCard
            label="Total Analyzed"
            value={stats.total_checked}
            unit=" images"
            sub={`↑ ${stats.weekly_count} this week`}
          />
          <StatCard
            label="Avg Confidence"
            value={stats.avg_confidence || 0}
            unit="%"
            sub="across all detections"
            subColor="#79c0ff"
          />
          <div className="stat-card">
            <div className="stat-label">Detection Breakdown</div>
            <div style={{ marginTop: "6px" }}>
              <RingChart fakeRate={fakeRate} realRate={stats.real_count / (stats.total_checked || 1) * 100} />
            </div>
          </div>
        </div>
      ) : (
        <div className="stats-row">
          <StatCard label="Total Checked" value={0} unit=" images" sub="Analyze your first image" subColor="#484f58" />
          <StatCard label="Avg Confidence" value="—" unit="" sub="No data yet" subColor="#484f58" />
        </div>
      )}

      {/* Recent detections */}
      <div className="section-header">
        <span className="section-title">Recent Detections</span>
        <span className="view-all" onClick={() => navigate("/history")}>View all →</span>
      </div>

      <div className="cards-grid">
        {history.length === 0 ? (
          <div className="empty-state">
            No detections yet. Upload a profile photo to get started.
          </div>
        ) : (
          history.map((item) => <DetectionCard key={item.id} item={item} />)
        )}
      </div>

      {/* Analyze CTA */}
      <div className="flex justify-center mt-5">
        <button className="analyze-banner" onClick={() => navigate("/upload")}>
          <div className="analyze-icon">👁</div>
          <div className="analyze-text">
            <div className="analyze-title">Analyze a profile photo</div>
            <div className="analyze-sub">14-algorithm hybrid detection pipeline</div>
          </div>
          <div className="analyze-arrow">→</div>
        </button>
      </div>
    </div>
  );
}

export default Dashboard;
