import { useEffect, useState } from "react";
import { api } from "../api/axiosInstance";

const verdictColors = {
  FAKE: { main: "#f85149", bg: "rgba(248,81,73,0.08)", badge: "rgba(248,81,73,0.15)", border: "rgba(248,81,73,0.25)" },
  REAL: { main: "#56d364", bg: "rgba(86,211,100,0.08)", badge: "rgba(86,211,100,0.15)", border: "rgba(86,211,100,0.25)" },
  SUSPICIOUS: { main: "#e3b341", bg: "rgba(227,179,65,0.08)", badge: "rgba(227,179,65,0.15)", border: "rgba(227,179,65,0.25)" },
};

const formatDate = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });
};

// ── History Row ────────────────────────────────────────────────────────────
const HistoryRow = ({ item }) => {
  const verdict = item.verdict || (item.is_fake ? "FAKE" : "REAL");
  const c = verdictColors[verdict] || verdictColors.SUSPICIOUS;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "12px",
      padding: "12px 16px", borderRadius: "12px",
      background: "rgba(13,17,23,0.95)", border: `0.5px solid ${c.border}`,
      marginBottom: "8px", transition: "all 0.2s"
    }}>
      {/* Image thumb */}
      <div style={{
        width: "48px", height: "48px", borderRadius: "10px", overflow: "hidden",
        flexShrink: 0, background: c.bg, display: "flex", alignItems: "center", justifyContent: "center"
      }}>
        {item.image ? (
          <img src={item.image} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <span style={{ fontSize: "20px", opacity: 0.4 }}>👤</span>
        )}
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: "12px", fontWeight: 500, color: "#e6edf3", marginBottom: "2px" }}>
          Detection #{item.id}
        </div>
        <div style={{ fontSize: "10px", color: "#484f58" }}>{formatDate(item.created_at)}</div>
      </div>

      {/* Verdict badge */}
      <div style={{
        padding: "3px 10px", borderRadius: "8px", fontSize: "10px", fontWeight: 600,
        letterSpacing: "0.05em", background: c.badge, color: c.main, border: `0.5px solid ${c.border}`
      }}>
        {verdict}
      </div>

      {/* Confidence */}
      <div style={{ textAlign: "right", minWidth: "52px" }}>
        <div style={{ fontSize: "14px", fontWeight: 700, color: c.main }}>
          {item.confidence_score ? `${Math.round(item.confidence_score)}%` : "—"}
        </div>
        <div style={{ fontSize: "9px", color: "#484f58" }}>confidence</div>
      </div>
    </div>
  );
};

// ── Filter Tabs ────────────────────────────────────────────────────────────
const FilterTab = ({ label, active, count, color, onClick }) => (
  <button onClick={onClick} style={{
    padding: "6px 14px", borderRadius: "10px", border: "none", cursor: "pointer",
    fontSize: "11px", fontWeight: 500,
    background: active ? "rgba(99,102,241,0.15)" : "rgba(255,255,255,0.03)",
    color: active ? "#818cf8" : "#484f58",
    transition: "all 0.2s"
  }}>
    {label}
    {count !== undefined && (
      <span style={{
        marginLeft: "5px", padding: "1px 5px", borderRadius: "5px",
        background: active ? "rgba(99,102,241,0.2)" : "rgba(255,255,255,0.05)",
        color: active ? "#818cf8" : "#484f58", fontSize: "9px"
      }}>
        {count}
      </span>
    )}
  </button>
);

// ── History Page ───────────────────────────────────────────────────────────
function HistoryPage() {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("ALL");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await api.get("/detection/");
        setDetections(res.data || []);
      } catch (err) {
        setError("Failed to load detection history.");
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const verdictCounts = detections.reduce((acc, d) => {
    const v = d.verdict || (d.is_fake ? "FAKE" : "REAL");
    acc[v] = (acc[v] || 0) + 1;
    return acc;
  }, {});

  const filtered = filter === "ALL"
    ? detections
    : detections.filter(d => (d.verdict || (d.is_fake ? "FAKE" : "REAL")) === filter);

  return (
    <div className="dash-page">
      <div className="mb-7">
        <h1 className="dash-greeting">Detection History</h1>
        <p className="dash-sub">{detections.length} total analyses · profile photo detection pipeline</p>
      </div>

      {/* Filter tabs */}
      <div style={{ display: "flex", gap: "6px", marginBottom: "20px", flexWrap: "wrap" }}>
        <FilterTab label="All" active={filter === "ALL"} count={detections.length} onClick={() => setFilter("ALL")} />
        <FilterTab label="Fake" active={filter === "FAKE"} count={verdictCounts.FAKE || 0} onClick={() => setFilter("FAKE")} />
        <FilterTab label="Real" active={filter === "REAL"} count={verdictCounts.REAL || 0} onClick={() => setFilter("REAL")} />
        <FilterTab label="Suspicious" active={filter === "SUSPICIOUS"} count={verdictCounts.SUSPICIOUS || 0} onClick={() => setFilter("SUSPICIOUS")} />
      </div>

      {/* Content */}
      {loading ? (
        <div style={{ textAlign: "center", padding: "40px", color: "#484f58", fontSize: "13px" }}>
          Loading history…
        </div>
      ) : error ? (
        <div style={{
          padding: "16px", borderRadius: "12px", fontSize: "12px",
          background: "rgba(248,81,73,0.08)", border: "0.5px solid rgba(248,81,73,0.2)", color: "#f85149"
        }}>
          {error}
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state" style={{ gridColumn: "1/-1" }}>
          {filter === "ALL" ? "No detections yet. Upload a profile photo to get started." : `No ${filter.toLowerCase()} detections found.`}
        </div>
      ) : (
        <div>
          {filtered.map(item => <HistoryRow key={item.id} item={item} />)}
        </div>
      )}
    </div>
  );
}

export default HistoryPage;
