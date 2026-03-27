import { useNavigate } from "react-router-dom";
import "../styles/detection.css";
import { useSelector } from "react-redux";
import { selectUser } from "../features/auth/authSlice";

const dummyStats = {
  totalChecked: 24,
  avgConfidence: 87,
  weeklyCount: 6,
};

const dummyDetections = [
  {
    id: 1,
    filename: "dune_jessica2.png",
    verdict: "REAL",
    confidence: 97.3,
    mode: "Deepfake",
    date: "Mar 27, 2026  · 10:30 AM",
  },
  {
    id: 2,
    filename: "me.jpg",
    verdict: "FAKE",
    confidence: 93.3,
    mode: "AI Generated",
    date: "Mar 27, 2026  · 10:11 AM",
  },
  {
    id: 3,
    filename: "tanish_ai.png",
    verdict: "FAKE",
    confidence: 91.2,
    mode: "Deepfake",
    date: "Mar 18, 2026  · 2:42 PM",
  },
];

const DetectionCard = ({ item }) => {
  const isFake = item.verdict === "FAKE";

  return (
    <div className="det-card">
      <div className={`det-img ${isFake ? "fake-bg" : "real-bg"}`}>
        <img
          src={`/images/${item.filename}`}
          alt={item.filename}
          className="det-img-tag"
        />
        <span className={`det-badge ${isFake ? "fake" : "real"}`}>
          {item.verdict}
        </span>
      </div>
      <div className="det-body">
        <div className="det-filename">{item.filename}</div>
        <div className="det-meta">
          <span className="det-mode">{item.mode}</span>
          <span className={`det-conf ${isFake ? "fake" : "real"}`}>
            {item.confidence}%
          </span>
        </div>
        <div className="det-date">{item.date}</div>
      </div>
    </div>
  );
};

function Dashboard() {
  const navigate = useNavigate();
  const user = useSelector(selectUser);

  const stats = dummyStats;
  const recentDetections = dummyDetections;

  return (
    <div className="dash-page">
      <div className="mb-7">
        <h1 className="dash-greeting">
          Welcome back, {user?.name ?? "there"} 👁
        </h1>
        <p className="dash-sub">Here's what's been detected recently</p>
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Total Checked</div>
          <div className="stat-value">
            {stats.totalChecked}
            <span>images</span>
          </div>
          <div className="stat-trend">↑ {stats.weeklyCount} this week</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value">
            {stats.avgConfidence}
            <span>%</span>
          </div>
          <div className="stat-trend neutral">across all detections</div>
        </div>
      </div>

      <div className="section-header">
        <span className="section-title">Recent detections</span>
        <span className="view-all" onClick={() => navigate("/history")}>
          View all →
        </span>
      </div>

      <div className="cards-grid">
        {recentDetections.length === 0 ? (
          <div className="empty-state">
            No detections yet. Analyze your first image below.
          </div>
        ) : (
          recentDetections.map((item) => (
            <DetectionCard key={item.id} item={item} />
          ))
        )}
      </div>

      <div className="flex justify-center mt-5">
        <button
          className="analyze-banner"
          onClick={() => navigate("/upload")}
        >
          <div className="analyze-icon">👁</div>
          <div className="analyze-text">
            <div className="analyze-title">Analyze a new image</div>
            <div className="analyze-sub">
              Detect deepfake or AI generated faces
            </div>
          </div>
          <div className="analyze-arrow">→</div>
        </button>
      </div>
    </div>
  );
}

export default Dashboard;
