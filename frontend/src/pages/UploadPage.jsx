import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { uploadDetectionImage } from "../api/uploadApi";
import "../styles/uploadPage.css";

// ── Utility: risk color ────────────────────────────────────────────────────
const riskColor = (verdict) => ({
  FAKE: { main: "#f85149", bg: "rgba(248,81,73,0.12)", border: "rgba(248,81,73,0.3)", glow: "rgba(248,81,73,0.15)" },
  REAL: { main: "#56d364", bg: "rgba(86,211,100,0.12)", border: "rgba(86,211,100,0.3)", glow: "rgba(86,211,100,0.15)" },
  SUSPICIOUS: { main: "#e3b341", bg: "rgba(227,179,65,0.12)", border: "rgba(227,179,65,0.3)", glow: "rgba(227,179,65,0.15)" },
}[verdict] || { main: "#8b949e", bg: "rgba(139,148,158,0.1)", border: "rgba(139,148,158,0.3)", glow: "rgba(139,148,158,0.1)" });

// ── Score Bar ──────────────────────────────────────────────────────────────
const ScoreBar = ({ label, score, isFakeness = true, description, weight }) => {
  const pct = Math.round(score * 100);
  const isBad = isFakeness ? score > 0.5 : score < 0.5;
  const color = isBad ? "#f85149" : "#56d364";
  const barPct = isFakeness ? pct : (100 - pct);
  return (
    <div className="score-bar-row">
      <div className="score-bar-header">
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className="score-bar-label">{label}</span>
          {weight && (
            <span style={{ fontSize: "9px", color: "#484f58", background: "rgba(255,255,255,0.04)", padding: "1px 5px", borderRadius: "4px" }}>
              {Math.round(weight * 100)}%
            </span>
          )}
        </div>
        <span className="score-bar-value" style={{ color }}>{pct}%</span>
      </div>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${barPct}%`, background: color }} />
      </div>
      {description && <span className="score-bar-desc">{description}</span>}
    </div>
  );
};

// ── Verdict Badge ──────────────────────────────────────────────────────────
const VerdictBadge = ({ verdict, riskLabel }) => {
  const icons = { FAKE: "⚠", REAL: "✓", SUSPICIOUS: "?" };
  const c = riskColor(verdict);
  return (
    <div className="verdict-badge-large" style={{ background: c.bg, border: `1px solid ${c.border}`, color: c.main }}>
      <span className="verdict-icon">{icons[verdict] || "?"}</span>
      <div>
        <div className="verdict-label">{verdict}</div>
        <div className="verdict-sublabel">{riskLabel || ""}</div>
      </div>
    </div>
  );
};

// ── Confidence Donut ───────────────────────────────────────────────────────
const ConfidenceDonut = ({ confidence, verdict }) => {
  const r = 38;
  const circ = 2 * Math.PI * r;
  const offset = circ - (confidence / 100) * circ;
  const color = riskColor(verdict).main;
  return (
    <div className="donut-wrapper">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
        <circle cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 50 50)" style={{ transition: "stroke-dashoffset 1.2s ease" }} />
      </svg>
      <div className="donut-center">
        <div className="donut-pct" style={{ color }}>{Math.round(confidence)}%</div>
        <div className="donut-label">confidence</div>
      </div>
    </div>
  );
};

// ── Profile Analysis Panel ─────────────────────────────────────────────────
const ProfileAnalysis = ({ profileAnalysis }) => {
  if (!profileAnalysis) return null;
  const { face_detected, face_count, single_face, face_quality_score,
    face_sharpness, symmetry_score, skin_uniformity, background_natural } = profileAnalysis;

  const items = [
    {
      label: "Face Detected",
      value: face_detected ? "Yes" : "No",
      good: face_detected,
      icon: "👤"
    },
    {
      label: "Face Count",
      value: face_count <= 0 ? "—" : face_count,
      good: face_count === 1,
      icon: face_count === 1 ? "✓" : face_count > 1 ? "⚠" : "—"
    },
    {
      label: "Face Quality",
      value: `${Math.round((face_quality_score || 0) * 100)}%`,
      good: (face_quality_score || 0) > 0.5,
      icon: "◈"
    },
    {
      label: "Sharpness",
      value: face_sharpness > 0 ? Math.round(face_sharpness) : "—",
      good: face_sharpness > 50,
      icon: "◎"
    },
    {
      label: "Symmetry",
      value: symmetry_score > 0 ? `${Math.round(symmetry_score * 100)}%` : "—",
      // AI faces are OVER-symmetric — so high symmetry is slightly bad
      good: symmetry_score > 0 && symmetry_score < 0.92,
      icon: "⊞"
    },
    {
      label: "Skin Tone",
      value: `${Math.round((1 - (skin_uniformity || 0.5)) * 100)}% natural`,
      good: (skin_uniformity || 0.5) < 0.5,
      icon: "◐"
    },
    {
      label: "Background",
      value: `${Math.round((background_natural || 0.5) * 100)}% natural`,
      good: (background_natural || 0.5) > 0.5,
      icon: "▣"
    },
    {
      label: "Single Face",
      value: single_face === null ? "—" : single_face ? "Yes" : "No",
      good: single_face === true,
      icon: single_face ? "✓" : "⚠"
    },
  ];

  return (
    <div style={{ marginBottom: "12px" }}>
      <div className="dsc-label" style={{ marginBottom: "8px" }}>Profile Photo Analysis</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
        {items.map((item) => (
          <div key={item.label} className="result-meta-item" style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span style={{ fontSize: "11px" }}>{item.icon}</span>
              <span className="result-meta-label">{item.label}</span>
            </div>
            <span className="result-meta-value" style={{ color: item.good ? "#56d364" : "#f85149", fontSize: "11px" }}>
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Analysis Breakdown (collapsible, grouped) ──────────────────────────────
const AnalysisSection = ({ analysis, weights, dlPrediction }) => {
  const [expanded, setExpanded] = useState(false);
  const [activeGroup, setActiveGroup] = useState("classic");

  if (!analysis) return null;

  const classicSignals = [
    { key: "frequency", label: "Frequency Analysis", desc: "GAN spectral fingerprints (DCT/FFT)", fakeness: true },
    { key: "texture_lbp", label: "LBP Skin Texture", desc: "Micro-texture richness", fakeness: true },
    { key: "color_stats", label: "Color Statistics", desc: "Color distribution naturalness", fakeness: true },
    { key: "pixel_diversity", label: "Pixel Diversity", desc: "K-Means cluster variance", fakeness: true },
    { key: "edge_sharpness", label: "Edge Sharpness", desc: "Sobel edge strength", fakeness: true },
    { key: "information_density", label: "Information Density", desc: "Shannon entropy", fakeness: true },
  ];

  const enhancedSignals = [
    { key: "ssim_texture", label: "SSIM Smoothness", desc: "Structural similarity patch analysis", fakeness: true },
    { key: "skin_tone_uniformity", label: "Skin Tone HSV", desc: "Hue distribution uniformity", fakeness: true },
    { key: "sharpness_profile", label: "Sharpness Profile", desc: "Multi-scale Laplacian analysis", fakeness: true },
    { key: "lens_physics", label: "Lens Physics", desc: "Chromatic aberration (real lens)", fakeness: true },
    { key: "background_analysis", label: "Background", desc: "Face/background coherence", fakeness: true },
    { key: "sensor_noise", label: "Sensor Noise", desc: "Camera noise pattern", fakeness: true },
  ];

  const groups = { classic: classicSignals, enhanced: enhancedSignals };

  return (
    <div className="analysis-section">
      <button className="analysis-toggle" onClick={() => setExpanded(v => !v)}>
        <span>Signal Breakdown ({Object.keys(analysis).length} algorithms)</span>
        <span style={{ fontSize: "10px", color: "#484f58" }}>{expanded ? "▲ hide" : "▼ show"}</span>
      </button>

      {expanded && (
        <div className="analysis-grid">
          {/* DL pill */}
          <div className="dl-pill">
            <span className="dl-pill-label">ResNet18 Deep Learning ({Math.round((weights?.deep_learning || 0.38) * 100)}% weight)</span>
            <span className="dl-pill-value" style={{ color: dlPrediction?.label === "Fake" ? "#f85149" : "#56d364" }}>
              {dlPrediction?.label} · {dlPrediction?.confidence}%
            </span>
          </div>

          {/* Group tabs */}
          <div style={{ display: "flex", gap: "6px" }}>
            {[["classic", "Classic (6)"], ["enhanced", "Enhanced (6)"]].map(([k, label]) => (
              <button key={k} onClick={() => setActiveGroup(k)}
                style={{
                  flex: 1, padding: "5px", borderRadius: "8px", border: "none", cursor: "pointer",
                  fontSize: "10px", fontFamily: "'DM Sans', sans-serif",
                  background: activeGroup === k ? "rgba(99,102,241,0.2)" : "rgba(255,255,255,0.03)",
                  color: activeGroup === k ? "#818cf8" : "#484f58",
                  transition: "all 0.2s"
                }}>
                {label}
              </button>
            ))}
          </div>

          {groups[activeGroup].map(({ key, label, desc, fakeness }) => {
            const val = analysis[key]?.score ?? 0;
            const w = analysis[key]?.weight;
            return (
              <ScoreBar key={key} label={label} score={val} isFakeness={fakeness} description={desc} weight={w} />
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Result Panel ───────────────────────────────────────────────────────────
const ResultPanel = ({ result, onReset }) => {
  if (!result) return (
    <div className="upload-panel">
      <span className="panel-label">Detection Result</span>
      <div className="result-empty">
        <div className="result-empty-icon">👁</div>
        <p className="result-empty-text">
          Upload a profile photo to run DrishtiAI's 14-algorithm detection pipeline
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", justifyContent: "center", marginTop: "8px" }}>
          {["ResNet18", "DCT Analysis", "LBP Texture", "SSIM", "HSV Skin", "Lens Physics"].map(t => (
            <span key={t} style={{ fontSize: "9px", padding: "2px 8px", borderRadius: "20px", background: "rgba(99,102,241,0.08)", border: "0.5px solid rgba(99,102,241,0.2)", color: "#818cf8" }}>{t}</span>
          ))}
        </div>
      </div>
    </div>
  );

  const verdict = result.verdict || result.final_label || "SUSPICIOUS";
  const confidence = result.confidence || 0;
  const heatmap = result.heatmap;
  const c = riskColor(verdict);

  return (
    <div className="upload-panel" style={{ position: "relative" }}>
      {/* Verdict glow */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: "3px",
        borderRadius: "20px 20px 0 0",
        background: `linear-gradient(90deg, ${c.main}, transparent)`
      }} />

      <span className="panel-label">Detection Result</span>

      {/* Heatmap */}
      {heatmap && (
        <div className="heatmap-wrap">
          <img src={heatmap} alt="Grad-CAM heatmap" className="heatmap-img" />
          <div className="heatmap-badge">Grad-CAM · Focus Map</div>
        </div>
      )}

      {/* Verdict + donut */}
      <div className="verdict-conf-row">
        <VerdictBadge verdict={verdict} riskLabel={result.risk_label} />
        <ConfidenceDonut confidence={confidence} verdict={verdict} />
      </div>

      {/* Decision score bar */}
      <div className="decision-score-row">
        <span className="dsc-label">Hybrid Decision Score (14 signals)</span>
        <div className="dsc-bar-wrap">
          <div className="dsc-bar-track">
            <div className="dsc-bar-fill" style={{
              width: `${Math.round((result.decision_score || 0) * 100)}%`,
              background: verdict === "FAKE"
                ? "linear-gradient(90deg, #f85149, #ff7875)"
                : verdict === "REAL"
                  ? "linear-gradient(90deg, #56d364, #85ef8a)"
                  : "linear-gradient(90deg, #e3b341, #f0d070)",
            }} />
            <div className="dsc-threshold" style={{ left: `${Math.round(36)}%` }} title="REAL threshold" />
            <div className="dsc-threshold" style={{ left: `${Math.round(60)}%` }} title="FAKE threshold" />
          </div>
          <div className="dsc-labels">
            <span style={{ color: "#56d364" }}>REAL</span>
            <span style={{ color: "#e3b341" }}>SUSPICIOUS</span>
            <span style={{ color: "#f85149" }}>FAKE</span>
          </div>
        </div>
      </div>

      {/* Profile photo analysis */}
      <ProfileAnalysis profileAnalysis={result.profile_analysis} />

      {/* Meta grid */}
      <div className="result-meta-grid">
        <div className="result-meta-item">
          <span className="result-meta-label">Processing</span>
          <span className="result-meta-value">{result.processing_time || "—"}s</span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">Pipeline</span>
          <span className="result-meta-value">{result.pipeline_version || "v3"} · 14 signals</span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">DL Verdict</span>
          <span className="result-meta-value" style={{ color: result.dl_prediction?.label === "Fake" ? "#f85149" : "#56d364" }}>
            {result.dl_prediction?.label || "—"}
          </span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">DL Confidence</span>
          <span className="result-meta-value">{result.dl_prediction?.confidence || "—"}%</span>
        </div>
      </div>

      {/* Algorithm breakdown */}
      <AnalysisSection
        analysis={result.analysis}
        weights={result.weights}
        dlPrediction={result.dl_prediction}
      />

      <button className="analyze-again-btn" onClick={onReset}>
        ↩ Analyze another image
      </button>
    </div>
  );
};

// ── Loading Messages ───────────────────────────────────────────────────────
const LOADING_STEPS = [
  { msg: "Detecting face region…", pct: 10 },
  { msg: "Running ResNet18 model…", pct: 25 },
  { msg: "DCT frequency analysis…", pct: 38 },
  { msg: "LBP texture + color stats…", pct: 50 },
  { msg: "SSIM patch analysis…", pct: 62 },
  { msg: "HSV skin uniformity…", pct: 72 },
  { msg: "Chromatic aberration check…", pct: 82 },
  { msg: "Noise pattern analysis…", pct: 90 },
  { msg: "Computing hybrid score…", pct: 96 },
];

// ── Main Upload Page ───────────────────────────────────────────────────────
const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [loadingStep, setLoadingStep] = useState(0);

  const startLoadingCycle = () => {
    let i = 0;
    return setInterval(() => {
      i = Math.min(i + 1, LOADING_STEPS.length - 1);
      setLoadingStep(i);
    }, 1400);
  };

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length > 0) {
      setError("Only JPG, PNG, or WEBP images accepted (max 15MB).");
      return;
    }
    const f = accepted[0];
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [], "image/png": [], "image/webp": [] },
    maxFiles: 1,
    maxSize: 15 * 1024 * 1024,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setLoadingStep(0);
    const interval = startLoadingCycle();
    try {
      const res = await uploadDetectionImage(file);
      setResult(res.data);
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.detail || "Analysis failed. Please try a clear face photo.";
      setError(msg);
    } finally {
      clearInterval(interval);
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  const step = LOADING_STEPS[loadingStep];

  return (
    <div className="upload-page">
      <div className="mb-7">
        <h1 className="dash-greeting">Fake Profile Detector</h1>
        <p className="dash-sub">
          14-algorithm hybrid pipeline · ResNet18 + 13 classical & prebuilt signals
        </p>
      </div>

      {/* Algorithm badges */}
      <div className="mode-info-bar">
        {[
          { label: "ResNet18 DL", color: "#6366f1" },
          { label: "DCT Frequency", color: "#06b6d4" },
          { label: "LBP + Color", color: "#8b5cf6" },
          { label: "SSIM Texture", color: "#10b981" },
          { label: "HSV Skin", color: "#f59e0b" },
          { label: "Lens Physics", color: "#ef4444" },
          { label: "Noise Pattern", color: "#84cc16" },
        ].map(({ label, color }) => (
          <div key={label} className="mode-info-item">
            <span className="mode-info-dot" style={{ background: color }} />
            {label}
          </div>
        ))}
      </div>

      <div className="upload-grid">
        {/* Upload panel */}
        <div className="upload-panel">
          <span className="panel-label">Upload Profile Photo</span>

          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "dragging" : ""} ${file ? "has-file" : ""}`}
          >
            <input {...getInputProps()} />
            {file && previewUrl ? (
              <>
                <img src={previewUrl} alt="preview" className="w-full h-60 object-cover rounded-xl" />
                <p className="drop-filename">{file.name}</p>
                <p className="drop-change">Click or drop to replace</p>
              </>
            ) : (
              <>
                <div className="drop-icon">👤</div>
                <p className="drop-title">{isDragActive ? "Drop it here!" : "Drop a profile photo"}</p>
                <p className="drop-sub">or</p>
                <p className="drop-browse">Browse files</p>
                <p className="drop-hint">JPG · PNG · WEBP · Max 15MB</p>
              </>
            )}
          </div>

          {error && <p className="upload-error">⚠ {error}</p>}

          {/* Loading progress bar */}
          {isLoading && (
            <div style={{ marginTop: "4px" }}>
              <div style={{ height: "2px", background: "rgba(255,255,255,0.06)", borderRadius: "2px", overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${step.pct}%`,
                  background: "linear-gradient(90deg, #6366f1, #818cf8)",
                  transition: "width 1.2s ease",
                  borderRadius: "2px"
                }} />
              </div>
              <p style={{ fontSize: "10px", color: "#484f58", marginTop: "4px", textAlign: "center" }}>
                {step.msg}
              </p>
            </div>
          )}

          <div className="upload-tips">
            <p className="tips-title">Tips for best results:</p>
            <ul className="tips-list">
              <li>Use front-facing profile photos (head &amp; shoulders)</li>
              <li>Single person, clearly visible face</li>
              <li>Good lighting, not blurry or cropped too tight</li>
              <li>Avoid sunglasses, heavy filters, or extreme angles</li>
            </ul>
          </div>

          <button className="analyze-btn" onClick={handleAnalyze} disabled={!file || isLoading}>
            {isLoading ? (
              <span className="loading-row">
                <span className="loading-spinner" />
                {step.msg}
              </span>
            ) : (
              "▶ Analyze Profile Photo"
            )}
          </button>
        </div>

        {/* Result panel */}
        <ResultPanel result={result} onReset={handleReset} />
      </div>
    </div>
  );
};

export default UploadPage;
