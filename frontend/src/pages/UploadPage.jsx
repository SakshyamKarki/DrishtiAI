import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { uploadDetectionImage } from "../api/uploadApi";
import "../styles/uploadPage.css";

// ── Score Bar Component ──────────────────────────────────────────────────────
const ScoreBar = ({ label, score, fakeness = true, description }) => {
  const pct     = Math.round(score * 100);
  const isBad   = fakeness ? score > 0.5 : score < 0.5;
  const color   = isBad ? "#f85149" : "#56d364";
  const barPct  = fakeness ? pct : (100 - pct);

  return (
    <div className="score-bar-row">
      <div className="score-bar-header">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-value" style={{ color }}>
          {pct}%
        </span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${barPct}%`, background: color }}
        />
      </div>
      {description && (
        <span className="score-bar-desc">{description}</span>
      )}
    </div>
  );
};

// ── Verdict Badge ────────────────────────────────────────────────────────────
const VerdictBadge = ({ verdict }) => {
  const config = {
    FAKE:       { color: "#f85149", bg: "rgba(248,81,73,0.12)",   border: "rgba(248,81,73,0.3)",  icon: "⚠", label: "AI Generated / Fake" },
    REAL:       { color: "#56d364", bg: "rgba(86,211,100,0.12)",  border: "rgba(86,211,100,0.3)", icon: "✓", label: "Authentic / Real" },
    SUSPICIOUS: { color: "#e3b341", bg: "rgba(227,179,65,0.12)",  border: "rgba(227,179,65,0.3)", icon: "?", label: "Suspicious / Uncertain" },
  };
  const c = config[verdict] || config.SUSPICIOUS;

  return (
    <div className="verdict-badge-large" style={{
      background: c.bg, border: `1px solid ${c.border}`, color: c.color
    }}>
      <span className="verdict-icon">{c.icon}</span>
      <div>
        <div className="verdict-label">{verdict}</div>
        <div className="verdict-sublabel">{c.label}</div>
      </div>
    </div>
  );
};

// ── Donut Confidence Chart ────────────────────────────────────────────────────
const ConfidenceDonut = ({ confidence, verdict }) => {
  const r      = 38;
  const circ   = 2 * Math.PI * r;
  const offset = circ - (confidence / 100) * circ;
  const color  = verdict === "FAKE" ? "#f85149" : verdict === "REAL" ? "#56d364" : "#e3b341";

  return (
    <div className="donut-wrapper">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
        <circle
          cx="50" cy="50" r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
      </svg>
      <div className="donut-center">
        <div className="donut-pct" style={{ color }}>{Math.round(confidence)}%</div>
        <div className="donut-label">confidence</div>
      </div>
    </div>
  );
};

// ── Analysis Section ─────────────────────────────────────────────────────────
const AnalysisSection = ({ analysis, dlPrediction }) => {
  const [expanded, setExpanded] = useState(false);

  if (!analysis) return null;

  const signals = [
    { key: "frequency",          label: "Frequency Analysis",    desc: "GAN spectral fingerprints in DCT domain",  fakeness: true },
    { key: "texture_lbp",        label: "LBP Texture",           desc: "Skin micro-texture richness",              fakeness: true },
    { key: "color_stats",        label: "Color Statistics",      desc: "Color distribution naturalness",           fakeness: true },
    { key: "pixel_diversity",    label: "Pixel Diversity",       desc: "K-Means cluster variance",                 fakeness: true },
    { key: "edge_sharpness",     label: "Edge Sharpness",        desc: "Sobel edge strength analysis",             fakeness: true },
    { key: "information_density",label: "Information Density",   desc: "Shannon entropy complexity",               fakeness: true },
  ];

  return (
    <div className="analysis-section">
      <button
        className="analysis-toggle"
        onClick={() => setExpanded(v => !v)}
      >
        <span>Algorithm Analysis Breakdown</span>
        <span style={{ fontSize: "10px", color: "#484f58" }}>
          {expanded ? "▲ hide" : "▼ show"}
        </span>
      </button>

      {expanded && (
        <div className="analysis-grid">
          {/* DL prediction pill */}
          <div className="dl-pill">
            <span className="dl-pill-label">ResNet18 (40% weight)</span>
            <span className="dl-pill-value" style={{
              color: dlPrediction?.label === "Fake" ? "#f85149" : "#56d364"
            }}>
              {dlPrediction?.label} · {dlPrediction?.confidence}%
            </span>
          </div>

          {signals.map(({ key, label, desc }) => {
            const val = analysis[key]?.score ?? 0;
            return (
              <ScoreBar
                key={key}
                label={label}
                score={val}
                fakeness
                description={desc}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Face Detection Meta ───────────────────────────────────────────────────────
const FaceMeta = ({ faceMeta }) => {
  if (!faceMeta || !faceMeta.found) return null;
  return (
    <div className="face-meta">
      <span className="face-meta-item">
        🎯 {faceMeta.cascade_used || "cascade"} detection
      </span>
      {faceMeta.sharpness > 0 && (
        <span className="face-meta-item">
          ◈ sharpness: {Math.round(faceMeta.sharpness)}
        </span>
      )}
      {faceMeta.symmetry_score > 0 && (
        <span className="face-meta-item">
          ⊞ symmetry: {Math.round(faceMeta.symmetry_score * 100)}%
        </span>
      )}
    </div>
  );
};

// ── Result Panel ─────────────────────────────────────────────────────────────
const ResultPanel = ({ result, onReset }) => {
  if (!result) return (
    <div className="upload-panel">
      <span className="panel-label">Detection Result</span>
      <div className="result-empty">
        <div className="result-empty-icon">👁</div>
        <p className="result-empty-text">
          Upload a face image and click Analyze to see the full breakdown
        </p>
      </div>
    </div>
  );

  const verdict    = result.final_label || "SUSPICIOUS";
  const confidence = result.confidence  || 0;
  const heatmap    = result.heatmap;

  return (
    <div className="upload-panel">
      <span className="panel-label">Detection Result</span>

      {/* Heatmap / original image */}
      {heatmap && (
        <div className="heatmap-wrap">
          <img src={heatmap} alt="Grad-CAM heatmap" className="heatmap-img" />
          <div className="heatmap-badge">Grad-CAM Activation Map</div>
        </div>
      )}

      {/* Verdict + confidence */}
      <div className="verdict-conf-row">
        <VerdictBadge verdict={verdict} />
        <ConfidenceDonut confidence={confidence} verdict={verdict} />
      </div>

      {/* Decision score */}
      <div className="decision-score-row">
        <span className="dsc-label">Hybrid Decision Score</span>
        <div className="dsc-bar-wrap">
          <div className="dsc-bar-track">
            <div
              className="dsc-bar-fill"
              style={{
                width: `${Math.round((result.decision_score || 0) * 100)}%`,
                background: verdict === "FAKE"
                  ? "linear-gradient(90deg, #f85149, #ff7875)"
                  : verdict === "REAL"
                  ? "linear-gradient(90deg, #56d364, #85ef8a)"
                  : "linear-gradient(90deg, #e3b341, #f0d070)",
              }}
            />
            {/* Threshold lines */}
            <div className="dsc-threshold" style={{ left: "38%" }} title="Real threshold" />
            <div className="dsc-threshold" style={{ left: "62%" }} title="Fake threshold" />
          </div>
          <div className="dsc-labels">
            <span style={{ color: "#56d364" }}>REAL</span>
            <span style={{ color: "#e3b341" }}>SUSPICIOUS</span>
            <span style={{ color: "#f85149" }}>FAKE</span>
          </div>
        </div>
      </div>

      {/* Processing meta */}
      <div className="result-meta-grid">
        <div className="result-meta-item">
          <span className="result-meta-label">Processing time</span>
          <span className="result-meta-value">{result.processing_time || "—"}s</span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">Model</span>
          <span className="result-meta-value">{result.model || "ResNet18"}</span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">DL verdict</span>
          <span className="result-meta-value"
            style={{ color: result.dl_prediction?.label === "Fake" ? "#f85149" : "#56d364" }}>
            {result.dl_prediction?.label || "—"}
          </span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">DL confidence</span>
          <span className="result-meta-value">{result.dl_prediction?.confidence || "—"}%</span>
        </div>
      </div>

      {/* Face detection meta */}
      <FaceMeta faceMeta={result.detailed_analysis?.face_meta} />

      {/* Collapsible analysis breakdown */}
      <AnalysisSection
        analysis={result.analysis}
        dlPrediction={result.dl_prediction}
      />

      <button className="analyze-again-btn" onClick={onReset}>
        ↩ Analyze another image
      </button>
    </div>
  );
};

// ── Main Upload Page ──────────────────────────────────────────────────────────
const UploadPage = () => {
  const [file,        setFile]        = useState(null);
  const [previewUrl,  setPreviewUrl]  = useState(null);
  const [isLoading,   setIsLoading]   = useState(false);
  const [error,       setError]       = useState(null);
  const [result,      setResult]      = useState(null);
  const [loadingMsg,  setLoadingMsg]  = useState("Analyzing...");

  // Cycle through loading messages to show progress
  const startLoadingCycle = () => {
    const msgs = [
      "Detecting face...",
      "Running ResNet18 model...",
      "Frequency domain analysis...",
      "LBP texture analysis...",
      "Color statistics...",
      "Computing hybrid score...",
    ];
    let i = 0;
    return setInterval(() => {
      setLoadingMsg(msgs[i % msgs.length]);
      i++;
    }, 1800);
  };

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length > 0) {
      setError("Only JPG, PNG, or WEBP images accepted (max 10MB).");
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
    maxSize: 10 * 1024 * 1024,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    const interval = startLoadingCycle();
    try {
      const res = await uploadDetectionImage(file);
      setResult(res.data);
    } catch (err) {
      const msg = err.response?.data?.error || "Analysis failed. Please try again.";
      setError(msg);
    } finally {
      clearInterval(interval);
      setIsLoading(false);
      setLoadingMsg("Analyzing...");
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  return (
    <div className="upload-page">
      <div className="mb-7">
        <h1 className="dash-greeting">Fake Profile Detector</h1>
        <p className="dash-sub">
          Upload a face image — DrishtiAI runs 7 detection algorithms to verify authenticity
        </p>
      </div>

      {/* Detection mode info bar */}
      <div className="mode-info-bar">
        <div className="mode-info-item">
          <span className="mode-info-dot" style={{ background: "#6366f1" }} />
          ResNet18 Deep Learning
        </div>
        <div className="mode-info-item">
          <span className="mode-info-dot" style={{ background: "#06b6d4" }} />
          DCT Frequency Analysis
        </div>
        <div className="mode-info-item">
          <span className="mode-info-dot" style={{ background: "#8b5cf6" }} />
          LBP Texture + Color Stats
        </div>
        <div className="mode-info-item">
          <span className="mode-info-dot" style={{ background: "#10b981" }} />
          K-Means · Edge · Entropy
        </div>
      </div>

      <div className="upload-grid">
        {/* Upload panel */}
        <div className="upload-panel">
          <span className="panel-label">Upload Image</span>

          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "dragging" : ""} ${file ? "has-file" : ""}`}
          >
            <input {...getInputProps()} />
            {file && previewUrl ? (
              <>
                <img src={previewUrl} alt="preview"
                  className="w-full h-60 object-cover rounded-xl" />
                <p className="drop-filename">{file.name}</p>
                <p className="drop-change">Click or drop to replace</p>
              </>
            ) : (
              <>
                <div className="drop-icon">🖼</div>
                <p className="drop-title">
                  {isDragActive ? "Drop it here!" : "Drag & drop a face image"}
                </p>
                <p className="drop-sub">or</p>
                <p className="drop-browse">Browse files</p>
                <p className="drop-hint">JPG · PNG · WEBP · Max 10MB</p>
              </>
            )}
          </div>

          {error && <p className="upload-error">⚠ {error}</p>}

          {/* Tips */}
          <div className="upload-tips">
            <p className="tips-title">For best results:</p>
            <ul className="tips-list">
              <li>Use front-facing face images</li>
              <li>Avoid extreme crops or rotations</li>
              <li>Works best on profile-photo style images</li>
            </ul>
          </div>

          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!file || isLoading}
          >
            {isLoading ? (
              <span className="loading-row">
                <span className="loading-spinner" />
                {loadingMsg}
              </span>
            ) : (
              "▶ Analyze Image"
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
