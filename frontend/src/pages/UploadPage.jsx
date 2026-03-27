import { useCallback, useState } from "react";
import "../styles/uploadPage.css";
import { useDropzone } from "react-dropzone";

const simulateAnalysis = () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        id: 1,
        verdict: "FAKE",
        confidence: 94.2,
        mode: "ai_generated",
        model_used: "ResNet18",
        face_detected: true,
        processing_time: 1.3,
        analyzed_at: "2026-03-22T10:30:00Z",
      });
    }, 2000);
  });
};

const ResultPanel = ({ result, previewUrl, onReset }) => {
  const isFake = result.verdict === "FAKE";

  return (
    <div className="upload-panel">
      <span className="panel-label">Detection Result</span>
      <div
        className="result-preview"
        style={{
          borderColor: isFake ? "rgba(248,81,73,0.2)" : "rgba(86,211,100,0.2)",
          background: isFake ? "rgba(248,81,73,0.04)" : "rgba(86,211,100,0.04)",
        }}
      >
        <img
          src={previewUrl}
          alt="analyzed"
          className="w-full h-full object-cover rounded-xl"
        />
      </div>
      <div className="verdict-row">
        <div>
          <div className="result-meta-label">Verdict</div>
          <div
            className="verdict-value"
            style={{ color: isFake ? "#f85149" : "#56d364" }}
          >
            {result.verdict}
          </div>
        </div>
        <div
          className="verdict-badge"
          style={{
            background: isFake
              ? "rgba(248,81,73,0.12)"
              : "rgba(86,211,100,0.12)",
            color: isFake ? "#f85149" : "#56d364",
            border: `0.5px solid ${isFake ? "rgba(248,81,73,0.3)" : "rgba(86,211,100,0.3)"}`,
          }}
        >
          {isFake ? "AI Generated" : "Authentic"}
        </div>
      </div>

      <div>
        <div className="conf-label-row">
          <span className="result-meta-label">Confidence</span>
          <span
            className="conf-value"
            style={{ color: isFake ? "#f85149" : "#56d364" }}
          >
            {result.confidence}%
          </span>
        </div>
        <div className="conf-bar-bg">
          <div
            className="conf-bar-fill"
            style={{
              width: `${result.confidence}%`,
              background: isFake
                ? "linear-gradient(90deg, #f85149, #ff7875)"
                : "linear-gradient(90deg, #56d364, #85ef8a)",
            }}
          />
        </div>
      </div>
      <div className="result-meta-grid">
        <div className="result-meta-item">
          <span className="result-meta-label">Processing time</span>
          <span className="result-meta-value">{result.processing_time}s</span>
        </div>
        <div className="result-meta-item">
          <span className="result-meta-label">Model used</span>
          <span className="result-meta-value">{result.model_used}</span>
        </div>
      </div>
      <button className="analyze-again-btn" onClick={onReset}>
        Analyze another image
      </button>
    </div>
  );
};

const EmptyResult = () => (
  <div className="upload-panel">
    <span className="panel-label">Detection result</span>
    <div className="result-empty">
      <div className="result-empty-icon">👁</div>
      <p className="result-empty-text">
        Upload an image and hit analyze to see results here
      </p>
    </div>
  </div>
);

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setError("Only JPG, PNG or WEBP images are accepted.");
      return;
    }
    const selected = acceptedFiles[0];
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [], "image/png": [], "image/webp": [] },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    try {
      //hit api
      const data = await simulateAnalysis();
      setResult(data);
    } catch (err) {
      setError("Analysis failed. Try again");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);

    if (previewUrl) URL.revokeObjectURL(previewUrl);
  };

  return (
    <div className="upload-page">
      <div className="mb-7">
        <h1 className="dash-greeting">Detect fake image</h1>
        <p className="dash-sub">
          Upload a face image to analyze if it is AI generated
        </p>
      </div>

      <div className="upload-grid">
        <div className="upload-panel">
          <span className="span-label">Upload Image</span>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? "dragging" : ""} ${file ? "has-file" : ""}`}
          >
            <input {...getInputProps()} />

            {file && previewUrl ? (
              <>
                <img
                  src={previewUrl}
                  alt="preview"
                  className="w-full h-36 object-cover rounded-xl"
                />
                <p className="drop-filename">{file.name}</p>
                <p className="drop-change">Click or drop to change</p>
              </>
            ) : (
              <>
                <div className="drop-icon">📁</div>
                <p className="drop-title">
                  {isDragActive ? "Drop it here" : "Drag & drop your image"}
                </p>
                <p className="drop-sub">or</p>
                <p className="drop-browse">Browse files</p>
                <p className="drop-hint">JPG, PNG, WEBP · Max 5MB</p>
              </>
            )}
          </div>
          {/* {error && <p className="upload-error">{error}</p>} */}
          {error && <p className="upload-error">{error}</p>}
          <div>
            <p className="panel-label mb-2">Detection mode</p>
            <div className="mode-badge">
              <div className="mode-dot" />
              AI Generated Detection
            </div>
          </div>

          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!file || isLoading}
          >
            {isLoading ? "Analyzing..." : "Analyze image"}
          </button>
        </div>
        {result ? (
          <ResultPanel
            result={result}
            previewUrl={previewUrl}
            onReset={handleReset}
          />
        ) : (
          <EmptyResult />
        )}
      </div>
    </div>
  );
};

export default UploadPage;
