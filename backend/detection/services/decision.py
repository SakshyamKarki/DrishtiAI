"""
DrishtiAI — Improved Hybrid Decision Engine v2
===============================================
Combines 7 analysis signals for robust fake-profile face detection:

  1. Deep Learning (ResNet18)        — primary signal, 40% weight
  2. Frequency Analysis (DCT/FFT)    — GAN spectral fingerprints, 15%
  3. LBP Texture                     — skin texture richness, 12%
  4. Color Statistics                — color naturalness + symmetry, 12%
  5. K-Means Variance                — pixel diversity, 8%
  6. Sobel Edge Score                — edge sharpness, 7%
  7. Shannon Entropy                 — information complexity, 6%

Weights calibrated for the fake-PROFILE-PHOTO use case:
  - Profile photos are typically front-facing, well-lit → classical signals useful
  - Frequency & LBP are especially strong for AI-generated faces
  - DL model still gets highest weight for reliability

Decision thresholds (learned from calibration analysis):
  score ≥ 0.62  → FAKE
  score ≤ 0.38  → REAL
  else           → SUSPICIOUS

Confidence calibration:
  - Scores near threshold boundaries are inherently uncertain
  - We use a sigmoid-shaped confidence ramp to avoid false certainty
"""

from dataclasses import dataclass
from typing import Literal

# ── Weights (must sum to 1.0) ───────────────────────────────────────────────────
DL_W    = 0.40   # Deep learning model (ResNet18)
FREQ_W  = 0.15   # Frequency domain analysis
LBP_W   = 0.12   # Local Binary Pattern texture
COLOR_W = 0.12   # Color statistics
KM_W    = 0.08   # K-Means pixel diversity
EDGE_W  = 0.07   # Sobel edge detection
ENTR_W  = 0.06   # Shannon entropy

_WEIGHT_SUM = DL_W + FREQ_W + LBP_W + COLOR_W + KM_W + EDGE_W + ENTR_W
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, f"Weights must sum to 1.0 (got {_WEIGHT_SUM})"

# ── Decision thresholds ─────────────────────────────────────────────────────────
FAKE_THRESH = 0.62
REAL_THRESH = 0.38

Verdict = Literal["FAKE", "REAL", "SUSPICIOUS"]


# ── Result dataclass ────────────────────────────────────────────────────────────

@dataclass
class HybridResultV2:
    verdict:            Verdict
    confidence:         float    # 0–100 %
    decision_score:     float    # 0–1 (higher → more likely fake)

    # Individual signal values (0–1)
    dl_is_fake:         bool
    dl_confidence:      float    # raw DL confidence (0–100)
    freq_score:         float    # frequency fakeness signal
    lbp_score:          float    # LBP realness → converted to fakeness
    color_score:        float    # color realness → converted to fakeness
    kmeans_score:       float    # kmeans variance (realness)
    edge_score:         float    # edge strength (realness)
    entropy_score:      float    # entropy (realness)

    # Face meta
    face_meta:          dict

    def to_api_dict(self) -> dict:
        """Produce the full JSON-serialisable response body."""
        return {
            # Primary result
            "final_label":    self.verdict,
            "is_fake":        self.verdict == "FAKE",
            "confidence":     round(self.confidence, 2),
            "decision_score": round(self.decision_score, 4),

            # Deep learning
            "dl_prediction": {
                "label":      "Fake" if self.dl_is_fake else "Real",
                "confidence": round(self.dl_confidence, 2),
            },

            # Classical analysis breakdown
            "analysis": {
                # Frequency domain
                "frequency": {
                    "score":       round(self.freq_score, 4),
                    "description": "GAN spectral fingerprint strength",
                    "signal":      "high = AI fingerprint detected",
                },
                # LBP texture
                "texture_lbp": {
                    "score":       round(1.0 - self.lbp_score, 4),
                    "description": "Skin texture richness (LBP)",
                    "signal":      "high = unnaturally smooth (AI)",
                },
                # Color statistics
                "color_stats": {
                    "score":       round(1.0 - self.color_score, 4),
                    "description": "Color distribution naturalness",
                    "signal":      "high = unnatural color patterns",
                },
                # K-Means
                "pixel_diversity": {
                    "score":       round(1.0 - self.kmeans_score, 4),
                    "description": "Pixel cluster diversity (K-Means)",
                    "signal":      "high = uniform/artificial texture",
                },
                # Sobel edge
                "edge_sharpness": {
                    "score":       round(1.0 - self.edge_score, 4),
                    "description": "Sobel edge strength",
                    "signal":      "high = weak/blurred edges",
                },
                # Entropy
                "information_density": {
                    "score":       round(1.0 - self.entropy_score, 4),
                    "description": "Shannon entropy complexity",
                    "signal":      "high = low information density",
                },
            },

            # Weights used
            "weights": {
                "deep_learning":    DL_W,
                "frequency_domain": FREQ_W,
                "lbp_texture":      LBP_W,
                "color_statistics": COLOR_W,
                "pixel_diversity":  KM_W,
                "edge_detection":   EDGE_W,
                "entropy":          ENTR_W,
            },

            # Face detection meta
            "face_detection": self.face_meta,
        }


# ── Confidence Calibration ──────────────────────────────────────────────────────

def _calibrate_confidence(score: float, verdict: Verdict) -> float:
    """
    Convert raw decision score to a human-interpretable confidence %.

    Uses a piecewise approach:
      - Very high/low scores → high confidence (90-99%)
      - Near-threshold scores → lower confidence (55-75%)
      - Suspicious range → fixed 50-70% confidence

    This avoids showing "100% FAKE" when the model is actually uncertain.
    """
    if verdict == "SUSPICIOUS":
        # Distance from midpoint within suspicious zone
        dist = abs(score - 0.50)
        return float(50.0 + dist * 100.0)

    # Distance from nearest threshold
    if verdict == "FAKE":
        dist = score - FAKE_THRESH   # 0 at threshold, max at 1.0
        max_dist = 1.0 - FAKE_THRESH
    else:  # REAL
        dist = REAL_THRESH - score   # 0 at threshold, max at 1.0
        max_dist = REAL_THRESH

    ratio = float(dist / max_dist)

    # Sigmoid-style ramp: low confidence near threshold, high far from it
    # Map [0, 1] → [55, 98]
    confidence = 55.0 + ratio * 43.0
    return float(min(confidence, 98.0))


# ── Main Engine ─────────────────────────────────────────────────────────────────

def compute_decision_v2(
    # Deep learning
    dl_is_fake:      bool,
    dl_confidence:   float,   # 0–1 (raw model confidence in its own label)

    # Frequency analysis
    freq_score:      float,   # 0–1 (higher = more fake signal)

    # LBP texture (realness score → we invert for fake signal)
    lbp_realness:    float,   # 0–1 (higher = more real texture)

    # Color statistics (realness score)
    color_realness:  float,   # 0–1 (higher = more natural color)

    # Classical from v1
    kmeans_variance: float,   # 0–1 (higher = more pixel diversity = real)
    edge_score:      float,   # 0–1 (higher = stronger edges = real)
    entropy_score:   float,   # 0–1 (higher = more complex = real)

    # Face detection metadata
    face_meta:       dict = None,
) -> HybridResultV2:
    """
    Run the hybrid decision engine on all 7 signals.

    All "realness" signals are converted to "fakeness" before weighting:
      fakeness = 1 - realness_score

    Args:
        dl_is_fake      : True if DL model predicts fake
        dl_confidence   : DL model's confidence (0–1) in its own prediction
        freq_score      : frequency domain fakeness (0=real, 1=fake)
        lbp_realness    : LBP texture realness (0=fake, 1=real)
        color_realness  : color statistics realness (0=fake, 1=real)
        kmeans_variance : K-Means pixel diversity (high=real)
        edge_score      : Sobel edge score (high=real)
        entropy_score   : Shannon entropy (high=real)
        face_meta       : dict from face detection (warnings, quality info)

    Returns:
        HybridResultV2 with full verdict and breakdown
    """
    face_meta = face_meta or {}

    # ── Convert DL output to p_fake ────────────────────────────────────────────
    p_fake = dl_confidence if dl_is_fake else (1.0 - dl_confidence)

    # ── Convert realness signals to fakeness ───────────────────────────────────
    lbp_fake   = 1.0 - lbp_realness
    color_fake = 1.0 - color_realness
    km_fake    = 1.0 - kmeans_variance
    edge_fake  = 1.0 - edge_score
    entr_fake  = 1.0 - entropy_score

    # ── Weighted combination ────────────────────────────────────────────────────
    score = (DL_W    * p_fake    +
             FREQ_W  * freq_score +
             LBP_W   * lbp_fake  +
             COLOR_W * color_fake +
             KM_W    * km_fake   +
             EDGE_W  * edge_fake +
             ENTR_W  * entr_fake)
    score = float(min(max(score, 0.0), 1.0))

    # ── Face quality adjustment ─────────────────────────────────────────────────
    # If face detection was uncertain, reduce confidence in classical signals
    no_face = "no_face_detected" in face_meta.get("warning", "")
    if no_face:
        # No face found: rely more on DL, discount classical signals
        score = 0.6 * p_fake + 0.4 * score

    # ── Verdict ─────────────────────────────────────────────────────────────────
    if score >= FAKE_THRESH:
        verdict: Verdict = "FAKE"
    elif score <= REAL_THRESH:
        verdict = "REAL"
    else:
        verdict = "SUSPICIOUS"

    confidence = _calibrate_confidence(score, verdict)

    return HybridResultV2(
        verdict          = verdict,
        confidence       = round(confidence, 2),
        decision_score   = round(score, 4),
        dl_is_fake       = dl_is_fake,
        dl_confidence    = round(dl_confidence * 100.0, 2),
        freq_score       = round(freq_score, 4),
        lbp_score        = round(lbp_realness, 4),
        color_score      = round(color_realness, 4),
        kmeans_score     = round(kmeans_variance, 4),
        edge_score       = round(edge_score, 4),
        entropy_score    = round(entropy_score, 4),
        face_meta        = face_meta,
    )
