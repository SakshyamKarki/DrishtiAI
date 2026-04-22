"""
DrishtiAI — Hybrid Decision Engine v3 [REFACTORED]
====================================================
Changes from v1:
  1. Added kmeans_elbow_fakeness to the weighted score (2% weight,
     taken from re-distributed KM_W to keep total = 1.0).
  2. Fixed a bug: the no_face modifier used `"no_face_detected" in
     face_meta.get("warning", "")` but NoFaceError now raises before
     reaching the decision engine — this branch is kept as a safety net.
  3. confidence calculation now guards against SUSPICIOUS verdict
     returning >98 in edge cases.
  4. HybridResultV3.to_api_dict() now includes bit_fakeness and elbow signal
     in the detailed breakdown.

Updated Weight Distribution (sums to 1.0):
  DL (ResNet18):        38%
  Frequency (DCT):      10%
  LBP Texture:           8%
  Color Statistics:      8%
  K-Means Variance:      4%   ← reduced by 1%
  K-Means Elbow:         1%   ← NEW signal
  Sobel Edge:            4%
  Shannon Entropy:       3%
  SSIM Texture:          5%
  HSV Skin Uniformity:   5%
  Laplacian Sharpness:   4%
  Chromatic Aberration:  4%
  Background Coherence:  3%
  Noise Pattern:         3%
"""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any

# ── Weights ──────────────────────────────────────────────────────────────────
DL_W    = 0.38
FREQ_W  = 0.10
LBP_W   = 0.08
COLOR_W = 0.08
KM_W    = 0.04   # reduced from 0.05
KM_EL_W = 0.01   # NEW: elbow signal
EDGE_W  = 0.04
ENTR_W  = 0.03
SSIM_W  = 0.05
HSV_W   = 0.05
LAP_W   = 0.04
CA_W    = 0.04
BG_W    = 0.03
NOISE_W = 0.03

_TOTAL = (DL_W + FREQ_W + LBP_W + COLOR_W + KM_W + KM_EL_W +
          EDGE_W + ENTR_W + SSIM_W + HSV_W + LAP_W + CA_W + BG_W + NOISE_W)
assert abs(_TOTAL - 1.0) < 1e-9, f"Weights must sum to 1.0 (got {_TOTAL:.6f})"

FAKE_THRESH  = 0.60
REAL_THRESH  = 0.36
Verdict = Literal["FAKE", "REAL", "SUSPICIOUS"]

RISK_LABELS = {
    "FAKE":       "High Risk — Likely AI Generated",
    "REAL":       "Low Risk — Likely Authentic",
    "SUSPICIOUS": "Medium Risk — Needs Review",
}


@dataclass
class HybridResultV3:
    verdict:        Verdict
    confidence:     float
    decision_score: float
    risk_label:     str

    dl_is_fake:     bool
    dl_confidence:  float

    freq_score:     float
    lbp_score:      float
    color_score:    float
    kmeans_score:   float
    km_elbow_fake:  float    # NEW
    edge_score:     float
    entropy_score:  float

    ssim_fakeness:  float
    hsv_fakeness:   float
    lap_realness:   float
    ca_realness:    float
    bg_realness:    float
    noise_realness: float

    face_meta:         dict
    enhanced_signals:  dict

    def to_api_dict(self) -> dict:
        return {
            "verdict":        self.verdict,
            "final_label":    self.verdict,
            "is_fake":        self.verdict == "FAKE",
            "confidence":     round(self.confidence, 2),
            "decision_score": round(self.decision_score, 4),
            "risk_label":     self.risk_label,

            "profile_analysis": {
                "face_detected":      self.face_meta.get("found", False),
                "face_count":         self.enhanced_signals.get("face_quality", {}).get("face_count", 0),
                "single_face":        self.enhanced_signals.get("face_quality", {}).get("single_face", None),
                "face_quality_score": round(self.enhanced_signals.get("face_quality", {}).get("quality", 0.5), 3),
                "face_sharpness":     self.face_meta.get("sharpness", 0),
                "symmetry_score":     self.face_meta.get("landmark_symmetry_score", 0),  # FIXED key
                "skin_uniformity":    round(self.hsv_fakeness, 3),
                "background_natural": round(self.bg_realness, 3),
            },

            "dl_prediction": {
                "label":      "Fake" if self.dl_is_fake else "Real",
                "confidence": round(self.dl_confidence, 2),
                "weight":     DL_W,
            },

            "analysis": {
                "frequency": {
                    "score": round(self.freq_score, 4),
                    "description": "GAN spectral fingerprint strength (DCT/FFT)",
                    "signal": "high = AI upsampling artifacts",
                    "weight": FREQ_W,
                },
                "texture_lbp": {
                    "score": round(1.0 - self.lbp_score, 4),
                    "description": "Skin micro-texture richness (LBP + bit-counting)",
                    "signal": "high = unnaturally smooth skin",
                    "weight": LBP_W,
                },
                "color_stats": {
                    "score": round(1.0 - self.color_score, 4),
                    "description": "Color distribution naturalness",
                    "signal": "high = unnatural color patterns",
                    "weight": COLOR_W,
                },
                "pixel_diversity": {
                    "score": round(1.0 - self.kmeans_score, 4),
                    "description": "Pixel cluster diversity (K-Means)",
                    "signal": "high = uniform/artificial texture",
                    "weight": KM_W,
                },
                "kmeans_elbow": {
                    "score": round(self.km_elbow_fake, 4),
                    "description": "K-Means elbow ratio W(k=2)/W(k=8)",
                    "signal": "high = near-uniform cluster decay (AI)",
                    "weight": KM_EL_W,
                },
                "edge_sharpness": {
                    "score": round(1.0 - self.edge_score, 4),
                    "description": "Sobel edge + gradient direction entropy",
                    "signal": "high = weak/blurred edges with few directions",
                    "weight": EDGE_W,
                },
                "information_density": {
                    "score": round(1.0 - self.entropy_score, 4),
                    "description": "Shannon entropy complexity",
                    "signal": "high = low information density",
                    "weight": ENTR_W,
                },
                "ssim_texture": {
                    "score": round(self.ssim_fakeness, 4),
                    "description": "SSIM patch smoothness",
                    "signal": "high = over-smooth AI skin texture",
                    "weight": SSIM_W,
                },
                "skin_tone_uniformity": {
                    "score": round(self.hsv_fakeness, 4),
                    "description": "HSV skin hue distribution analysis",
                    "signal": "high = unnaturally uniform skin tone",
                    "weight": HSV_W,
                },
                "sharpness_profile": {
                    "score": round(1.0 - self.lap_realness, 4),
                    "description": "Multi-scale Laplacian sharpness profile",
                    "signal": "high = unnatural sharpness distribution",
                    "weight": LAP_W,
                },
                "lens_physics": {
                    "score": round(1.0 - self.ca_realness, 4),
                    "description": "Chromatic aberration (real lens physics)",
                    "signal": "high = missing natural lens aberration",
                    "weight": CA_W,
                },
                "background_analysis": {
                    "score": round(1.0 - self.bg_realness, 4),
                    "description": "Face/background coherence",
                    "signal": "high = artificial background separation",
                    "weight": BG_W,
                },
                "sensor_noise": {
                    "score": round(1.0 - self.noise_realness, 4),
                    "description": "Camera sensor noise pattern analysis",
                    "signal": "high = missing/structured noise (AI artifact)",
                    "weight": NOISE_W,
                },
            },

            "weights": {
                "deep_learning":        DL_W,
                "frequency_domain":     FREQ_W,
                "lbp_texture":          LBP_W,
                "color_statistics":     COLOR_W,
                "pixel_diversity":      KM_W,
                "kmeans_elbow":         KM_EL_W,
                "edge_detection":       EDGE_W,
                "entropy":              ENTR_W,
                "ssim_texture":         SSIM_W,
                "skin_hsv":             HSV_W,
                "laplacian_sharpness":  LAP_W,
                "chromatic_aberration": CA_W,
                "background_coherence": BG_W,
                "noise_pattern":        NOISE_W,
            },

            "face_detection": self.face_meta,
        }


def _calibrate_confidence_v3(score: float, verdict: Verdict) -> float:
    if verdict == "SUSPICIOUS":
        dist = abs(score - 0.50)
        # Cap at 70% for suspicious — high confidence suspicious is an oxymoron
        return float(min(50.0 + dist * 80.0, 70.0))

    if verdict == "FAKE":
        dist    = score - FAKE_THRESH
        max_dist = 1.0 - FAKE_THRESH
    else:
        dist    = REAL_THRESH - score
        max_dist = REAL_THRESH

    ratio = float(dist / (max_dist + 1e-9))
    confidence = 55.0 + ratio * 43.0
    return float(min(confidence, 98.0))


def compute_decision_v3(
    dl_is_fake:      bool,
    dl_confidence:   float,
    freq_score:      float,
    lbp_realness:    float,
    color_realness:  float,
    kmeans_variance: float,
    edge_score:      float,
    entropy_score:   float,
    enhanced_signals: Dict[str, Any],
    face_meta:       dict = None,
) -> HybridResultV3:

    face_meta        = face_meta or {}
    enhanced_signals = enhanced_signals or {}

    p_fake = dl_confidence if dl_is_fake else (1.0 - dl_confidence)

    lbp_fake   = 1.0 - lbp_realness
    color_fake = 1.0 - color_realness
    km_fake    = 1.0 - kmeans_variance
    edge_fake  = 1.0 - edge_score
    entr_fake  = 1.0 - entropy_score

    ssim_fake  = float(enhanced_signals.get("ssim_texture_fakeness",           0.5))
    hsv_fake   = float(enhanced_signals.get("hsv_skin_uniformity_fakeness",    0.5))
    lap_real   = float(enhanced_signals.get("laplacian_sharpness_realness",    0.5))
    ca_real    = float(enhanced_signals.get("chromatic_aberration_realness",   0.5))
    bg_real    = float(enhanced_signals.get("background_coherence_realness",   0.5))
    noise_real = float(enhanced_signals.get("noise_pattern_realness",          0.5))
    km_elbow   = float(enhanced_signals.get("kmeans_elbow_fakeness",           0.5))  # NEW

    lap_fake   = 1.0 - lap_real
    ca_fake    = 1.0 - ca_real
    bg_fake    = 1.0 - bg_real
    noise_fake = 1.0 - noise_real

    score = (
        DL_W    * p_fake    +
        FREQ_W  * freq_score +
        LBP_W   * lbp_fake  +
        COLOR_W * color_fake +
        KM_W    * km_fake   +
        KM_EL_W * km_elbow  +
        EDGE_W  * edge_fake  +
        ENTR_W  * entr_fake  +
        SSIM_W  * ssim_fake  +
        HSV_W   * hsv_fake   +
        LAP_W   * lap_fake   +
        CA_W    * ca_fake    +
        BG_W    * bg_fake    +
        NOISE_W * noise_fake
    )
    score = float(min(max(score, 0.0), 1.0))

    # Profile-photo modifiers
    face_quality = enhanced_signals.get("face_quality", {})
    face_count   = face_quality.get("face_count", 1)
    fq_score     = face_quality.get("quality", 0.5)

    if isinstance(face_count, int) and face_count > 1:
        score = score * 0.7 + 0.5 * 0.3

    # Safety net: no_face should never reach here after inference_v3 guards,
    # but keep for robustness
    no_face = "no_face_detected" in face_meta.get("warning", "")
    if no_face:
        score = 0.65 * p_fake + 0.35 * score

    if score >= FAKE_THRESH:
        verdict: Verdict = "FAKE"
    elif score <= REAL_THRESH:
        verdict = "REAL"
    else:
        verdict = "SUSPICIOUS"

    confidence = _calibrate_confidence_v3(score, verdict)
    if fq_score < 0.3:
        confidence *= 0.85

    return HybridResultV3(
        verdict=verdict,
        confidence=round(confidence, 2),
        decision_score=round(score, 4),
        risk_label=RISK_LABELS[verdict],
        dl_is_fake=dl_is_fake,
        dl_confidence=round(dl_confidence * 100.0, 2),
        freq_score=round(freq_score, 4),
        lbp_score=round(lbp_realness, 4),
        color_score=round(color_realness, 4),
        kmeans_score=round(kmeans_variance, 4),
        km_elbow_fake=round(km_elbow, 4),
        edge_score=round(edge_score, 4),
        entropy_score=round(entropy_score, 4),
        ssim_fakeness=round(ssim_fake, 4),
        hsv_fakeness=round(hsv_fake, 4),
        lap_realness=round(lap_real, 4),
        ca_realness=round(ca_real, 4),
        bg_realness=round(bg_real, 4),
        noise_realness=round(noise_real, 4),
        face_meta=face_meta,
        enhanced_signals=enhanced_signals,
    )
