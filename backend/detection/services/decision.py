"""
Hybrid Decision Engine
======================
Combines deep-learning prediction with three classical algorithm scores
to produce a final verdict that is more robust than any single signal.

Weighted scoring formula
------------------------
  decision_score = DL_W · p_fake
                 + KM_W · (1 − kmeans_variance)
                 + ED_W · (1 − edge_score)
                 + EN_W · (1 − entropy_score)

Where:
  p_fake           = DL model's probability that the image is fake
  (1 − km_var)     = LOW diversity  → suspicious
  (1 − edge_score) = WEAK edges     → suspicious
  (1 − entropy)    = LOW complexity → suspicious

Final label:
  decision_score ≥ 0.60  → "Deepfake"
  decision_score ≤ 0.40  → "Real"
  0.40 < score < 0.60    → "Suspicious"

Weights intentionally give the DL model the most influence (55%) while
allowing the classical signals to correct clear DL errors (45% combined).
"""

from dataclasses import dataclass, field
from typing import Literal

# ── weights ────────────────────────────────────────────────────────────────────
DL_W = 0.55
KM_W = 0.15
ED_W = 0.15
EN_W = 0.15
assert abs(DL_W + KM_W + ED_W + EN_W - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── thresholds ─────────────────────────────────────────────────────────────────
FAKE_THRESH = 0.60
REAL_THRESH = 0.40

Verdict = Literal["Real", "Deepfake", "Suspicious"]


@dataclass
class HybridResult:
    final_label:      Verdict
    confidence:       float          # 0–100 %
    decision_score:   float          # 0–1  (higher → more likely fake)
    dl_is_fake:       bool
    dl_confidence:    float          # 0–100 %
    kmeans_variance:  float          # 0–1
    edge_score:       float          # 0–1
    entropy_score:    float          # 0–1

    def to_api_dict(self) -> dict:
        return {
            "final_label":    self.final_label,
            "confidence":     round(self.confidence,    2),
            "decision_score": round(self.decision_score, 4),
            "dl_is_fake":     self.dl_is_fake,
            "dl_confidence":  round(self.dl_confidence, 2),
            "analysis": {
                "kmeans_variance": round(self.kmeans_variance, 4),
                "edge_score":      round(self.edge_score,      4),
                "entropy_score":   round(self.entropy_score,   4),
            },
            "weights": {
                "deep_learning":   DL_W,
                "kmeans":          KM_W,
                "edge_detection":  ED_W,
                "entropy":         EN_W,
            },
        }


def compute_decision(
    dl_is_fake:       bool,
    dl_confidence:    float,   # raw model confidence in its own prediction (0–1)
    kmeans_variance:  float,   # 0–1
    edge_score:       float,   # 0–1
    entropy_score:    float,   # 0–1
) -> HybridResult:
    """
    Combine all signals into a final verdict.

    Args:
        dl_is_fake     : True when the DL model predicts the image is fake
        dl_confidence  : DL model's confidence (0–1) in its own prediction
        kmeans_variance: cluster diversity score (high = real)
        edge_score     : Sobel edge strength (high = real)
        entropy_score  : Shannon entropy (high = real)

    Returns:
        HybridResult with verdict, confidence, and all raw scores
    """
    # convert DL output to a "probability of being fake" in [0, 1]
    p_fake = dl_confidence if dl_is_fake else (1.0 - dl_confidence)

    # each classical metric: high = real → invert to get a "fake signal"
    km_fake  = 1.0 - kmeans_variance
    ed_fake  = 1.0 - edge_score
    en_fake  = 1.0 - entropy_score

    # weighted combination
    score = (DL_W * p_fake  +
             KM_W * km_fake +
             ED_W * ed_fake +
             EN_W * en_fake)

    # verdict
    if score >= FAKE_THRESH:
        label: Verdict = "Deepfake"
        conf  = score * 100.0
    elif score <= REAL_THRESH:
        label = "Real"
        conf  = (1.0 - score) * 100.0
    else:
        label = "Suspicious"
        conf  = 50.0 + abs(score - 0.5) * 100.0

    conf = max(50.0, min(99.0, conf))

    return HybridResult(
        final_label     = label,
        confidence      = round(conf,  2),
        decision_score  = round(score, 4),
        dl_is_fake      = dl_is_fake,
        dl_confidence   = round(dl_confidence * 100.0, 2),
        kmeans_variance = round(kmeans_variance, 4),
        edge_score      = round(edge_score,      4),
        entropy_score   = round(entropy_score,   4),
    )
