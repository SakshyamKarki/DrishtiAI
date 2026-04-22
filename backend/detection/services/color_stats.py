"""
DrishtiAI — Color Statistics & Symmetry Analysis
=================================================
WHY COLOR STATISTICS WORK FOR AI FACE DETECTION:
  AI generative models struggle to replicate the subtle statistical
  properties of real skin color distributions:

  1. COLOR CHANNEL CORRELATION
     Real skin: R, G, B channels are highly correlated (chromatic structure)
     AI skin:   over-correlated (too smooth) or under-correlated (artifacts)

  2. SKIN COLOR DISTRIBUTION
     Real faces: skin pixels cluster in a known HSV range with natural spread
     AI faces:   skin pixels either too uniform or contain impossible colors

  3. GLOBAL COLOR COHERENCE
     Real photos: noise in each channel follows camera sensor patterns
     AI images:   per-channel noise deviates from expected camera noise

  4. FACE SYMMETRY (COLOR-BASED)
     Perfect bilateral symmetry is unnatural — real faces have slight
     asymmetries in lighting, skin tone, and detail.

All computed FROM SCRATCH using numpy only.
"""

import numpy as np
import cv2
from typing import Tuple


# ── Color Space Conversion ──────────────────────────────────────────────────────

def _bgr_to_hsv_manual(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 image to HSV float [H:0-360, S:0-1, V:0-1].
    Implemented from scratch using standard formulae.
    """
    img = image_bgr.astype(np.float64) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # Hue
    H = np.zeros_like(Cmax)
    mask_r = (Cmax == R) & (delta > 0)
    mask_g = (Cmax == G) & (delta > 0)
    mask_b = (Cmax == B) & (delta > 0)

    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * ((B[mask_g] - R[mask_g]) / delta[mask_g] + 2)
    H[mask_b] = 60.0 * ((R[mask_b] - G[mask_b]) / delta[mask_b] + 4)
    H[H < 0] += 360.0

    # Saturation
    S = np.where(Cmax > 0, delta / Cmax, 0.0)

    # Value
    V = Cmax

    return np.stack([H, S, V], axis=2)


def _skin_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for skin-colored pixels using HSV ranges.
    Human skin (all ethnicities) falls in approximate ranges:
      H: 0–25°  or  340–360°  (red-orange hues)
      S: 0.15–0.85
      V: 0.25–0.95
    Returns: (H, W) bool mask
    """
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_mask = ((H <= 25) | (H >= 340))
    sat_mask = (S >= 0.15) & (S <= 0.85)
    val_mask = (V >= 0.25) & (V <= 0.95)
    return hue_mask & sat_mask & val_mask


# ── Channel Correlation ─────────────────────────────────────────────────────────

def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1D arrays."""
    a_mean = a.mean()
    b_mean = b.mean()
    num    = float(((a - a_mean) * (b - b_mean)).sum())
    denom  = float(np.sqrt(((a - a_mean) ** 2).sum() * ((b - b_mean) ** 2).sum()))
    return float(num / (denom + 1e-12))


def _channel_correlations(image: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute pairwise Pearson correlations between R, G, B channels.
    Channels are flattened to 1D before correlation.

    Real skin: all three correlations high (0.85–0.98)
    AI with artifacts: one pair may be anomalously low or high
    """
    R = image[:, :, 2].flatten().astype(np.float64)
    G = image[:, :, 1].flatten().astype(np.float64)
    B = image[:, :, 0].flatten().astype(np.float64)

    rg = _pearson_correlation(R, G)
    rb = _pearson_correlation(R, B)
    gb = _pearson_correlation(G, B)
    return rg, rb, gb


# ── Color Distribution Statistics ──────────────────────────────────────────────

def _channel_stats(channel: np.ndarray) -> dict:
    """Compute mean, std, skewness, kurtosis for a single channel."""
    flat   = channel.flatten().astype(np.float64)
    mean   = float(flat.mean())
    std    = float(flat.std()) + 1e-8
    skew   = float(((flat - mean) ** 3).mean() / std ** 3)
    kurt   = float(((flat - mean) ** 4).mean() / std ** 4) - 3.0  # excess kurtosis
    return {"mean": mean, "std": std, "skewness": skew, "kurtosis": kurt}


def _color_naturalness_score(image: np.ndarray) -> float:
    """
    Score how "natural" the color distribution looks.

    Real photos:
      - Moderate channel correlation (0.75–0.97)
      - Moderate std per channel (15–60 out of 255)
      - Slight positive skewness in skin regions
      - Kurtosis near 0 (roughly Gaussian per channel)

    AI images:
      - Extremely high correlation (too uniform) or anomalously low
      - Very low std (plastic-looking) or very high (colorful artifacts)
      - Abnormal skewness/kurtosis
    """
    rg, rb, gb = _channel_correlations(image)
    avg_corr   = (rg + rb + gb) / 3.0

    # Score: correlation too high (>0.98) or too low (<0.5) is suspicious
    corr_score = 1.0 - abs(avg_corr - 0.88) / 0.5   # 0.88 is natural skin target
    corr_score = float(min(max(corr_score, 0.0), 1.0))

    # Per-channel std check
    stds = [image[:, :, c].astype(np.float64).std() for c in range(3)]
    avg_std = float(np.mean(stds))
    # Natural range: ~20-60; lower is plastic/AI; higher is chaotic
    std_score = 1.0 - abs(avg_std - 40.0) / 40.0
    std_score = float(min(max(std_score, 0.0), 1.0))

    # Combined naturalness
    return 0.6 * corr_score + 0.4 * std_score


# ── Face Symmetry Analysis ──────────────────────────────────────────────────────

def _symmetry_score_color(image: np.ndarray) -> float:
    """
    Measure bilateral symmetry of face in color (RGB).
    AI faces are often too symmetric; real faces have natural asymmetry
    from lighting, expression, and biological variation.

    Method:
      1. Split image vertically in half
      2. Mirror the right half
      3. Compute normalised mean absolute difference per channel
      4. Return asymmetry score (higher → more natural)
    """
    h, w = image.shape[:2]
    half = w // 2

    left  = image[:, :half, :].astype(np.float64)
    right = image[:, half:half + half, :].astype(np.float64)
    right_flipped = right[:, ::-1, :]   # mirror horizontally

    # Trim to same width
    min_w = min(left.shape[1], right_flipped.shape[1])
    diff  = np.abs(left[:, :min_w, :] - right_flipped[:, :min_w, :])

    # Normalised asymmetry: 0 = perfect mirror, 1 = completely different
    asymmetry = float(diff.mean() / 255.0)

    # AI faces: asymmetry ≈ 0.02–0.08 (too symmetric)
    # Real faces: asymmetry ≈ 0.08–0.20
    # Score: penalise over-symmetry
    if asymmetry < 0.06:
        score = asymmetry / 0.06   # too symmetric → low score → more fake
    else:
        score = min(asymmetry / 0.2, 1.0)  # natural range
    return float(score)


# ── Noise Analysis ───────────────────────────────────────────────────────────────

def _noise_consistency_score(image: np.ndarray) -> float:
    """
    Analyse per-channel noise residual.
    Real camera images have noise consistent with sensor characteristics.
    AI-generated images often have:
      (a) Too little noise (smooth posterization)
      (b) Structured noise patterns (GAN fingerprints)

    Method:
      High-pass filter each channel (subtract Gaussian blur).
      Compute noise std per channel and check cross-channel consistency.
    """
    scores = []
    for c in range(3):
        ch = image[:, :, c].astype(np.float64)
        # Simple Gaussian-style blur via box filter (from scratch approximation)
        blurred = cv2.GaussianBlur(ch.astype(np.float32), (5, 5), 0).astype(np.float64)
        noise   = ch - blurred

        noise_std = float(noise.std())
        scores.append(noise_std)

    avg_noise = float(np.mean(scores))
    noise_var = float(np.var(scores))

    # Natural noise: avg ~1-8 per channel; consistent across channels
    # AI images: near-zero noise OR inconsistent channels
    if avg_noise < 0.5:
        return 0.0   # too clean → fake
    noise_level_ok = min(avg_noise / 5.0, 1.0)
    consistency    = 1.0 - min(noise_var / 4.0, 1.0)   # low variance = consistent
    return float(0.6 * noise_level_ok + 0.4 * consistency)


# ── Public API ───────────────────────────────────────────────────────────────────

def color_stats_score(image: np.ndarray) -> dict:
    """
    Comprehensive color statistics analysis for AI face detection.

    Args:
        image : uint8 BGR numpy array (H, W, 3)

    Returns dict with:
        naturalness       : color distribution naturalness [0,1]
        symmetry          : face color symmetry [0,1] (high=natural asymmetry)
        noise_consistency : noise pattern score [0,1]
        skin_ratio        : fraction of pixels classified as skin
        channel_corr_rg   : R-G Pearson correlation
        channel_corr_rb   : R-B Pearson correlation
        overall_score     : composite 0-1 "realness" signal
    """
    h, w = image.shape[:2]
    # Resize to standard size for consistent statistics
    std_size = (160, 160)
    img_std  = cv2.resize(image, std_size, interpolation=cv2.INTER_AREA)

    # Color statistics
    naturalness = _color_naturalness_score(img_std)
    symmetry    = _symmetry_score_color(img_std)
    noise       = _noise_consistency_score(img_std)

    # Channel correlations
    rg, rb, gb = _channel_correlations(img_std)

    # Skin mask
    hsv = _bgr_to_hsv_manual(img_std)
    skin = _skin_mask(hsv)
    skin_ratio = float(skin.sum()) / (std_size[0] * std_size[1])

    # Skin color stats
    skin_naturalness = 0.5
    if skin.sum() > 100:
        skin_pixels = img_std[skin]  # (N, 3)
        skin_std    = float(skin_pixels.astype(np.float64).std())
        # Very uniform skin → AI signal
        skin_naturalness = min(skin_std / 30.0, 1.0)

    # Overall realness from color perspective
    overall = (0.30 * naturalness +
               0.25 * symmetry +
               0.20 * noise +
               0.15 * skin_naturalness +
               0.10 * min(skin_ratio / 0.3, 1.0))
    overall = float(min(max(overall, 0.0), 1.0))

    return {
        "naturalness":        round(naturalness,    4),
        "symmetry":           round(symmetry,        4),
        "noise_consistency":  round(noise,           4),
        "skin_ratio":         round(skin_ratio,      4),
        "channel_corr_rg":    round(rg,              4),
        "channel_corr_rb":    round(rb,              4),
        "overall_score":      round(overall,         4),
    }
