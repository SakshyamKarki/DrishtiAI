"""
Shannon Entropy — Implemented from Scratch
==========================================
Purpose: Measure information complexity of a face image.

Shannon entropy formula:
    H = -∑ p_i · log₂(p_i)       (sum over all non-zero probabilities)

  p_i = (count of pixels with intensity i) / total_pixels

Interpretation for deepfake detection:
  Real images   → high entropy  (random, complex textures)  → HIGH score
  AI/Deepfake   → lower entropy (smooth, repetitive textures) → LOW score

Maximum entropy for 8-bit (256 levels): H_max = log₂(256) = 8.0 bits
We divide H by H_max to get a score in [0, 1].
"""

import numpy as np

_MAX_ENTROPY_BITS = np.log2(256)   # = 8.0


# ── helpers ────────────────────────────────────────────────────────────────────

def _histogram(channel: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Build a pixel-intensity histogram with numpy bincount (O(N), no loops).
    bincount is conceptually equivalent to a manual counter array — it tallies
    occurrences of each integer value in the input.
    """
    flat = channel.flatten().astype(np.int32)
    return np.bincount(flat, minlength=bins)[:bins]


def _shannon(hist: np.ndarray) -> float:
    """
    Compute Shannon entropy from a histogram.

    H = -∑ p_i · log₂(p_i)   for p_i > 0

    We skip zero-count bins because 0 · log₂(0) = 0 by convention.
    """
    total = int(hist.sum())
    if total == 0:
        return 0.0
    probs    = hist.astype(np.float64) / total          # normalise to [0,1]
    nonzero  = probs[probs > 0]                         # avoid log(0)
    return float(-np.sum(nonzero * np.log2(nonzero)))   # Shannon formula


# ── public API ─────────────────────────────────────────────────────────────────

def image_entropy_score(image: np.ndarray) -> float:
    """
    Compute the normalised Shannon entropy score of an image.

    For colour images the entropy is computed per-channel and averaged.
    The result is divided by log₂(256) = 8 bits to give a [0,1] score.

    Args:
        image : uint8 numpy array (H, W, 3) or (H, W)

    Returns:
        float in [0, 1]  — higher means more complexity (→ more likely real)
    """
    if image.ndim == 2:
        h   = _shannon(_histogram(image))
        return float(min(h / _MAX_ENTROPY_BITS, 1.0))

    entropies = [
        _shannon(_histogram(image[:, :, c]))
        for c in range(image.shape[2])
    ]
    avg = float(np.mean(entropies))
    return float(min(avg / _MAX_ENTROPY_BITS, 1.0))
