"""
DrishtiAI — Local Binary Pattern (LBP) Texture Analysis
=========================================================
WHY LBP WORKS FOR AI FACE DETECTION:
  AI-generated faces exhibit unnaturally smooth skin texture because:
    1. The generative model learns averaged/interpolated skin patches
    2. The loss function penalises sharp local texture variation
    3. Post-processing in diffusion models smears fine detail

  LBP encodes local texture by comparing each pixel to its circular
  neighbours.  Real skin has rich micro-texture (pores, fine hairs, pores).
  AI skin lacks this → lower LBP entropy and different histogram shape.

ALGORITHM (Circular LBP — implemented from scratch):
  For each pixel p at (y, x):
    1. Sample P neighbours on a circle of radius R
       using bilinear interpolation (handles non-integer coordinates)
    2. Compare each neighbour to centre pixel:
         bit_k = 1 if neighbour_k >= p else 0
    3. Form binary code: sum(bit_k * 2^k) for k in 0..P-1
    4. Use UNIFORM LBP: codes with ≤2 bit transitions → pattern class
       Non-uniform → single "noise" class
  Result: LBP histogram over the face region

ANALYSIS:
  - Entropy of LBP histogram → low entropy = repetitive texture (AI signal)
  - Uniformity ratio → proportion of uniform patterns (AI has more uniform)
  - Texture richness → variance of LBP codes
"""

import numpy as np
import cv2


# ── Bilinear Interpolation ───────────────────────────────────────────────────────

def _bilinear_interp(image: np.ndarray, y: float, x: float) -> float:
    """
    Bilinear interpolation at fractional pixel position (y, x).
    Used to sample circular neighbours that don't fall on integer grid.
    """
    H, W = image.shape
    # Clamp to valid range
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    y0 = max(y0, 0)
    x0 = max(x0, 0)

    dy = y - y0
    dx = x - x0

    # 4-point bilinear blend
    top    = (1 - dx) * float(image[y0, x0]) + dx * float(image[y0, x1])
    bottom = (1 - dx) * float(image[y1, x0]) + dx * float(image[y1, x1])
    return (1 - dy) * top + dy * bottom


# ── Uniform LBP ─────────────────────────────────────────────────────────────────

def _is_uniform(code: int, P: int) -> bool:
    """
    A binary pattern is UNIFORM if it has at most 2 bit transitions.
    Circular transition: last bit wraps back to first bit.

    Uniform patterns correspond to edges, corners, flat regions.
    Non-uniform patterns are noise.
    """
    transitions = 0
    prev_bit = (code >> (P - 1)) & 1   # last bit wraps to compare with first
    for k in range(P):
        curr_bit = (code >> k) & 1
        if curr_bit != prev_bit:
            transitions += 1
        prev_bit = curr_bit
    return transitions <= 2


def _build_lbp_lookup(P: int) -> np.ndarray:
    """
    Pre-compute mapping: LBP code → uniform pattern index.
    Uniform patterns each get a unique index 0..n_uniform-1.
    All non-uniform patterns map to index n_uniform (last bin).

    This gives a histogram of size (n_uniform + 1).
    For P=8, there are 58 uniform patterns + 1 non-uniform bin = 59 bins.
    """
    lookup = np.zeros(2 ** P, dtype=np.int32)
    uniform_idx = 0
    for code in range(2 ** P):
        if _is_uniform(code, P):
            lookup[code] = uniform_idx
            uniform_idx += 1
        else:
            lookup[code] = uniform_idx   # will be overwritten each time
    # Assign all non-uniform codes to the last bin
    for code in range(2 ** P):
        if not _is_uniform(code, P):
            lookup[code] = uniform_idx
    return lookup, uniform_idx + 1   # (lookup table, n_bins)


# Pre-compute lookup for P=8 (most common setting)
_LBP_P8_LOOKUP, _LBP_P8_BINS = _build_lbp_lookup(P=8)


def _compute_lbp(gray: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """
    Compute circular LBP map for an entire grayscale image.

    Args:
        gray : (H, W) uint8 grayscale image
        P    : number of neighbours on circle (default 8)
        R    : radius of circle in pixels   (default 1.0)

    Returns:
        lbp_map : (H, W) int32 array of LBP uniform-pattern indices
    """
    H, W = gray.shape
    lookup, n_bins = _build_lbp_lookup(P)

    # Pre-compute neighbour offsets for the circular neighbourhood
    angles = [2 * np.pi * k / P for k in range(P)]
    dy     = [-R * np.sin(a) for a in angles]   # -sin because y axis is inverted
    dx     = [ R * np.cos(a) for a in angles]

    lbp_map = np.zeros((H, W), dtype=np.int32)
    pad = int(np.ceil(R)) + 1

    # Iterate over valid interior pixels (avoiding border)
    for y in range(pad, H - pad):
        for x in range(pad, W - pad):
            centre = float(gray[y, x])
            code = 0
            for k in range(P):
                ny = y + dy[k]
                nx = x + dx[k]
                neighbour = _bilinear_interp(gray, ny, nx)
                if neighbour >= centre:
                    code |= (1 << k)
            lbp_map[y, x] = lookup[code]

    return lbp_map, n_bins


def _lbp_histogram(lbp_map: np.ndarray, n_bins: int,
                   mask: np.ndarray = None) -> np.ndarray:
    """
    Compute normalised histogram of LBP codes.
    mask: optional binary mask (only include masked pixels)
    """
    if mask is not None:
        codes = lbp_map[mask > 0]
    else:
        codes = lbp_map.flatten()
    # Exclude border zeros
    codes = codes[codes > 0]
    hist = np.bincount(codes, minlength=n_bins).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _histogram_entropy(hist: np.ndarray) -> float:
    """Shannon entropy of a normalised histogram."""
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero + 1e-12)))


# ── Public API ───────────────────────────────────────────────────────────────────

def lbp_texture_score(image: np.ndarray, P: int = 8, R: float = 1.0) -> dict:
    """
    Compute LBP-based texture richness score for a face image.

    For AI face detection:
      Real faces  → high LBP entropy, diverse histogram, low uniformity ratio
      AI faces    → low LBP entropy, histogram peaks on uniform patterns

    Args:
        image : uint8 RGB array (H, W, 3)
        P     : number of LBP neighbours (default 8)
        R     : sampling radius          (default 1.0)

    Returns dict:
        entropy          : LBP histogram entropy (higher = richer texture)
        uniformity_ratio : fraction of uniform LBP codes (higher = smoother/more AI)
        texture_variance : variance of LBP map values
        overall_score    : 0-1 "realness" score (higher = more natural texture)
    """
    # Work on smaller image for speed (LBP pixel loop is O(H*W))
    h, w = image.shape[:2]
    scale = 96.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    lbp_map, n_bins = _compute_lbp(gray, P=P, R=R)
    hist = _lbp_histogram(lbp_map, n_bins)

    entropy = _histogram_entropy(hist)
    max_entropy = np.log2(n_bins)

    # Uniform patterns: bins 0 to n_bins-2; non-uniform: last bin
    uniform_ratio = float(hist[:-1].sum())
    texture_var   = float(lbp_map[lbp_map > 0].var()) if lbp_map.any() else 0.0

    # Normalised entropy: higher → richer texture → more likely real
    norm_entropy = float(entropy / (max_entropy + 1e-9))

    # Combine into overall "realness" score
    # High entropy + low uniformity ratio + high variance → real
    realness = (0.5 * norm_entropy +
                0.3 * (1.0 - uniform_ratio) +
                0.2 * min(texture_var / 1000.0, 1.0))
    realness = float(min(max(realness, 0.0), 1.0))

    return {
        "entropy":          round(entropy, 4),
        "uniformity_ratio": round(uniform_ratio, 4),
        "texture_variance": round(texture_var, 4),
        "overall_score":    round(realness, 4),
    }
