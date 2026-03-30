"""
Sobel Edge Detection — Implemented from Scratch
================================================
Purpose: Measure edge strength to distinguish real vs AI-generated faces.

Insight for deepfake detection:
  Real images   → sharp, irregular edges from real scene complexity → HIGH score
  AI/Deepfake   → smooth, blurred, or unnaturally regular edges     → LOW score

Sobel kernels (3×3):
  Kx detects horizontal gradients (vertical edges)
  Ky detects vertical gradients   (horizontal edges)

  Kx = [[-1, 0, 1],    Ky = [[-1, -2, -1],
         [-2, 0, 2],          [ 0,  0,  0],
         [-1, 0, 1]]           [ 1,  2,  1]]

Gradient magnitude:  G = sqrt(Gx^2 + Gy^2)
Edge score         : mean(G) / max_possible_magnitude   → [0, 1]
"""

import numpy as np

# ── Sobel filter kernels ───────────────────────────────────────────────────────
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)


# ── helpers ────────────────────────────────────────────────────────────────────

def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB → grayscale using luminosity weights.
    Y = 0.2989·R + 0.5870·G + 0.1140·B   (matches human eye sensitivity)
    Output: float64 in [0, 1]
    """
    if image.ndim == 2:
        return image.astype(np.float64) / 255.0
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255.0


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D convolution using numpy stride tricks (vectorised, no for-loops).

    Steps:
      1. Flip kernel (convolution = correlation with flipped kernel)
      2. Build sliding-window view of the image patches using as_strided
      3. Element-wise multiply each patch by the kernel and sum → output pixel

    Output shape: (H - kH + 1,  W - kW + 1)  — valid mode, no padding
    """
    kh, kw = kernel.shape
    oh = img.shape[0] - kh + 1
    ow = img.shape[1] - kw + 1

    flipped = kernel[::-1, ::-1]

    # Build shape/strides for the 4-D view: (oh, ow, kh, kw)
    shape   = (oh, ow, kh, kw)
    strides = (img.strides[0], img.strides[1],
               img.strides[0], img.strides[1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    return np.einsum('ijkl,kl->ij', patches, flipped)   # (oh, ow)


# ── public API ─────────────────────────────────────────────────────────────────

def sobel_edge_score(image: np.ndarray, downsample_to: int = 128) -> float:
    """
    Compute Sobel edge strength score for a face image.

    Pipeline:
      image (RGB) → grayscale → optional downsample → Sobel Gx, Gy
      → magnitude = sqrt(Gx² + Gy²) → normalised mean → score

    Args:
        image         : uint8 numpy array (H, W, 3)
        downsample_to : max side length before processing (speed/memory)

    Returns:
        float in [0, 1]  — higher means stronger natural edges (→ more likely real)
    """
    # 1. grayscale
    gray = _rgb_to_gray(image)           # float64 [0,1]

    # 2. downsample (simple index-based resizing)
    h, w = gray.shape
    if max(h, w) > downsample_to:
        scale   = downsample_to / max(h, w)
        new_h   = max(3, int(h * scale))
        new_w   = max(3, int(w * scale))
        row_idx = np.linspace(0, h - 1, new_h).astype(int)
        col_idx = np.linspace(0, w - 1, new_w).astype(int)
        gray    = gray[np.ix_(row_idx, col_idx)]

    # 3. apply Sobel kernels
    try:
        gx = _convolve2d(gray, SOBEL_X)
        gy = _convolve2d(gray, SOBEL_Y)
    except Exception:
        # ultra-safe fallback: manual double-loop (slow but always works)
        kh, kw = 3, 3
        oh = gray.shape[0] - kh + 1
        ow = gray.shape[1] - kw + 1
        gx = np.zeros((oh, ow))
        gy = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                patch   = gray[i:i+kh, j:j+kw]
                gx[i,j] = np.sum(patch * SOBEL_X[::-1, ::-1])
                gy[i,j] = np.sum(patch * SOBEL_Y[::-1, ::-1])

    # 4. gradient magnitude and normalised score
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Maximum theoretical magnitude for a [0,1] image through the Sobel kernel
    # Each kernel sums to 4·sqrt(2) ≈ 5.657 in the worst case
    max_possible = 4.0 * np.sqrt(2)
    score        = float(np.mean(magnitude) / max_possible)
    return min(score, 1.0)
