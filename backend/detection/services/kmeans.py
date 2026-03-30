"""
K-Means Clustering — Implemented from Scratch
==============================================
Purpose : Measure pixel diversity of an uploaded face image.

Insight for deepfake detection:
  Real images  → diverse pixel clusters  → HIGH variance score
  Fake/AI images → unnaturally uniform textures → LOW variance score

Algorithm steps:
  1. Flatten image pixels to (N, C) array
  2. Randomly initialise k centroids
  3. Assign every pixel to the nearest centroid (Euclidean distance)
  4. Recompute centroids as the mean of each cluster
  5. Repeat until convergence (centroid shift < tol)
  6. Compute within-cluster sum of squared distances (inertia)
  7. Normalise inertia → score in [0, 1]
"""

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def _init_centroids(pixels: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Pick k random pixels as initial centroids."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=k, replace=False)
    return pixels[idx].astype(np.float64)


def _assign(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each pixel to its nearest centroid.

    Uses broadcasting:
      pixels    shape (N, C)  →  (N, 1, C)
      centroids shape (k, C)  →  (1, k, C)
      diff      shape (N, k, C)
    """
    diff    = pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (N, k, C)
    sq_dist = np.sum(diff ** 2, axis=2)                                 # (N, k)
    return np.argmin(sq_dist, axis=1)                                   # (N,)


def _update(pixels: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Move each centroid to the mean of its assigned pixels."""
    new_c = np.zeros((k, pixels.shape[1]), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            new_c[c] = pixels[mask].mean(axis=0)
    return new_c


def _inertia(pixels: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Within-cluster sum of squared distances (WCSS / inertia).
    Lower → tighter, more uniform clusters (suspicious for deepfake).
    Higher → spread-out clusters (natural image diversity).
    """
    total = 0.0
    for i in range(len(pixels)):
        diff   = pixels[i] - centroids[labels[i]]
        total += float(np.dot(diff, diff))
    return total


# ── public API ─────────────────────────────────────────────────────────────────

def kmeans_variance(
    image:       np.ndarray,
    k:           int   = 8,
    max_iters:   int   = 30,
    tol:         float = 1e-4,
    sample_size: int   = 3000,
    seed:        int   = 42,
) -> float:
    """
    Run K-Means on image pixels and return a normalised diversity score.

    Args:
        image       : uint8 numpy array (H, W, C)
        k           : number of clusters  (default 8)
        max_iters   : iteration cap       (default 30)
        tol         : convergence threshold (default 1e-4)
        sample_size : pixels to subsample for speed (default 3 000)
        seed        : reproducibility seed

    Returns:
        float in [0, 1] — higher means more pixel diversity (→ more likely real)
    """
    # 1. flatten + normalise to [0,1]
    pixels = image.reshape(-1, image.shape[2]).astype(np.float64) / 255.0

    # 2. subsample
    n = len(pixels)
    if n > sample_size:
        rng = np.random.default_rng(seed)
        pixels = pixels[rng.choice(n, size=sample_size, replace=False)]

    # 3. run k-means
    centroids = _init_centroids(pixels, k, seed)
    labels    = np.zeros(len(pixels), dtype=int)

    for _ in range(max_iters):
        new_labels    = _assign(pixels, centroids)
        new_centroids = _update(pixels, new_labels, k)
        shift         = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids     = new_centroids
        labels        = new_labels
        if shift < tol:
            break

    # 4. compute and normalise inertia
    raw      = _inertia(pixels, labels, centroids)
    n_px, c  = pixels.shape
    max_val  = n_px * c * 1.0          # maximum possible value when all pixels at corners
    score    = float(min(raw / (max_val + 1e-8), 1.0))
    return score
