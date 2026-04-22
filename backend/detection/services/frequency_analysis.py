"""
DrishtiAI — DCT Frequency Domain Analysis
==========================================
WHY FREQUENCY ANALYSIS WORKS FOR AI FACE DETECTION:
  GAN / Diffusion models generate images in ways that leave spectral
  fingerprints invisible to the human eye but measurable in the frequency
  domain.  Specifically:
    - Real photographs have 1/f power spectrum (energy ∝ 1/frequency)
    - GAN-generated images show periodic spectral peaks at multiples of
      training resolution (e.g., 64×64 grid artefacts in the DCT spectrum)
    - Upsampling artefacts from deconvolutional layers create regularly
      spaced high-frequency energy spikes

This module implements:
  1. 2D DCT from scratch using the analytical separable formula
  2. Power spectral density (PSD) computation
  3. High-frequency energy ratio  ←  key fake signal
  4. Azimuthal PSD averaging for rotation-invariant spectrum
  5. Grid artefact score (detects periodic GAN upsampling patterns)

Reference insight: Wang et al. (2020) "CNN-Generated Images Are Surprisingly
Easy to Spot...For Now" — frequency domain features generalize across models.
"""

import numpy as np
from typing import Tuple


# ── DCT Implementation ──────────────────────────────────────────────────────────

def _dct1d(signal: np.ndarray) -> np.ndarray:
    """
    1D Type-II DCT implemented analytically.

    DCT-II formula:
      X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1) / (2N))

    where the k=0 term uses factor sqrt(1/N) and k>0 terms use sqrt(2/N)
    (orthonormal form).

    This is O(N²) — suitable for small 1D signals.
    The signal is the per-row or per-column slice.
    """
    N  = len(signal)
    k  = np.arange(N)
    n  = np.arange(N)
    # cos matrix: shape (N, N)
    cos_mat = np.cos(np.pi * np.outer(k, 2 * n + 1) / (2 * N))
    X = 2.0 * (cos_mat @ signal.astype(np.float64))

    # Orthonormal scaling
    X[0]  *= np.sqrt(1.0 / (4 * N))
    X[1:] *= np.sqrt(1.0 / (2 * N))
    return X


def _dct2d(block: np.ndarray) -> np.ndarray:
    """
    2D DCT via separable 1D transforms (row then column).
    Input:  (H, W) float array
    Output: (H, W) DCT coefficient matrix
    """
    # Apply 1D DCT to each row
    row_dct = np.apply_along_axis(_dct1d, axis=1, arr=block)
    # Apply 1D DCT to each column of the row-transformed matrix
    return np.apply_along_axis(_dct1d, axis=0, arr=row_dct)


def _fast_dct2d(block: np.ndarray) -> np.ndarray:
    """
    Fast approximation using numpy FFT-based DCT.
    We use the identity: DCT-II of a signal x equals
    the real part of the FFT of the zero-extended signal,
    properly phase-rotated.

    This is O(N log N) and used for larger blocks.
    """
    block = block.astype(np.float64)
    H, W = block.shape

    # Row-wise DCT via FFT
    def dct_via_fft(x):
        N = len(x)
        # Extend: [x[0], x[1], ..., x[N-1], x[N-1], ..., x[0]]
        v = np.concatenate([x, x[::-1]])
        V = np.fft.rfft(v)
        k = np.arange(len(V))
        W_k = np.exp(-1j * np.pi * k / (2 * N))
        X = np.real(V * W_k)[:N]
        X[0]  *= np.sqrt(1.0 / (4 * N))
        X[1:] *= np.sqrt(1.0 / (2 * N))
        return X

    row_dct = np.apply_along_axis(dct_via_fft, axis=1, arr=block)
    return np.apply_along_axis(dct_via_fft, axis=0, arr=row_dct)


# ── Frequency Analysis ──────────────────────────────────────────────────────────

def _rgb_to_ycbcr_y(image: np.ndarray) -> np.ndarray:
    """
    Extract luminance channel (Y) from RGB image.
    Y captures structural frequency content without color bias.
    """
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _compute_magnitude_spectrum(channel: np.ndarray) -> np.ndarray:
    """
    Compute 2D power spectrum using numpy FFT (efficient).
    Shifted so DC is at centre.
    Returns log-scaled magnitude for visualisation convenience.
    """
    F = np.fft.fft2(channel)
    F_shift = np.fft.fftshift(F)
    magnitude = np.abs(F_shift) ** 2   # power spectrum
    return magnitude


def _radial_profile(spectrum: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    Compute azimuthal (radial) average of power spectrum.
    Reduces 2D spectrum to 1D frequency profile — rotation invariant.

    For each radial distance r from centre, averages all spectrum values
    within that ring.  This collapses orientation dependence.
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    y, x   = np.ogrid[:H, :W]
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    bins  = np.linspace(0, max_r, n_bins + 1, dtype=int)
    profile = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() > 0:
            profile[i] = float(spectrum[mask].mean())

    return profile


def _hf_energy_ratio(spectrum: np.ndarray, low_cutoff: float = 0.3) -> float:
    """
    Ratio of high-frequency energy to total energy in the spectrum.

    AI-generated images tend to have LESS high-frequency energy
    (smoother textures) OR anomalous peaks at specific frequencies
    due to upsampling layers.

    low_cutoff: fraction of max radius below which we call it "low frequency"
    Returns: float in [0, 1]  — higher means more HF energy (→ more natural)
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    y, x   = np.ogrid[:H, :W]
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r  = min(cx, cy)

    lf_mask = r <= (low_cutoff * max_r)
    hf_mask = r >  (low_cutoff * max_r)

    total_energy = float(spectrum.sum()) + 1e-12
    hf_energy    = float(spectrum[hf_mask].sum())
    return float(hf_energy / total_energy)


def _grid_artifact_score(spectrum: np.ndarray, grid_size: int = 8) -> float:
    """
    Detect periodic grid artefacts from GAN upsampling layers.

    GAN generators built from repeated 4×4 or 8×8 conv-transpose blocks
    produce regular spectral peaks at multiples of (image_size/block_size).
    We look for unexpectedly high energy at those exact frequency bins.

    Returns: float [0, 1] — higher means stronger grid artefacts (→ more fake)
    """
    H, W = spectrum.shape
    # Expected artefact frequencies as fractions of image size
    # e.g., for 128×128 with 8×8 blocks → peaks at 128/8=16 pixel intervals
    expected_freqs = [W // (grid_size * k) for k in range(1, 5) if W // (grid_size * k) > 0]

    cy, cx = H // 2, W // 2
    scores = []
    for f in expected_freqs:
        # Check energy at frequency f in both horizontal and vertical
        # Compare to average energy at neighbouring frequencies
        f_clipped = min(f, cx - 1)
        target_h  = spectrum[cy, cx + f_clipped] if cx + f_clipped < W else 0
        target_v  = spectrum[cy + f_clipped, cx] if cy + f_clipped < H else 0

        # Local average (excluding the target bin)
        window = max(1, f_clipped // 2)
        lo, hi = max(cx, cx + f_clipped - window), min(W, cx + f_clipped + window + 1)
        local_mean = float(spectrum[cy, lo:hi].mean()) + 1e-12
        scores.append(float(max(target_h, target_v) / local_mean))

    if not scores:
        return 0.0
    # Normalise: score > 1 means energy spike above local average
    raw = float(np.mean(scores))
    return float(min((raw - 1.0) / 10.0, 1.0)) if raw > 1.0 else 0.0


def _spectral_slope(radial_profile: np.ndarray) -> float:
    """
    Fit a line to log-log plot of radial spectrum profile.
    Real photographs follow 1/f^α law (α ≈ 2 for natural scenes).
    AI images deviate from this — slope too flat or irregular.

    Returns: negative slope (steeper is more natural).
    """
    n = len(radial_profile)
    freqs = np.arange(1, n + 1, dtype=np.float64)
    power = radial_profile + 1e-12

    log_f = np.log(freqs)
    log_p = np.log(power)

    # Linear regression on log-log
    A = np.vstack([log_f, np.ones(n)]).T
    slope, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
    return float(slope)


# ── Public API ──────────────────────────────────────────────────────────────────

def frequency_analysis_score(image: np.ndarray) -> dict:
    """
    Full frequency domain analysis for deepfake/AI face detection.

    Args:
        image : uint8 RGB numpy array (H, W, 3)

    Returns dict with:
        hf_ratio      : high-frequency energy ratio  [0,1]  (low → fake signal)
        grid_artifact : periodic GAN upsampling score [0,1]  (high → fake signal)
        spectral_slope: log-log slope                        (near 0 → flat → fake)
        overall_score : 0-1 "fakeness" signal from frequency domain alone
    """
    # Work on luminance
    lum = _rgb_to_ycbcr_y(image)

    # Resize to power-of-2 for stable FFT (128×128 is sufficient)
    import cv2
    lum_resized = cv2.resize(lum, (128, 128), interpolation=cv2.INTER_AREA)

    spectrum = _compute_magnitude_spectrum(lum_resized)
    profile  = _radial_profile(spectrum, n_bins=32)

    hf_ratio   = _hf_energy_ratio(spectrum, low_cutoff=0.3)
    grid_score = _grid_artifact_score(spectrum, grid_size=8)
    slope      = _spectral_slope(profile)

    # Natural images: hf_ratio ~0.3-0.5, slope ~ -2 to -3, grid_score ~0
    # AI images:      hf_ratio <0.2 or >0.6, slope near 0, grid_score >0

    # Fakeness signal per feature
    hf_fake    = 1.0 - min(max(hf_ratio / 0.5, 0), 1.0)   # low HF → fake
    slope_fake = min(max((slope + 1.0) / 2.0, 0), 1.0)     # slope near 0 → fake
    grid_fake  = grid_score                                  # directly a fake signal

    overall = 0.4 * hf_fake + 0.3 * slope_fake + 0.3 * grid_fake
    overall = float(min(max(overall, 0.0), 1.0))

    return {
        "hf_ratio":       round(hf_ratio, 4),
        "grid_artifact":  round(grid_score, 4),
        "spectral_slope": round(slope, 4),
        "overall_score":  round(overall, 4),
    }
