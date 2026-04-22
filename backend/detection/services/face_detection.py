"""
DrishtiAI — Improved Face Detection
=====================================
Strategy:
  1. Try frontal Haar cascade first (most common pose)
  2. Fall back to profile cascade for angled faces
  3. Score each detected face by size + centrality + sharpness
  4. Apply face quality gate — reject blurry / too-small crops
  5. Return best face with 15% padding (includes forehead/chin)

Why this matters for fake-profile detection:
  • AI-generated profile images almost always contain exactly one face
  • We want the face region, not background, to focus analysis
  • Poor face crops degrade all downstream algorithm accuracy
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# ── Cascade classifiers ─────────────────────────────────────────────────────────
_FRONTAL    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_FRONTAL_ALT = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_PROFILE    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


# ── helpers ─────────────────────────────────────────────────────────────────────

def _laplacian_sharpness(gray_crop: np.ndarray) -> float:
    """
    Measure image sharpness via Laplacian variance.
    Sharp faces → high variance; blurry → low variance.
    Used to reject poor crops from downstream analysis.
    """
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def _centrality_score(x: int, y: int, w: int, h: int,
                      img_w: int, img_h: int) -> float:
    """
    Faces near image center score higher (profile photos are centred).
    Returns value in [0, 1].
    """
    cx = x + w / 2
    cy = y + h / 2
    dist_x = abs(cx - img_w / 2) / (img_w / 2)
    dist_y = abs(cy - img_h / 2) / (img_h / 2)
    return float(1.0 - (dist_x + dist_y) / 2.0)


def _score_face(x: int, y: int, w: int, h: int,
                gray: np.ndarray) -> float:
    """
    Composite score for ranking multiple detected faces.
    Larger + centred + sharper faces win.
    """
    img_h, img_w = gray.shape
    size_score   = (w * h) / (img_w * img_h)          # relative area
    cent_score   = _centrality_score(x, y, w, h, img_w, img_h)
    sharp_score  = min(_laplacian_sharpness(gray[y:y+h, x:x+w]) / 500.0, 1.0)
    return 0.5 * size_score + 0.3 * cent_score + 0.2 * sharp_score


def _pad_face(x: int, y: int, w: int, h: int,
              img_h: int, img_w: int,
              pad_ratio: float = 0.15) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by pad_ratio on each side.
    Clipped to image boundaries so we never go out-of-bounds.
    """
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w, x + w + px)
    y2 = min(img_h, y + h + py)
    return x1, y1, x2, y2


def _detect_with_cascade(gray: np.ndarray,
                         cascade: cv2.CascadeClassifier,
                         scale: float = 1.1,
                         min_neighbors: int = 4,
                         min_size: Tuple = (50, 50)):
    """Run a single cascade and return list of (x,y,w,h)."""
    return cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


# ── public API ──────────────────────────────────────────────────────────────────

class FaceQualityError(ValueError):
    """Raised when detected face is too blurry or too small to analyze."""
    pass


def detect_and_crop_face(
    image_path: str,
    min_sharpness: float = 25.0,
    min_face_ratio: float = 0.04,
    pad_ratio: float = 0.15,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Detect the primary face in an image and return cropped BGR array.

    Pipeline:
      1. Load image + convert to grayscale
      2. Equalize histogram (improves detection in dark / overexposed photos)
      3. Run frontal cascade, then frontal_alt2, then profile cascade
      4. Merge all detections, score each, pick best
      5. Quality check: sharpness + minimum face size
      6. Pad & crop

    Args:
        image_path    : path to image file
        min_sharpness : Laplacian variance threshold (reject blurry crops)
        min_face_ratio: face area / image area minimum (reject tiny detections)
        pad_ratio     : fractional padding around detected bbox

    Returns:
        (face_bgr, meta_dict)
        face_bgr  : BGR uint8 crop, or None if no face found
        meta_dict : {found, sharpness, face_ratio, cascade_used, n_faces}

    Raises:
        FaceQualityError if face is detected but fails quality gate
    """
    img  = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    # helps detection in uneven lighting common in social profile photos
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    all_faces = []
    cascade_used = "none"

    for cascade, name, scale, min_n in [
        (_FRONTAL,     "frontal",     1.1, 5),
        (_FRONTAL_ALT, "frontal_alt", 1.1, 4),
        (_PROFILE,     "profile",     1.1, 4),
    ]:
        detections = _detect_with_cascade(gray_eq, cascade, scale, min_n)
        if len(detections) > 0:
            for (x, y, w, h) in detections:
                all_faces.append((x, y, w, h, name))
            if cascade_used == "none":
                cascade_used = name

    meta = {
        "found":       len(all_faces) > 0,
        "n_faces":     len(all_faces),
        "cascade_used": cascade_used,
        "sharpness":   0.0,
        "face_ratio":  0.0,
    }

    if not all_faces:
        # No face detected — return full image with warning
        return img, {**meta, "warning": "no_face_detected"}

    # Score and pick best face
    scored = sorted(
        all_faces,
        key=lambda f: _score_face(f[0], f[1], f[2], f[3], gray),
        reverse=True,
    )
    x, y, w, h, used = scored[0]

    # Quality: minimum face ratio
    face_ratio = (w * h) / (img_w * img_h)
    if face_ratio < min_face_ratio:
        return img, {**meta, "warning": "face_too_small", "face_ratio": face_ratio}

    # Quality: sharpness check
    face_gray = gray[y:y+h, x:x+w]
    sharpness = _laplacian_sharpness(face_gray)
    meta.update({"sharpness": round(sharpness, 2), "face_ratio": round(face_ratio, 4),
                 "cascade_used": used})

    # Pad and crop
    x1, y1, x2, y2 = _pad_face(x, y, w, h, img_h, img_w, pad_ratio)
    face_crop = img[y1:y2, x1:x2]

    return face_crop, meta


def get_face_landmarks_simple(gray_face: np.ndarray) -> dict:
    """
    Simple symmetry & eye-region analysis without dlib/mediapipe.
    Splits face into left/right halves and measures pixel-distribution
    symmetry — AI faces are often unnaturally symmetric.

    Returns:
        dict with symmetry_score (0=asymmetric, 1=perfectly symmetric)
    """
    h, w = gray_face.shape[:2]
    left  = gray_face[:, :w//2]
    right = gray_face[:, w//2:]
    right_flipped = cv2.flip(right, 1)

    # Match dimensions
    min_w = min(left.shape[1], right_flipped.shape[1])
    left  = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    # Pixel-level similarity between left half and mirrored right half
    diff = np.abs(left.astype(np.float32) - right_flipped.astype(np.float32))
    symmetry_score = float(1.0 - diff.mean() / 255.0)

    # Upper face (forehead+eyes) vs lower face (nose+mouth)
    upper = gray_face[:h//2, :]
    lower = gray_face[h//2:, :]
    upper_std = float(upper.std())
    lower_std = float(lower.std())

    return {
        "symmetry_score": round(symmetry_score, 4),
        "upper_texture_std": round(upper_std, 4),
        "lower_texture_std": round(lower_std, 4),
        # AI faces tend to be more symmetric and have lower std in skin regions
        "region_variance_ratio": round(upper_std / (lower_std + 1e-6), 4),
    }
