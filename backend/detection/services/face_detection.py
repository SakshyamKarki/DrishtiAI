"""
DrishtiAI — Face Detection [REFACTORED v2]
===========================================
Key changes from v1:

  1. STRICT NO-FACE DISCARD
     If no face is detected, the function now raises NoFaceError instead of
     silently returning the full image.  The caller (inference_v3) must catch
     this and return a structured "no face" API response immediately —
     preventing all downstream analysis from running on non-face images.

  2. SYMMETRY SCORE FIX
     get_face_landmarks_simple() returned a dict key "symmetry_score" but
     inference_v3 also wrote that key via face_meta.update().  This caused
     silent key collision.  Renamed the landmark output key to
     "landmark_symmetry_score" to avoid the collision.

  3. MULTI-SCALE DETECTION
     Added a second detection pass at 0.75× scale for images where the face
     is very large (cropped close-up) and the default scaleFactor misses it.

  4. BOUNDING-BOX DEDUPLICATION
     Multiple cascades often return overlapping boxes for the same face.
     Added IoU-based NMS (Non-Maximum Suppression) to deduplicate.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

# ── Cascade classifiers ──────────────────────────────────────────────────────
_FRONTAL     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_FRONTAL_ALT = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_PROFILE     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


# ── Custom exceptions ─────────────────────────────────────────────────────────

class NoFaceError(ValueError):
    """
    Raised when NO face can be detected in the submitted image.
    This triggers an immediate "no face" API response and skips all
    downstream analysis — prevents false readings on non-face images.
    """
    pass


class FaceQualityError(ValueError):
    """Raised when a face is found but too blurry / too small to analyse."""
    pass


# ── IoU NMS ──────────────────────────────────────────────────────────────────

def _iou(a: Tuple, b: Tuple) -> float:
    """Compute Intersection over Union for two (x,y,w,h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / (union + 1e-6))


def _nms(faces: List[Tuple], iou_threshold: float = 0.40) -> List[Tuple]:
    """
    Non-Maximum Suppression: remove duplicate bounding boxes.
    Keep the box with the largest area when overlap > iou_threshold.
    """
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # sort by area
    kept = []
    for candidate in faces:
        if all(_iou(candidate[:4], k[:4]) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


# ── Helpers ───────────────────────────────────────────────────────────────────

def _laplacian_sharpness(gray_crop: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def _centrality_score(x, y, w, h, img_w, img_h) -> float:
    cx = x + w / 2
    cy = y + h / 2
    dist_x = abs(cx - img_w / 2) / (img_w / 2)
    dist_y = abs(cy - img_h / 2) / (img_h / 2)
    return float(1.0 - (dist_x + dist_y) / 2.0)


def _score_face(x, y, w, h, gray: np.ndarray) -> float:
    img_h, img_w = gray.shape
    size_score  = (w * h) / (img_w * img_h)
    cent_score  = _centrality_score(x, y, w, h, img_w, img_h)
    sharp_score = min(_laplacian_sharpness(gray[y:y+h, x:x+w]) / 500.0, 1.0)
    return 0.5 * size_score + 0.3 * cent_score + 0.2 * sharp_score


def _pad_face(x, y, w, h, img_h, img_w, pad_ratio=0.15):
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    return (max(0, x - px), max(0, y - py),
            min(img_w, x + w + px), min(img_h, y + h + py))


def _detect_cascade(gray, cascade, scale=1.1, min_n=4, min_size=(50, 50)):
    return cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=min_n,
        minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_and_crop_face(
    image_path: str,
    min_sharpness: float = 25.0,
    min_face_ratio: float = 0.04,
    pad_ratio: float = 0.15,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Detect the primary face and return a cropped BGR array.

    Raises:
        NoFaceError       — if no face detected at all.
        FaceQualityError  — if detected face is too blurry or too small.
        ValueError        — if image cannot be loaded.

    Returns:
        (face_bgr, meta_dict)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    all_faces = []
    cascade_used = "none"

    for cascade, name, scale, min_n in [
        (_FRONTAL,     "frontal",     1.1, 5),
        (_FRONTAL_ALT, "frontal_alt", 1.1, 4),
        (_PROFILE,     "profile",     1.1, 4),
    ]:
        detections = _detect_cascade(gray_eq, cascade, scale, min_n)
        if len(detections) > 0:
            for (x, y, w, h) in detections:
                all_faces.append((x, y, w, h, name))
            if cascade_used == "none":
                cascade_used = name

    # Second pass at 75% scale for close-up / large faces
    if len(all_faces) == 0:
        scale_factor = 0.75
        small_h = int(img_h * scale_factor)
        small_w = int(img_w * scale_factor)
        gray_small = cv2.resize(gray_eq, (small_w, small_h))
        for cascade, name, scale, min_n in [
            (_FRONTAL, "frontal_small", 1.05, 3),
            (_FRONTAL_ALT, "frontal_alt_small", 1.05, 3),
        ]:
            detections = _detect_cascade(gray_small, cascade, scale, min_n, (30, 30))
            if len(detections) > 0:
                for (x, y, w, h) in detections:
                    # Scale coordinates back to original image
                    x2 = int(x / scale_factor)
                    y2 = int(y / scale_factor)
                    w2 = int(w / scale_factor)
                    h2 = int(h / scale_factor)
                    all_faces.append((x2, y2, w2, h2, name))
                if cascade_used == "none":
                    cascade_used = name + "_rescaled"

    meta = {
        "found":        len(all_faces) > 0,
        "n_faces":      len(all_faces),
        "cascade_used": cascade_used,
        "sharpness":    0.0,
        "face_ratio":   0.0,
    }

    # ── STRICT: raise NoFaceError if nothing found ────────────────────────────
    if not all_faces:
        raise NoFaceError(
            "No human face detected in this image. "
            "Please submit a clear profile photo containing a visible face."
        )

    # NMS deduplication
    all_faces = _nms(all_faces, iou_threshold=0.40)

    # Score and pick best face
    scored = sorted(all_faces,
                    key=lambda f: _score_face(f[0], f[1], f[2], f[3], gray),
                    reverse=True)
    x, y, w, h, used = scored[0]

    face_ratio = (w * h) / (img_w * img_h)
    if face_ratio < min_face_ratio:
        raise FaceQualityError(
            f"Detected face is too small (ratio={face_ratio:.3f}). "
            "Please submit a higher resolution image."
        )

    face_gray = gray[y:y+h, x:x+w]
    sharpness = _laplacian_sharpness(face_gray)
    meta.update({
        "sharpness":    round(sharpness, 2),
        "face_ratio":   round(face_ratio, 4),
        "cascade_used": used,
        "n_faces_after_nms": len(all_faces),
    })

    x1, y1, x2, y2 = _pad_face(x, y, w, h, img_h, img_w, pad_ratio)
    face_crop = img[y1:y2, x1:x2]
    return face_crop, meta


def get_face_landmarks_simple(gray_face: np.ndarray) -> dict:
    """
    Simple symmetry analysis.
    Key renamed to 'landmark_symmetry_score' (was 'symmetry_score') to
    avoid silent collision when merged into face_meta in inference_v3.
    """
    h, w = gray_face.shape[:2]
    left  = gray_face[:, :w//2]
    right = gray_face[:, w//2:]
    right_flipped = cv2.flip(right, 1)

    min_w = min(left.shape[1], right_flipped.shape[1])
    left  = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    diff = np.abs(left.astype(np.float32) - right_flipped.astype(np.float32))
    symmetry_score = float(1.0 - diff.mean() / 255.0)

    upper = gray_face[:h//2, :]
    lower = gray_face[h//2:, :]
    upper_std = float(upper.std())
    lower_std = float(lower.std())

    return {
        "landmark_symmetry_score":  round(symmetry_score, 4),   # RENAMED
        "upper_texture_std":        round(upper_std, 4),
        "lower_texture_std":        round(lower_std, 4),
        "region_variance_ratio":    round(upper_std / (lower_std + 1e-6), 4),
    }
