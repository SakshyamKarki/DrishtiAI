"""
DrishtiAI — Inference Pipeline v3 [REFACTORED]
===============================================
Key changes:
  1. NoFaceError now exits immediately with a structured API response.
     No downstream analysis runs on non-face images.
  2. FaceQualityError similarly returns a clean API response.
  3. face_meta.update(landmark_meta) no longer clobbers "symmetry_score"
     because face_detection.py now uses "landmark_symmetry_score".
  4. kmeans_elbow_signal() wired in as an additional fakeness signal.
  5. All except blocks log the actual exception (not silent pass).
  6. Removed duplicate "confidence_score" / "dl_confidence" confusion:
     top-level "confidence_score" now always stores the 0-100 DL confidence.
"""

import os
import time
import torch
import cv2
import numpy as np
from django.conf import settings

from .model_loader import load_model
from .preprocess import preprocess_from_bgr, preprocess
from .gradcam import generate_cam
from .face_detection import (
    detect_and_crop_face, get_face_landmarks_simple,
    NoFaceError, FaceQualityError,
)

from .kmeans import kmeans_variance, kmeans_elbow_signal
from .edge import sobel_edge_score
from .entropy import image_entropy_score
from .frequency_analysis import frequency_analysis_score
from .lbp import lbp_texture_score
from .color_stats import color_stats_score
from .enhanced_pipeline import compute_enhanced_signals
from .decision_v3 import compute_decision_v3

import logging
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _face_to_rgb(face_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)


def _resize_for_analysis(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def _no_face_response(reason: str, processing_time: float) -> dict:
    """
    Structured API response when no usable face is found.
    Returned immediately — no inference is run.
    """
    return {
        "is_fake":          False,
        "verdict":          "NO_FACE",
        "final_label":      "NO_FACE",
        "confidence_score": 0.0,
        "decision_score":   0.0,
        "risk_label":       "Cannot Analyse — No Face Detected",
        "message":          reason,
        "processing_time":  processing_time,
        "profile_analysis": {
            "face_detected": False,
            "face_count":    0,
        },
        "_api_dict": {
            "verdict":      "NO_FACE",
            "is_fake":      False,
            "confidence":   0.0,
            "risk_label":   "Cannot Analyse — No Face Detected",
            "message":      reason,
            "processing_time": processing_time,
        },
    }


# ── Main inference function ───────────────────────────────────────────────────

def run_inference_v3(image_path: str, instance_id: int) -> dict:
    """
    Full enhanced hybrid inference pipeline v3.
    Returns dict suitable for Django view serialisation.
    """
    t_start = time.time()

    # ── 1. Face detection — strict mode ──────────────────────────────────────
    try:
        face_bgr, face_meta = detect_and_crop_face(
            image_path,
            min_sharpness=15.0,
            min_face_ratio=0.02,
            pad_ratio=0.18,
        )
    except NoFaceError as e:
        elapsed = round(time.time() - t_start, 3)
        logger.info(f"[Inference v3] NoFaceError for {image_path}: {e}")
        return _no_face_response(str(e), elapsed)
    except FaceQualityError as e:
        elapsed = round(time.time() - t_start, 3)
        logger.info(f"[Inference v3] FaceQualityError for {image_path}: {e}")
        return _no_face_response(str(e), elapsed)
    except Exception as e:
        elapsed = round(time.time() - t_start, 3)
        logger.error(f"[Inference v3] Face detection crashed for {image_path}: {e}")
        return _no_face_response(
            "Face detection encountered an unexpected error.", elapsed)

    face_rgb = _face_to_rgb(face_bgr)
    face_224 = _resize_for_analysis(face_rgb, size=224)

    # ── 2. Landmarks & symmetry ───────────────────────────────────────────────
    gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_224  = cv2.resize(gray_face, (224, 224))
    landmark_meta = get_face_landmarks_simple(gray_224)
    face_meta.update(landmark_meta)   # safe: no key collision after rename

    # ── 3. DL inference ───────────────────────────────────────────────────────
    model = load_model()
    try:
        tensor = preprocess_from_bgr(face_bgr)
    except Exception as e:
        logger.warning(f"[Inference v3] preprocess_from_bgr failed ({e}), using fallback")
        tensor, _ = preprocess(image_path)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    p_real    = float(probs[0].item())
    p_fake_dl = float(probs[1].item())
    dl_is_fake = p_fake_dl >= p_real
    dl_conf    = p_fake_dl if dl_is_fake else p_real
    pred_class = 1 if dl_is_fake else 0

    # ── 4. Classical algorithms ───────────────────────────────────────────────
    face_bgr_224 = cv2.cvtColor(face_224, cv2.COLOR_RGB2BGR)

    try:
        km_var = kmeans_variance(image=face_224, k=8, max_iters=30, sample_size=3000)
    except Exception as e:
        logger.warning(f"[Inference v3] kmeans_variance failed: {e}")
        km_var = 0.5

    # Elbow signal (new hardcoded algorithm)
    try:
        km_elbow_fake = kmeans_elbow_signal(image=face_224, sample_size=2000)
    except Exception as e:
        logger.warning(f"[Inference v3] kmeans_elbow_signal failed: {e}")
        km_elbow_fake = 0.5

    try:
        edge_sc = sobel_edge_score(image=face_224, downsample_to=128)
    except Exception as e:
        logger.warning(f"[Inference v3] sobel_edge_score failed: {e}")
        edge_sc = 0.5

    try:
        entr_sc = image_entropy_score(image=face_224)
    except Exception as e:
        logger.warning(f"[Inference v3] image_entropy_score failed: {e}")
        entr_sc = 0.5

    try:
        freq_result = frequency_analysis_score(image=face_224)
        freq_sc = freq_result["overall_score"]
    except Exception as e:
        logger.warning(f"[Inference v3] frequency_analysis_score failed: {e}")
        freq_result = {"overall_score": 0.5, "hf_ratio": 0.5,
                       "grid_artifact": 0.0, "spectral_slope": -2.0}
        freq_sc = 0.5

    try:
        lbp_result    = lbp_texture_score(image=face_bgr, P=8, R=1.0)
        lbp_realness  = lbp_result["overall_score"]
    except Exception as e:
        logger.warning(f"[Inference v3] lbp_texture_score failed: {e}")
        lbp_result   = {"overall_score": 0.5}
        lbp_realness = 0.5

    try:
        color_result    = color_stats_score(image=face_bgr)
        color_realness  = color_result["overall_score"]
    except Exception as e:
        logger.warning(f"[Inference v3] color_stats_score failed: {e}")
        color_result   = {"overall_score": 0.5}
        color_realness = 0.5

    # ── 5. Enhanced prebuilt algorithm signals ────────────────────────────────
    try:
        enhanced = compute_enhanced_signals(face_bgr_224)
    except Exception as e:
        logger.error(f"[Inference v3] Enhanced signals failed: {e}")
        enhanced = {
            "ssim_texture_fakeness":           0.5,
            "hsv_skin_uniformity_fakeness":    0.5,
            "laplacian_sharpness_realness":    0.5,
            "chromatic_aberration_realness":   0.5,
            "background_coherence_realness":   0.5,
            "noise_pattern_realness":          0.5,
            "face_quality": {"face_count": 1, "quality": 0.5, "single_face": True},
            "single_face_confidence":          1.0,
            "enhanced_fakeness_score":         0.5,
        }

    # Inject elbow fakeness into enhanced signals so decision_v3 can use it
    enhanced["kmeans_elbow_fakeness"] = km_elbow_fake

    # ── 6. Grad-CAM ───────────────────────────────────────────────────────────
    heatmap_rel = None
    try:
        heatmap_filename = f"heatmap_{instance_id}.jpg"
        heatmap_abs = os.path.join(settings.MEDIA_ROOT, "heatmaps", heatmap_filename)
        generate_cam(
            model=model,
            tensor=tensor,
            target_layer=model.layer4[-1],
            target_class=pred_class,
            save_path=heatmap_abs,
        )
        heatmap_rel = f"heatmaps/{heatmap_filename}"
    except Exception as e:
        logger.warning(f"[Inference v3] Grad-CAM failed: {e}")

    # ── 7. Decision engine v3 ─────────────────────────────────────────────────
    result = compute_decision_v3(
        dl_is_fake=dl_is_fake,
        dl_confidence=dl_conf,
        freq_score=freq_sc,
        lbp_realness=lbp_realness,
        color_realness=color_realness,
        kmeans_variance=km_var,
        edge_score=edge_sc,
        entropy_score=entr_sc,
        enhanced_signals=enhanced,
        face_meta=face_meta,
    )

    processing_time = round(time.time() - t_start, 3)

    api_dict = result.to_api_dict()
    api_dict["processing_time"] = processing_time
    api_dict["detailed_analysis"] = {
        "frequency": freq_result,
        "lbp":       lbp_result,
        "color":     color_result,
        "enhanced":  enhanced,
        "face_meta": face_meta,
        "kmeans_elbow_fakeness": km_elbow_fake,
    }

    return {
        "is_fake":           result.verdict == "FAKE",
        "confidence_score":  round(result.dl_confidence, 2),   # 0-100 DL confidence
        "verdict":           result.verdict,
        "decision_score":    result.decision_score,
        "freq_score":        freq_sc,
        "lbp_score":         lbp_realness,
        "color_score":       color_realness,
        "kmeans_variance":   km_var,
        "kmeans_elbow_fake": km_elbow_fake,
        "edge_score":        edge_sc,
        "entropy_score":     entr_sc,
        "heatmap_path":      heatmap_rel,
        "processing_time":   processing_time,
        "_api_dict":         api_dict,
    }
