"""
DrishtiAI — Inference Pipeline v3 (Enhanced)
=============================================
Integrates the enhanced pipeline (8 additional prebuilt algorithm signals)
on top of the existing 7-signal hybrid system.

Total signals: 14
  DL (ResNet18):           40% of base weight → adjusted
  Frequency (DCT/FFT):     reduced to 10%
  LBP Texture:             reduced to 8%
  Color Statistics:        8%
  K-Means Variance:        5%
  Sobel Edge:              5%
  Shannon Entropy:         4%
  ── Enhanced signals (20% total weight) ──
  SSIM Texture:            4%
  HSV Skin Uniformity:     4%
  Laplacian Sharpness:     3%
  Chromatic Aberration:    3%
  Background Coherence:    3%
  Noise Pattern:           3%

Profile-photo specific post-processing:
  - Single face penalty if multiple faces detected
  - Face quality gating
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
from .face_detection import detect_and_crop_face, get_face_landmarks_simple

from .kmeans import kmeans_variance
from .edge import sobel_edge_score
from .entropy import image_entropy_score
from .frequency_analysis import frequency_analysis_score
from .lbp import lbp_texture_score
from .color_stats import color_stats_score
from .enhanced_pipeline import compute_enhanced_signals
from .decision_v3 import compute_decision_v3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _face_to_rgb(face_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)


def _resize_for_analysis(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def run_inference_v3(image_path: str, instance_id: int) -> dict:
    """
    Full enhanced hybrid inference pipeline v3.

    Returns dict with all scores, metadata, and API response.
    """
    t_start = time.time()

    # ── 1. Face detection ───────────────────────────────────────────────────
    face_bgr, face_meta = detect_and_crop_face(
        image_path,
        min_sharpness=15.0,
        min_face_ratio=0.02,
        pad_ratio=0.18,
    )

    face_rgb = _face_to_rgb(face_bgr)
    face_224 = _resize_for_analysis(face_rgb, size=224)

    # ── 2. Landmarks & symmetry ─────────────────────────────────────────────
    gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_224 = cv2.resize(gray_face, (224, 224))
    landmark_meta = get_face_landmarks_simple(gray_224)
    face_meta.update(landmark_meta)

    # ── 3. DL inference ─────────────────────────────────────────────────────
    model = load_model()
    try:
        tensor = preprocess_from_bgr(face_bgr)
    except Exception:
        tensor, _ = preprocess(image_path)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    p_real = float(probs[0].item())
    p_fake_dl = float(probs[1].item())
    dl_is_fake = p_fake_dl >= p_real
    dl_conf = p_fake_dl if dl_is_fake else p_real
    pred_class = 1 if dl_is_fake else 0

    # ── 4. Classical algorithms ─────────────────────────────────────────────
    face_bgr_224 = cv2.cvtColor(face_224, cv2.COLOR_RGB2BGR)

    try:
        km_var = kmeans_variance(image=face_224, k=8, max_iters=30, sample_size=3000)
    except Exception:
        km_var = 0.5

    try:
        edge_sc = sobel_edge_score(image=face_224, downsample_to=128)
    except Exception:
        edge_sc = 0.5

    try:
        entr_sc = image_entropy_score(image=face_224)
    except Exception:
        entr_sc = 0.5

    try:
        freq_result = frequency_analysis_score(image=face_224)
        freq_sc = freq_result["overall_score"]
    except Exception:
        freq_result = {"overall_score": 0.5, "hf_ratio": 0.5,
                       "grid_artifact": 0.0, "spectral_slope": -2.0}
        freq_sc = 0.5

    try:
        lbp_result = lbp_texture_score(image=face_bgr, P=8, R=1.0)
        lbp_realness = lbp_result["overall_score"]
    except Exception:
        lbp_result = {"overall_score": 0.5}
        lbp_realness = 0.5

    try:
        color_result = color_stats_score(image=face_bgr)
        color_realness = color_result["overall_score"]
    except Exception:
        color_result = {"overall_score": 0.5}
        color_realness = 0.5

    # ── 5. Enhanced prebuilt algorithm signals ──────────────────────────────
    try:
        enhanced = compute_enhanced_signals(face_bgr_224)
    except Exception as e:
        print(f"[Inference v3] Enhanced signals failed: {e}")
        enhanced = {
            "ssim_texture_fakeness": 0.5,
            "hsv_skin_uniformity_fakeness": 0.5,
            "laplacian_sharpness_realness": 0.5,
            "chromatic_aberration_realness": 0.5,
            "background_coherence_realness": 0.5,
            "noise_pattern_realness": 0.5,
            "face_quality": {"face_count": 1, "quality": 0.5, "single_face": True},
            "single_face_confidence": 1.0,
            "enhanced_fakeness_score": 0.5,
        }

    # ── 6. Grad-CAM ─────────────────────────────────────────────────────────
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
    except Exception as exc:
        print(f"[Inference v3] Grad-CAM failed: {exc}")

    # ── 7. Decision engine v3 ────────────────────────────────────────────────
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
        "lbp": lbp_result,
        "color": color_result,
        "enhanced": enhanced,
        "face_meta": face_meta,
    }

    return {
        "is_fake": result.verdict == "FAKE",
        "confidence_score": result.dl_confidence,
        "verdict": result.verdict,
        "decision_score": result.decision_score,
        "freq_score": freq_sc,
        "lbp_score": lbp_realness,
        "color_score": color_realness,
        "kmeans_variance": km_var,
        "edge_score": edge_sc,
        "entropy_score": entr_sc,
        "heatmap_path": heatmap_rel,
        "processing_time": processing_time,
        "_api_dict": api_dict,
    }
