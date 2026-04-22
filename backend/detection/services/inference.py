"""
DrishtiAI — Improved Inference Pipeline v2
============================================
Full end-to-end pipeline for fake profile photo detection:

  1. Load image + run improved face detection
  2. Preprocess (crop, resize, normalise for DL)
  3. ResNet18 forward pass (GPU if available)
  4. Classical algorithms (all from scratch):
       a. K-Means clustering variance
       b. Sobel edge detection
       c. Shannon entropy
       d. DCT frequency analysis  ← NEW
       e. LBP texture analysis    ← NEW
       f. Color statistics        ← NEW
  5. Grad-CAM heatmap
  6. Hybrid decision engine v2 (7 signals)
  7. Return structured result dict

Performance notes:
  - LBP uses downscaled image (96×96) for speed
  - Frequency analysis uses 128×128 crop
  - K-Means uses 3000 pixel sample
  - All operations are CPU-friendly, GPU only for DL
"""

import os
import time
import torch
import cv2
import numpy as np
from django.conf import settings

# ── Service imports ─────────────────────────────────────────────────────────────
from .model_loader          import load_model
from .preprocess            import preprocess_from_bgr, preprocess
from .gradcam               import generate_cam
from .face_detection        import detect_and_crop_face, get_face_landmarks_simple

# Classical algorithms (from scratch)
from .kmeans                import kmeans_variance
from .edge                  import sobel_edge_score
from .entropy               import image_entropy_score

# New algorithms (from scratch)
from .frequency_analysis    import frequency_analysis_score
from .lbp                   import lbp_texture_score
from .color_stats           import color_stats_score

from .decision              import compute_decision_v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helper: safe face crop to RGB numpy ─────────────────────────────────────────

def _face_to_rgb(face_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR crop to RGB for classical algorithms."""
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)


def _resize_for_analysis(image_rgb: np.ndarray,
                          size: int = 224) -> np.ndarray:
    """Resize to square for consistent algorithm input."""
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


# ── Main Pipeline ────────────────────────────────────────────────────────────────

def run_inference_v2(image_path: str, instance_id: int) -> dict:
    """
    Full hybrid inference pipeline for one uploaded image.

    Args:
        image_path  : absolute path to saved upload
        instance_id : DetectionResult primary key (for heatmap filename)

    Returns dict with all scores and metadata needed by the view.
    Keys:
      is_fake, confidence, verdict, decision_score,
      freq_score, lbp_score, color_score,
      kmeans_variance, edge_score, entropy_score,
      heatmap_path, face_meta, processing_time, _api_dict
    """
    t_start = time.time()

    # ── 1. Improved face detection ──────────────────────────────────────────────
    face_bgr, face_meta = detect_and_crop_face(
        image_path,
        min_sharpness=20.0,
        min_face_ratio=0.03,
        pad_ratio=0.15,
    )

    face_rgb = _face_to_rgb(face_bgr)
    face_224 = _resize_for_analysis(face_rgb, size=224)

    # ── 2. Face landmarks & symmetry (bonus meta) ───────────────────────────────
    gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_224  = cv2.resize(gray_face, (224, 224))
    landmark_meta = get_face_landmarks_simple(gray_224)
    face_meta.update(landmark_meta)

    # ── 3. DL preprocessing → tensor ───────────────────────────────────────────
    model = load_model()

    # Try to use the improved preprocess_from_bgr if available,
    # fall back to path-based preprocess
    try:
        tensor = preprocess_from_bgr(face_bgr)
    except (AttributeError, Exception):
        tensor, _ = preprocess(image_path)

    # ── 4. ResNet18 forward pass ────────────────────────────────────────────────
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    # Label convention: 0 = real, 1 = fake
    p_real     = float(probs[0].item())
    p_fake_dl  = float(probs[1].item())
    dl_is_fake = p_fake_dl >= p_real
    dl_conf    = p_fake_dl if dl_is_fake else p_real
    pred_class = 1 if dl_is_fake else 0

    # ── 5. Classical algorithms (all on the face crop) ──────────────────────────

    # 5a. K-Means pixel diversity
    try:
        km_var = kmeans_variance(image=face_224, k=8, max_iters=30, sample_size=3000)
    except Exception as e:
        print(f"[Inference] K-Means failed: {e}")
        km_var = 0.5

    # 5b. Sobel edge detection
    try:
        edge_sc = sobel_edge_score(image=face_224, downsample_to=128)
    except Exception as e:
        print(f"[Inference] Sobel failed: {e}")
        edge_sc = 0.5

    # 5c. Shannon entropy
    try:
        entr_sc = image_entropy_score(image=face_224)
    except Exception as e:
        print(f"[Inference] Entropy failed: {e}")
        entr_sc = 0.5

    # 5d. DCT Frequency Analysis (new)
    try:
        freq_result = frequency_analysis_score(image=face_224)
        freq_sc     = freq_result["overall_score"]
    except Exception as e:
        print(f"[Inference] Frequency analysis failed: {e}")
        freq_result = {"overall_score": 0.5, "hf_ratio": 0.5,
                       "grid_artifact": 0.0, "spectral_slope": -2.0}
        freq_sc = 0.5

    # 5e. LBP Texture Analysis (new)
    try:
        lbp_result   = lbp_texture_score(image=face_bgr, P=8, R=1.0)
        lbp_realness = lbp_result["overall_score"]
    except Exception as e:
        print(f"[Inference] LBP failed: {e}")
        lbp_result   = {"overall_score": 0.5, "entropy": 3.0,
                        "uniformity_ratio": 0.5, "texture_variance": 100.0}
        lbp_realness = 0.5

    # 5f. Color Statistics (new)
    try:
        color_result   = color_stats_score(image=face_bgr)
        color_realness = color_result["overall_score"]
    except Exception as e:
        print(f"[Inference] Color stats failed: {e}")
        color_result   = {"overall_score": 0.5}
        color_realness = 0.5

    # ── 6. Grad-CAM Heatmap ─────────────────────────────────────────────────────
    heatmap_rel = None
    try:
        heatmap_filename = f"heatmap_{instance_id}.jpg"
        heatmap_abs      = os.path.join(settings.MEDIA_ROOT, "heatmaps", heatmap_filename)

        generate_cam(
            model        = model,
            tensor       = tensor,
            target_layer = model.layer4[-1],
            target_class = pred_class,
            save_path    = heatmap_abs,
        )
        heatmap_rel = f"heatmaps/{heatmap_filename}"
    except Exception as exc:
        print(f"[Inference] Grad-CAM failed (non-fatal): {exc}")

    # ── 7. Hybrid Decision Engine v2 ────────────────────────────────────────────
    result = compute_decision_v2(
        dl_is_fake      = dl_is_fake,
        dl_confidence   = dl_conf,
        freq_score      = freq_sc,
        lbp_realness    = lbp_realness,
        color_realness  = color_realness,
        kmeans_variance = km_var,
        edge_score      = edge_sc,
        entropy_score   = entr_sc,
        face_meta       = face_meta,
    )

    processing_time = round(time.time() - t_start, 3)

    # ── Build response dict ─────────────────────────────────────────────────────
    api_dict = result.to_api_dict()
    api_dict["processing_time"] = processing_time

    # Attach detailed sub-scores for frontend charts
    api_dict["detailed_analysis"] = {
        "frequency": freq_result,
        "lbp":       lbp_result,
        "color":     color_result,
        "face_meta": face_meta,
    }

    return {
        # Scalar fields for DB storage
        "is_fake":          result.verdict == "FAKE",
        "confidence_score": result.dl_confidence,    # DL confidence for DB field
        "verdict":          result.verdict,
        "decision_score":   result.decision_score,
        "freq_score":       freq_sc,
        "lbp_score":        lbp_realness,
        "color_score":      color_realness,
        "kmeans_variance":  km_var,
        "edge_score":       edge_sc,
        "entropy_score":    entr_sc,
        "heatmap_path":     heatmap_rel,
        "processing_time":  processing_time,
        "_api_dict":        api_dict,
    }
