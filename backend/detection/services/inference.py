"""
Updated Inference Pipeline — DrishtiAI
=======================================
Integrates the three classical algorithms alongside the ResNet18 DL model.

Full pipeline for one image:
  preprocess  →  ResNet18 forward pass
              →  K-Means variance
              →  Sobel edge score
              →  Shannon entropy
              →  Grad-CAM heatmap
              →  Hybrid decision engine
              →  API response dict
"""

import os
import torch
import cv2
import numpy as np
from django.conf import settings

from .model_loader import load_model
from .preprocess   import preprocess
from .gradcam      import generate_cam

# ── classical algorithm imports (implemented from scratch) ─────────────────────
from .kmeans   import kmeans_variance
from .edge     import sobel_edge_score
from .entropy  import image_entropy_score
from .decision import compute_decision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(image_path: str, instance_id: int) -> dict:
    """
    End-to-end inference for a single uploaded image.

    Args:
        image_path  : absolute path to the saved image file
        instance_id : DetectionResult primary key (used for heatmap filename)

    Returns:
        dict with keys:
          is_fake, confidence_score, final_label, decision_score,
          kmeans_variance, edge_score, entropy_score,
          heatmap_path
    """
    # ── 1. load model ──────────────────────────────────────────────────────────
    model = load_model()

    # ── 2. preprocess: face crop + resize + normalise ──────────────────────────
    # returns: (1, C, H, W) tensor  +  (H, W, 3) uint8 RGB numpy array
    tensor, face_rgb = preprocess(image_path)

    # ── 3. deep learning prediction ────────────────────────────────────────────
    with torch.no_grad():
        output = model(tensor)                              # (1, 2) logits
        probs  = torch.softmax(output, dim=1)[0]          # (2,)

    # label convention assumed in training: 0 = real, 1 = fake
    p_real      = float(probs[0].item())
    p_fake      = float(probs[1].item())
    dl_is_fake  = p_fake >= p_real
    dl_conf     = p_fake if dl_is_fake else p_real        # confidence in own prediction
    pred_class  = 1 if dl_is_fake else 0

    # ── 4. classical algorithms (all implemented from scratch) ─────────────────
    # resize face to a standard 224×224 for consistent classical analysis
    face_224 = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    km_var    = kmeans_variance(image=face_224, k=8, max_iters=30, sample_size=3000)
    edge_sc   = sobel_edge_score(image=face_224, downsample_to=128)
    entropy_sc = image_entropy_score(image=face_224)

    # ── 5. Grad-CAM heatmap ────────────────────────────────────────────────────
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

    # ── 6. hybrid decision engine ──────────────────────────────────────────────
    result = compute_decision(
        dl_is_fake      = dl_is_fake,
        dl_confidence   = dl_conf,
        kmeans_variance = km_var,
        edge_score      = edge_sc,
        entropy_score   = entropy_sc,
    )

    return {
        # DL output
        "is_fake":          result.dl_is_fake,
        "confidence_score": result.dl_confidence,   # store DL confidence in existing field

        # hybrid output
        "final_label":    result.final_label,
        "decision_score": result.decision_score,

        # classical algorithm scores
        "kmeans_variance": result.kmeans_variance,
        "edge_score":      result.edge_score,
        "entropy_score":   result.entropy_score,

        # files
        "heatmap_path": heatmap_rel,

        # full API dict for the response body
        "_api_dict": result.to_api_dict(),
    }
