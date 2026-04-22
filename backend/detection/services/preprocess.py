"""
DrishtiAI — Improved Image Preprocessing
==========================================
Provides two entry points:
  1. preprocess(image_path)      — original path-based (backward compat)
  2. preprocess_from_bgr(bgr)   — accepts pre-cropped BGR numpy array

Improvements over v1:
  - Accepts pre-cropped face from improved face detection
  - CLAHE equalization for better model performance on dark/overexposed faces
  - Handles grayscale and RGBA inputs
  - Returns both tensor and face_rgb for classical algorithms
"""

import torch
import cv2
import numpy as np
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet normalization (same as training)
_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def _bgr_to_rgb_safe(img: np.ndarray) -> np.ndarray:
    """
    Convert any OpenCV image to uint8 RGB regardless of input format.
    Handles: BGR, BGRA, grayscale.
    """
    if img is None:
        raise ValueError("Image array is None")
    if len(img.shape) == 2:
        # Grayscale → RGB
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        # BGRA → RGB
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def preprocess_from_bgr(face_bgr: np.ndarray):
    """
    Preprocess a pre-cropped BGR face for ResNet18 inference.

    Steps:
      1. Convert BGR → RGB (safe)
      2. Apply mild CLAHE to luminance for consistent exposure
      3. Apply ImageNet normalisation transform
      4. Move tensor to device

    Args:
        face_bgr : BGR uint8 numpy array (any size)

    Returns:
        tensor : (1, 3, 224, 224) float32 on DEVICE
    """
    face_rgb = _bgr_to_rgb_safe(face_bgr)

    # Mild CLAHE in LAB space to normalise exposure without distorting color
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    face_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    tensor = _TRANSFORM(face_rgb).unsqueeze(0).to(DEVICE)
    return tensor


def preprocess(image_path: str):
    """
    Original path-based preprocessing (backward compatible).
    Used as fallback when face detection hasn't run yet.

    Returns:
        (tensor, face_rgb)
    """
    from .face_detection import detect_and_crop_face

    face_bgr, _ = detect_and_crop_face(image_path)
    face_rgb = _bgr_to_rgb_safe(face_bgr)
    tensor = preprocess_from_bgr(face_bgr)
    return tensor, face_rgb
