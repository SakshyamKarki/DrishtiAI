<div align="center">

```
██████╗ ██████╗ ██╗███████╗██╗  ██╗████████╗██╗ █████╗ ██╗
██╔══██╗██╔══██╗██║██╔════╝██║  ██║╚══██╔══╝██║██╔══██╗██║
██║  ██║██████╔╝██║███████╗███████║   ██║   ██║███████║██║
██║  ██║██╔══██╗██║╚════██║██╔══██║   ██║   ██║██╔══██║██║
██████╔╝██║  ██║██║███████║██║  ██║   ██║   ██║██║  ██║██║
╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚═╝
```

**14-signal deepfake detection engine for profile photos**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2-green?style=flat-square&logo=django&logoColor=white)](https://djangoproject.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

*"See through the lie."*

</div>

---

## What is DrishtiAI?

**Drishti** (दृष्टि) means *sight* or *vision* in Sanskrit. DrishtiAI is a forensic deepfake detection system purpose-built for **profile photo verification** — the specific problem of AI-generated faces used in fake social media accounts, dating profiles, and identity fraud.

It fuses a fine-tuned **ResNet18** deep learning classifier with **13 additional classical and physics-based signals**, all processed in a single pipeline that returns a verdict in under 3 seconds with a full forensic breakdown and Grad-CAM attention heatmap.

---

## Table of Contents

- [Signal Architecture](#signal-architecture)
- [Detection Pipeline](#detection-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [How Each Signal Works](#how-each-signal-works)
- [New Signals (v3)](#new-signals-v3)
- [Bug Fixes & Refactors](#bug-fixes--refactors)
- [Frontend](#frontend)
- [Authentication](#authentication)
- [Contributing](#contributing)

---

## Signal Architecture

DrishtiAI merges **14 independent fakeness signals** into a single calibrated decision score:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SIGNAL WEIGHT TABLE                         │
├────────────────────────────┬──────────┬────────────────────────┤
│ Signal                     │ Weight   │ Category               │
├────────────────────────────┼──────────┼────────────────────────┤
│ ResNet18 Deep Learning     │  38.0%   │ Deep Learning          │
│ DCT Frequency Analysis     │  10.0%   │ Classical              │
│ LBP Texture + Bit Count    │   8.0%   │ Classical              │
│ Color Statistics           │   8.0%   │ Classical              │
│ K-Means Diversity          │   4.0%   │ Classical              │
│ K-Means Elbow Ratio ★      │   1.0%   │ Classical (NEW)        │
│ Sobel Edge + Dir. Entropy★ │   4.0%   │ Classical              │
│ Shannon Entropy            │   3.0%   │ Classical              │
│ SSIM Patch Smoothness      │   5.0%   │ Enhanced               │
│ HSV Skin Uniformity        │   5.0%   │ Enhanced               │
│ Laplacian Sharpness        │   4.0%   │ Enhanced               │
│ Chromatic Aberration       │   4.0%   │ Enhanced               │
│ Background Coherence       │   3.0%   │ Enhanced               │
│ Sensor Noise Pattern       │   3.0%   │ Enhanced               │
├────────────────────────────┼──────────┼────────────────────────┤
│ TOTAL                      │ 100.0%   │                        │
└────────────────────────────┴──────────┴────────────────────────┘
★ New signals added in v3 refactor
```

**Verdicts:**

| Score     | Verdict    | Risk Label                    |
|-----------|------------|-------------------------------|
| ≥ 0.60    | `FAKE`     | High Risk — Likely AI Generated |
| 0.37–0.59 | `SUSPICIOUS` | Medium Risk — Needs Review  |
| ≤ 0.36    | `REAL`     | Low Risk — Likely Authentic   |
| N/A       | `NO_FACE`  | Cannot Analyse — No Face Detected |

---

## Detection Pipeline

```
                         INPUT: profile photo
                               │
                    ┌──────────▼──────────┐
                    │   Face Detection    │
                    │  Haar + CLAHE +NMS  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               │ NoFaceError   │               │
               ▼               ▼               │
        NO_FACE response   Face crop          │
        (immediate)        224×224             │
                               │               │
            ┌──────────────────┼───────────────┘
            │                  │
    ┌───────▼──────┐  ┌────────▼────────────────┐
    │  ResNet18    │  │  13 Classical/Enhanced   │
    │  inference   │  │  signals in parallel     │
    │  + Grad-CAM  │  │                          │
    └───────┬──────┘  └────────┬────────────────┘
            │                  │
            └────────┬─────────┘
                     ▼
          ┌──────────────────────┐
          │   Decision Engine v3 │
          │  Weighted fusion +   │
          │  profile modifiers   │
          └──────────┬───────────┘
                     ▼
           REAL / SUSPICIOUS / FAKE
           + confidence + breakdown
```

---

## Project Structure

```
DrishtiAI/
├── backend/
│   ├── config/
│   │   ├── settings.py
│   │   └── auth_settings.py          ← JWT + session config
│   ├── detection/
│   │   ├── models.py                 ← DetectionInstance Django model
│   │   ├── views.py                  ← API views
│   │   ├── urls.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── model_loader.py       ← ResNet18 singleton loader
│   │       ├── preprocess.py         ← CLAHE face preprocessing
│   │       ├── face_detection.py     ← Haar + NMS + NoFaceError
│   │       ├── inference_v3.py       ← Main pipeline orchestrator
│   │       ├── decision_v3.py        ← Weighted fusion engine
│   │       ├── lbp.py                ← LBP texture + bit-counting
│   │       ├── kmeans.py             ← K-Means + elbow ratio
│   │       ├── edge.py               ← Sobel + direction entropy
│   │       ├── entropy.py            ← Shannon entropy
│   │       ├── frequency_analysis.py ← DCT/FFT spectral analysis
│   │       ├── color_stats.py        ← Color statistics + symmetry
│   │       ├── enhanced_pipeline.py  ← SSIM, HSV, CA, BG, Noise
│   │       └── gradcam.py            ← Grad-CAM heatmap generation
│   └── ml_models/
│       └── DrishtiAI_AI_Image.pth    ← Pre-trained ResNet18 weights
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.js             ← Axios + JWT auto-refresh
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx       ← Hero + features (JSX)
│   │   │   └── index.html            ← Standalone HTML hero page
│   │   └── components/
│   └── public/
│
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional but recommended)

### Backend

```bash
# 1. Clone
git clone https://github.com/yourusername/DrishtiAI.git
cd DrishtiAI

# 2. Virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Optional but recommended — faster LBP and edge detection:
pip install scikit-image scipy

# 4. Place model weights
# Download DrishtiAI_AI_Image.pth and place in:
mkdir -p backend/ml_models
cp /path/to/DrishtiAI_AI_Image.pth backend/ml_models/

# 5. Environment variables
cp .env.example .env
# Edit .env: set SECRET_KEY, DEBUG, DATABASE_URL, ML_MODELS_DIR

# 6. Database setup
cd backend
python manage.py migrate

# 7. Add JWT blacklist table
# Ensure 'rest_framework_simplejwt.token_blacklist' is in INSTALLED_APPS
python manage.py migrate token_blacklist

# 8. Run
python manage.py runserver
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Set: VITE_API_BASE_URL=http://localhost:8000

npm run dev
# → http://localhost:5173
```

---

## Usage

### Web Interface

1. Navigate to `http://localhost:5173`
2. Log in or register
3. Upload a profile photo (JPEG/PNG/WebP)
4. View verdict, confidence score, Grad-CAM heatmap, and full 14-signal breakdown

### Python Client

```python
import requests

# Authenticate
r = requests.post("http://localhost:8000/api/token/", json={
    "username": "your_user",
    "password": "your_pass"
})
tokens = r.json()
headers = {"Authorization": f"Bearer {tokens['access']}"}

# Analyse a face
with open("profile.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/detect/",
        headers=headers,
        files={"image": f}
    )

result = r.json()
print(result["verdict"])          # "FAKE" | "REAL" | "SUSPICIOUS" | "NO_FACE"
print(result["confidence"])       # e.g. 94.2
print(result["decision_score"])   # e.g. 0.7831
print(result["risk_label"])       # "High Risk — Likely AI Generated"
```

---

## API Reference

### `POST /api/token/`
Obtain JWT access + refresh tokens.

**Request:**
```json
{ "username": "string", "password": "string" }
```
**Response:**
```json
{ "access": "eyJ...", "refresh": "eyJ..." }
```

---

### `POST /api/token/refresh/`
Silently refresh an expired access token.

**Request:**
```json
{ "refresh": "eyJ..." }
```
**Response:**
```json
{ "access": "eyJ...", "refresh": "eyJ..." }
```

---

### `POST /api/detect/`
Analyse a profile photo. Requires `Authorization: Bearer <access_token>`.

**Request:** `multipart/form-data`
| Field | Type   | Required |
|-------|--------|----------|
| image | file   | ✅        |

**Response (success):**
```json
{
  "verdict": "FAKE",
  "final_label": "FAKE",
  "is_fake": true,
  "confidence": 94.20,
  "decision_score": 0.7831,
  "risk_label": "High Risk — Likely AI Generated",
  "processing_time": 2.431,

  "profile_analysis": {
    "face_detected": true,
    "face_count": 1,
    "single_face": true,
    "face_quality_score": 0.812,
    "face_sharpness": 312.4,
    "skin_uniformity": 0.821,
    "background_natural": 0.342
  },

  "dl_prediction": {
    "label": "Fake",
    "confidence": 0.9420,
    "weight": 0.38
  },

  "analysis": {
    "frequency":          { "score": 0.71, "weight": 0.10, ... },
    "texture_lbp":        { "score": 0.68, "weight": 0.08, ... },
    "color_stats":        { "score": 0.55, "weight": 0.08, ... },
    "pixel_diversity":    { "score": 0.42, "weight": 0.04, ... },
    "kmeans_elbow":       { "score": 0.76, "weight": 0.01, ... },
    "edge_sharpness":     { "score": 0.38, "weight": 0.04, ... },
    "information_density":{ "score": 0.44, "weight": 0.03, ... },
    "ssim_texture":       { "score": 0.83, "weight": 0.05, ... },
    "skin_tone_uniformity":{ "score": 0.79, "weight": 0.05, ... },
    "sharpness_profile":  { "score": 0.51, "weight": 0.04, ... },
    "lens_physics":       { "score": 0.88, "weight": 0.04, ... },
    "background_analysis":{ "score": 0.62, "weight": 0.03, ... },
    "sensor_noise":       { "score": 0.74, "weight": 0.03, ... }
  },

  "heatmap_path": "heatmaps/heatmap_42.jpg"
}
```

**Response (no face detected):**
```json
{
  "verdict": "NO_FACE",
  "is_fake": false,
  "risk_label": "Cannot Analyse — No Face Detected",
  "message": "No human face detected in this image. Please submit a clear profile photo containing a visible face.",
  "processing_time": 0.031,
  "profile_analysis": { "face_detected": false, "face_count": 0 }
}
```

---

## Configuration

### `backend/config/auth_settings.py`

```python
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME":    timedelta(hours=8),   # was 5 minutes
    "REFRESH_TOKEN_LIFETIME":   timedelta(days=30),
    "ROTATE_REFRESH_TOKENS":    True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN":        True,
}
SESSION_COOKIE_AGE = 60 * 60 * 8   # 8 hours
SESSION_SAVE_EVERY_REQUEST = True
```

Include in `settings.py`:
```python
from .auth_settings import *
```

### Environment Variables (`.env`)

```env
SECRET_KEY=your-django-secret-key
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
ML_MODELS_DIR=/absolute/path/to/backend/ml_models
MEDIA_ROOT=/absolute/path/to/media
ALLOWED_HOSTS=localhost,127.0.0.1
```

---

## How Each Signal Works

### 1. ResNet18 Deep Learning (38%)

Fine-tuned ResNet18 with a `Dropout(0.3) → Linear(512→2)` head.
- Input: 224×224 CLAHE-normalised face crop (ImageNet normalised)
- Output: softmax probabilities `[p_real, p_fake]`
- Grad-CAM heatmap generated from `model.layer4[-1]`

### 2. DCT Frequency Analysis (10%)

AI generators leave spectral fingerprints:
- **HF energy ratio**: `E_high / E_total` — real photos have more HF energy
- **Grid artifact score**: peaks at GAN upsampling frequencies (N/8, N/16 etc.)
- **Spectral slope**: log-log regression on radial PSD — natural images follow 1/f²

### 3. LBP Texture + Bit Counting (8%)

Circular LBP encodes each pixel's relationship to its 8 circular neighbours:
- **Entropy** of uniform-pattern histogram — real skin is richer
- **Uniformity ratio** — AI skin produces more uniform LBP patterns
- **Popcount KL divergence** *(new)*: counts set bits in each 8-bit LBP code using a precomputed 256-entry lookup table, measures divergence from uniform popcount distribution

### 4. Color Statistics (8%)

- **Pearson correlation** between R, G, B channels — real skin: 0.75–0.97
- **Channel std** — AI skin is too uniform (low std) or noisy (high std)
- **Bilateral symmetry** — AI faces are too symmetric
- **Noise consistency** — real cameras have consistent per-channel noise

### 5. K-Means Variance + Elbow Ratio (5%)

- **WCSS (inertia) at k=8**: low = too-uniform clusters = AI
- **Elbow ratio W(k=2)/W(k=8)** *(new)*: real images show a steep inertia drop from k=2→8 (structure); AI images show near-linear decay

### 6. Sobel Edge + Gradient Direction Entropy (4%)

- **Gradient magnitude** (mean Sobel): real photos have stronger edges
- **Direction entropy** *(new)*: angle histogram (8 bins) of strong-edge pixels — AI faces concentrate gradients at canonical angles (0°, 45°, 90°...)

### 7. Shannon Entropy (3%)

Per-channel pixel intensity entropy. AI textures occupy fewer intensity levels → lower entropy.

### 8. SSIM Patch Smoothness (5%)

Structural similarity between adjacent 16×16 skin patches. AI skin generators produce over-smooth interpolations → abnormally high inter-patch SSIM.

### 9. HSV Skin Uniformity (5%)

Hue entropy of skin-masked pixels. Real skin has a broad hue distribution from vascular variation, lighting, and shadow. AI skin is unnaturally peaked.

### 10. Laplacian Sharpness Profile (4%)

4-level Laplacian pyramid. Real photos show monotone energy decay (level 0 > 1 > 2 > 3). AI images violate this — either too sharp at coarse scales or uniform across all scales.

### 11. Chromatic Aberration (4%)

Physical camera lenses cause R and B channels to misalign slightly at edges. This signal measures R/B edge map alignment — perfect alignment (AI) vs. natural misalignment (real lens).

### 12. Background Coherence (3%)

Profile-photo-specific: analyses variance ratio between face centre and background corners. AI generators produce unnatural depth-of-field separation (near-zero background variance).

### 13. Sensor Noise Pattern (3%)

High-pass residual (image − Gaussian blur) per channel. Real cameras: consistent noise std 1–8; AI images: too clean (<0.5) or inconsistent across channels.

---

## New Signals (v3)

Three new hardcoded signals were added in the v3 refactor, all derived from bit-manipulation and information theory principles:

### Popcount KL Divergence (in LBP)

```python
# Each LBP code is an 8-bit integer
# Count set bits via precomputed lookup table (O(1) per pixel)
_POPCOUNT_LUT = [bin(i).count('1') for i in range(256)]

# Real skin: popcount spread 0–8 (diverse texture)
# AI skin:   popcount peaked at 3–5 (smooth, neighbours ≈ centre)
# Measure: KL divergence from uniform distribution
fakeness = min(kl_divergence / 0.50, 1.0)
```

### K-Means Elbow Ratio

```python
# Run k-means at k=2 and k=8
w2 = inertia(pixels, k=2)
w8 = inertia(pixels, k=8)
elbow_ratio = w2 / (w8 + 1e-12)

# Real: ratio 4–10   (big structural drop = diverse pixel clusters)
# AI:   ratio 1.5–3  (linear decay = uniform texture)
fakeness = 1 - clamp((ratio - 2) / 4, 0, 1)
```

### Gradient Direction Entropy

```python
# Compute gradient angles: θ = atan2(Gy, Gx)
# Keep top 30% strongest edges, quantise into 8 bins (0°–360°)
# Weighted histogram (weight = magnitude)
# Shannon entropy of histogram

# Real faces: rich mix of directions → HIGH entropy
# AI faces:   canonical directions dominate → LOW entropy
realness = entropy / log2(8)
```

---

## Bug Fixes & Refactors

| Bug | File | Fix |
|-----|------|-----|
| `symmetry_score` key collision between face detection and landmarks | `face_detection.py` | Renamed to `landmark_symmetry_score` |
| Silent `except: pass` hides all signal failures | `inference_v3.py` | All exceptions now logged via `logger.warning/error` |
| No-face images ran full pipeline, produced meaningless scores | `face_detection.py` + `inference_v3.py` | `NoFaceError` raised, caught, returns `NO_FACE` verdict immediately |
| `confidence_score` vs `dl_confidence` inconsistency (0–1 vs 0–100) | `inference_v3.py` | `confidence_score` consistently 0–100 |
| SUSPICIOUS verdict confidence could exceed 70% | `decision_v3.py` | Capped at 70% |
| O(H×W) Python loop in LBP | `lbp.py` | Replaced with scipy/skimage vectorised fast paths |
| O(N) Python loop in `_inertia()` | `kmeans.py` | Replaced with `np.einsum` + fancy indexing |
| 7-minute JWT session expiry | `auth_settings.py` | 8-hour access token, 30-day rotating refresh |
| No token refresh on 401 → forced logout | `client.js` | Axios interceptor with silent refresh + request queue |
| Multiple cascade detections for same face | `face_detection.py` | IoU-based NMS deduplication added |

---

## Frontend

### Hero Page (`frontend/public/index.html`)

Standalone HTML/CSS/JS — no build step required.
- **Aesthetic:** Biopunk surveillance — dark ink background, acid-green (#c8ff00) accents, orthographic grid
- **Animated SVG eye** with rotating orbits, scan arc, signal lines
- **Floating data readouts** showing live analysis values
- **Custom cursor** with lag ring
- **14-signal grid** with animated weight bars
- **Interactive accuracy bars** that animate on scroll
- **Terminal readout panel** showing full JSON output

### React Landing Page (`frontend/src/pages/LandingPage.jsx`)

Same content as a React component for integration into the Vite app.

### Axios Client (`frontend/src/api/client.js`)

```javascript
import api from "./api/client";

// All authenticated requests — token auto-attached + auto-refreshed
const result = await api.post("/api/detect/", formData);
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-signal`
3. Add your signal in `backend/detection/services/`
4. Register it in `inference_v3.py` and `decision_v3.py`
5. Update the weight table (must sum to 1.0)
6. Add tests in `backend/detection/tests/`
7. Submit a pull request with benchmark results

### Adding a New Signal

```python
# 1. Create your signal function
def my_signal_score(image_bgr: np.ndarray) -> float:
    """Returns fakeness score [0, 1]. Higher = more fake."""
    ...

# 2. Call it in inference_v3.py
try:
    my_score = my_signal_score(face_bgr_224)
except Exception as e:
    logger.warning(f"my_signal_score failed: {e}")
    my_score = 0.5

# 3. Add weight in decision_v3.py
MY_W = 0.02   # adjust other weights so total = 1.0

# 4. Use in compute_decision_v3()
score += MY_W * my_score
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built with PyTorch, OpenCV, Django, and React.*

**DRISHTI**AI — *See through the lie.*

</div>
