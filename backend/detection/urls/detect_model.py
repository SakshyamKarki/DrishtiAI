from django.db import models
from core.models import BaseModel


class DetectionResult(BaseModel):

    class Status(models.TextChoices):
        PENDING   = "pending",   "Pending"
        COMPLETED = "completed", "Completed"
        FAILED    = "failed",    "Failed"

    class Label(models.TextChoices):
        REAL       = "Real",       "Real"
        DEEPFAKE   = "Deepfake",   "Deepfake"
        SUSPICIOUS = "Suspicious", "Suspicious"
        UNKNOWN    = "Unknown",    "Unknown"

    # ── relations ──────────────────────────────────────────────────────────────
    user = models.ForeignKey(
        'users.User',
        on_delete=models.CASCADE,
        related_name='detections',
    )

    # ── image files ────────────────────────────────────────────────────────────
    image   = models.ImageField(upload_to='detection/')
    heatmap = models.ImageField(upload_to='heatmaps/', null=True, blank=True)

    # ── deep-learning raw result ───────────────────────────────────────────────
    confidence_score = models.FloatField(null=True, blank=True)  # DL confidence (0-100)
    is_fake          = models.BooleanField(null=True, blank=True)

    # ── hybrid decision engine output ─────────────────────────────────────────
    final_label    = models.CharField(
        max_length=20,
        choices=Label.choices,
        default=Label.UNKNOWN,
    )
    decision_score = models.FloatField(null=True, blank=True)   # 0–1

    # ── classical algorithm scores (all from scratch) ──────────────────────────
    kmeans_variance = models.FloatField(null=True, blank=True)  # K-Means cluster diversity
    edge_score      = models.FloatField(null=True, blank=True)  # Sobel edge strength
    entropy_score   = models.FloatField(null=True, blank=True)  # Shannon entropy

    # ── metadata ───────────────────────────────────────────────────────────────
    status        = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    model_version = models.CharField(max_length=50, default="resnet18_v1")

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"[{self.pk}] {self.final_label} ({self.confidence_score}%) — {self.status}"
