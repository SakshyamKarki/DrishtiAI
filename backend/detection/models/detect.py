from django.db import models
from core.models import BaseModel  


class DetectionResult(BaseModel):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        'users.User',
        on_delete=models.CASCADE,
        related_name='detections'
    )

    image = models.ImageField(upload_to='detection/')

    confidence_score = models.FloatField(null=True, blank=True)
    is_fake = models.BooleanField(null=True, blank=True)

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )

    model_version = models.CharField(max_length=50, default="resnet18_v1")
    heatmap = models.ImageField(upload_to="heatmaps/", null=True, blank=True)

    def __str__(self):
        return f"{self.id} - {self.user} - {self.status}"