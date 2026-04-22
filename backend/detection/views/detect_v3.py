"""
DrishtiAI — Detection View v3
==============================
Enhanced fake PROFILE PHOTO detection with 14-signal hybrid pipeline.
Returns richer API response with profile-photo-specific analysis.
"""

from django.conf import settings
from rest_framework.viewsets import ModelViewSet
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action

from detection.models.detect import DetectionResult
from detection.serializers.detectserializer import DetectionResultSerializer
from detection.services.inference_v3 import run_inference_v3


class DetectionResultViewSet(ModelViewSet):
    queryset = DetectionResult.objects.all()
    serializer_class = DetectionResultSerializer
    parser_classes = [MultiPartParser, FormParser]

    # Allowed content types
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    MAX_FILE_SIZE_MB = 15

    def create(self, request, *args, **kwargs):
        # ── Validate upload ─────────────────────────────────────────────────
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {
                    "error": "No image file provided.",
                    "detail": "Include the image as multipart field 'image'.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if image_file.content_type not in self.ALLOWED_TYPES:
            return Response(
                {
                    "error": f"Unsupported file type: {image_file.content_type}",
                    "detail": "Accepted formats: JPEG, PNG, WEBP",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if image_file.size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            return Response(
                {
                    "error": f"File too large. Maximum size: {self.MAX_FILE_SIZE_MB}MB",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── Create pending DB record ────────────────────────────────────────
        user = request.user if request.user.is_authenticated else None
        instance = DetectionResult.objects.create(
            user=user,
            image=image_file,
            status=DetectionResult.Status.PENDING,
        )

        try:
            # ── Run enhanced inference pipeline v3 ──────────────────────────
            result = run_inference_v3(
                image_path=instance.image.path,
                instance_id=instance.pk,
            )

            # ── Persist to DB ────────────────────────────────────────────────
            instance.is_fake = result["is_fake"]
            instance.confidence_score = result["confidence_score"]
            instance.status = DetectionResult.Status.COMPLETED

            _safe_set(instance, "verdict", result["verdict"])
            _safe_set(instance, "decision_score", result["decision_score"])
            _safe_set(instance, "freq_score", result["freq_score"])
            _safe_set(instance, "lbp_score", result["lbp_score"])
            _safe_set(instance, "color_score", result["color_score"])
            _safe_set(instance, "kmeans_variance", result["kmeans_variance"])
            _safe_set(instance, "edge_score", result["edge_score"])
            _safe_set(instance, "entropy_score", result["entropy_score"])

            if result.get("heatmap_path"):
                instance.heatmap = result["heatmap_path"]

            instance.save()

            # ── Build response ───────────────────────────────────────────────
            api = result["_api_dict"]
            api["id"] = instance.pk
            api["image"] = request.build_absolute_uri(instance.image.url)
            api["model"] = getattr(instance, "model_version", "resnet18_v3")
            api["pipeline_version"] = "v3"

            if instance.heatmap:
                api["heatmap"] = request.build_absolute_uri(
                    f"{settings.MEDIA_URL}{instance.heatmap}"
                )

            return Response(api, status=status.HTTP_201_CREATED)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            instance.status = DetectionResult.Status.FAILED
            instance.save()
            return Response(
                {
                    "error": f"Analysis failed: {str(exc)}",
                    "detail": "The image could not be analyzed. Please try a clear face photo.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def list(self, request, *args, **kwargs):
        """Return detection history for authenticated user."""
        if not request.user.is_authenticated:
            return Response(
                {"error": "Authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        qs = DetectionResult.objects.filter(
            user=request.user
        ).order_by("-created_at")[:50]
        serializer = DetectionResultSerializer(
            qs, many=True, context={"request": request}
        )
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def stats(self, request):
        """Return aggregate detection stats for the authenticated user."""
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"}, status=401)

        from django.db.models import Count, Avg
        qs = DetectionResult.objects.filter(user=request.user)
        total = qs.count()
        fake_count = qs.filter(is_fake=True).count()
        real_count = qs.filter(is_fake=False).count()
        avg_conf = qs.aggregate(avg=Avg("confidence_score"))["avg"] or 0

        # Weekly count
        from django.utils import timezone
        from datetime import timedelta
        week_ago = timezone.now() - timedelta(days=7)
        weekly = qs.filter(created_at__gte=week_ago).count()

        return Response({
            "total_checked": total,
            "fake_count": fake_count,
            "real_count": real_count,
            "suspicious_count": total - fake_count - real_count,
            "avg_confidence": round(avg_conf, 1),
            "weekly_count": weekly,
            "fake_rate": round(fake_count / total * 100, 1) if total > 0 else 0,
        })


def _safe_set(instance, attr: str, value):
    if hasattr(instance, attr):
        setattr(instance, attr, value)
