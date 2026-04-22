"""
DrishtiAI — Improved Detection View (v2)
=========================================
Uses the full hybrid inference pipeline with 7 signals.
Stores all scores in DB for analytics and history.
"""

from django.conf import settings
from rest_framework.viewsets import ModelViewSet
from rest_framework.parsers  import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework          import status

from detection.models.detect                import DetectionResult
from detection.serializers.detectserializer import DetectionResultSerializer
from detection.services.inference           import run_inference_v2


class DetectionResultViewSet(ModelViewSet):
    queryset         = DetectionResult.objects.all()
    serializer_class = DetectionResultSerializer
    parser_classes   = [MultiPartParser, FormParser]

    def create(self, request, *args, **kwargs):
        # ── 1. Validate upload ──────────────────────────────────────────────────
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "No image file provided. Include field: image"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate content type
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
        if image_file.content_type not in allowed_types:
            return Response(
                {"error": f"Unsupported file type: {image_file.content_type}. "
                          f"Allowed: JPEG, PNG, WEBP"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── 2. Create pending DB record ─────────────────────────────────────────
        user = request.user if request.user.is_authenticated else None
        instance = DetectionResult.objects.create(
            user   = user,
            image  = image_file,
            status = DetectionResult.Status.PENDING,
        )

        try:
            # ── 3. Run full hybrid inference ────────────────────────────────────
            result = run_inference_v2(
                image_path  = instance.image.path,
                instance_id = instance.pk,
            )

            # ── 4. Persist scores to DB ─────────────────────────────────────────
            instance.is_fake          = result["is_fake"]
            instance.confidence_score = result["confidence_score"]
            instance.status           = DetectionResult.Status.COMPLETED

            # Extended fields (add to model if not present)
            _safe_set(instance, "verdict",         result["verdict"])
            _safe_set(instance, "decision_score",  result["decision_score"])
            _safe_set(instance, "freq_score",       result["freq_score"])
            _safe_set(instance, "lbp_score",        result["lbp_score"])
            _safe_set(instance, "color_score",      result["color_score"])
            _safe_set(instance, "kmeans_variance",  result["kmeans_variance"])
            _safe_set(instance, "edge_score",       result["edge_score"])
            _safe_set(instance, "entropy_score",    result["entropy_score"])

            if result.get("heatmap_path"):
                instance.heatmap = result["heatmap_path"]

            instance.save()

            # ── 5. Build response ────────────────────────────────────────────────
            api = result["_api_dict"]
            api["id"]    = instance.pk
            api["image"] = request.build_absolute_uri(instance.image.url)
            api["model"] = getattr(instance, "model_version", "resnet18_v1")

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
                {"error": f"Analysis failed: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def list(self, request, *args, **kwargs):
        """Return paginated detection history for authenticated user."""
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"},
                            status=status.HTTP_401_UNAUTHORIZED)
        qs = DetectionResult.objects.filter(user=request.user).order_by("-created_at")[:50]
        serializer = DetectionResultSerializer(
            qs, many=True, context={"request": request}
        )
        return Response(serializer.data)


def _safe_set(instance, attr: str, value):
    """Set attribute only if the model field exists (avoids migration errors)."""
    if hasattr(instance, attr):
        setattr(instance, attr, value)
