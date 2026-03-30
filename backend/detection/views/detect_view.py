"""
Updated DetectionResultViewSet
================================
Calls run_inference() which now integrates:
  • ResNet18 deep learning model
  • K-Means clustering (from scratch)
  • Sobel edge detection (from scratch)
  • Shannon entropy     (from scratch)
  • Hybrid decision engine
  • Grad-CAM heatmap
"""

from django.conf import settings
from rest_framework.viewsets import ModelViewSet
from rest_framework.parsers  import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework          import status

from detection.models.detect              import DetectionResult
from detection.serializers.detectserializer import DetectionResultSerializer

# unified inference pipeline (DL + classical algorithms + decision engine)
from detection.services.inference import run_inference


class DetectionResultViewSet(ModelViewSet):
    queryset         = DetectionResult.objects.all()
    serializer_class = DetectionResultSerializer
    parser_classes   = [MultiPartParser, FormParser]

    def create(self, request, *args, **kwargs):
        # ── validate image upload ──────────────────────────────────────────────
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "Image file is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── create pending DB record ───────────────────────────────────────────
        instance = DetectionResult.objects.create(
            user   = request.user if request.user.is_authenticated else None,
            image  = image_file,
            status = DetectionResult.Status.PENDING,
        )

        try:
            # ── run the full hybrid inference pipeline ─────────────────────────
            result = run_inference(
                image_path  = instance.image.path,
                instance_id = instance.pk,
            )

            # ── persist all scores ─────────────────────────────────────────────
            instance.is_fake          = result["is_fake"]
            instance.confidence_score = result["confidence_score"]
            instance.final_label      = result["final_label"]
            instance.decision_score   = result["decision_score"]
            instance.kmeans_variance  = result["kmeans_variance"]
            instance.edge_score       = result["edge_score"]
            instance.entropy_score    = result["entropy_score"]
            instance.status           = DetectionResult.Status.COMPLETED

            if result.get("heatmap_path"):
                instance.heatmap = result["heatmap_path"]

            instance.save()

            # ── build response ─────────────────────────────────────────────────
            api = result["_api_dict"]
            api["id"]          = instance.pk
            api["image"]       = request.build_absolute_uri(instance.image.url)
            api["model"]       = instance.model_version

            if instance.heatmap:
                api["heatmap_url"] = request.build_absolute_uri(
                    f"{settings.MEDIA_URL}{instance.heatmap}"
                )

            return Response(api, status=status.HTTP_201_CREATED)

        except Exception as exc:
            instance.status = DetectionResult.Status.FAILED
            instance.save()
            return Response(
                {"error": f"Inference failed: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
