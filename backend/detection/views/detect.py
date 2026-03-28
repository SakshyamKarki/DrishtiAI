import os
import torch
from django.conf import settings
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from detection.models.detect import DetectionResult
from detection.serializers.detectserializer import DetectionResultSerializer
from detection.services.inference import preprocess
from detection.services.model_loader import load_model
from detection.services.gradcam import generate_cam


resnet18 = load_model()  

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import torch

class DetectionResultViewSet(ModelViewSet):
    queryset = DetectionResult.objects.all()
    serializer_class = DetectionResultSerializer
    parser_classes = [MultiPartParser, FormParser]  

    def create(self, request, *args, **kwargs):
        image = request.FILES.get("image")
        print("USER:", request.user)
        print("AUTH:", request.user.is_authenticated)

        if not image:
            return Response(
                {"error": "Image file is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        instance = DetectionResult.objects.create(
            user=request.user,
            image=image,
            status="processing"
        )

        image_path = instance.image.path

        tensor, _ = preprocess(image_path)

        with torch.no_grad():
            output = resnet18(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        instance.confidence_score = round(float(confidence.item())*100, 2)
        instance.is_fake = bool(pred.item())
        instance.status = "completed"

        heatmap_filename = f"heatmap_{instance.id}.jpg"
        heatmap_path = os.path.join(settings.MEDIA_ROOT, "heatmaps", heatmap_filename)

        generate_cam(
            model=resnet18,
            tensor=tensor,
            target_layer=resnet18.layer4[-1],
            target_class=int(pred.item()),
            save_path=heatmap_path
        )

        instance.heatmap = f"heatmaps/{heatmap_filename}"

        instance.save()

        return Response({
            "id": instance.id,
            "image": request.build_absolute_uri(instance.image.url),
            "model": "resnet18",
            "confidence": f"{instance.confidence_score}%",
            "is_fake": instance.is_fake,
            "heatmap": request.build_absolute_uri(
                f"{settings.MEDIA_URL}{instance.heatmap}"
            )
        }, status=status.HTTP_201_CREATED)


