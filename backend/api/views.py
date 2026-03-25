import os
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

from .model_loader import load_models
from .inference import preprocess, predict_ensemble, predict_resnet18
from .gradcam import generate_cam

class PredictView(APIView):

    def post(self, request):
        file = request.FILES['image']

        # ===== Save input image =====
        filename = str(uuid.uuid4()) + ".jpg"
        path = os.path.join(settings.MEDIA_ROOT, filename)

        with open(path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        # ===== Preprocess (Face Detection included) =====
        tensor, face_img = preprocess(path)

        # ===== Load Models =====
        swin, resnet50, resnet18 = load_models()

        # ===== Ensemble Prediction =====
        label1, conf1, pred_class = predict_ensemble(swin, resnet50, tensor)

        # ===== Optional AI Detection =====
        label2, conf2 = None, None
        if label1 == "Deepfake":
            label2, conf2 = predict_resnet18(resnet18, tensor)

        # ===== Grad-CAM =====
        cam_filename = str(uuid.uuid4()) + "_cam.jpg"
        cam_path = os.path.join(settings.MEDIA_ROOT, cam_filename)

        # Use Swin backbone last layer
        target_layer = list(swin.children())[-1]

        generate_cam(
            model=swin,
            tensor=tensor,
            target_layer=target_layer,
            target_class=pred_class,
            save_path=cam_path
        )

        # ===== Response =====
        return Response({
            "prediction": {
                "deepfake_detection": label1,
                "confidence": round(conf1, 4),
                "ai_generated": label2,
                "ai_confidence": round(conf2, 4) if conf2 else None
            },
            "images": {
                "original": request.build_absolute_uri(settings.MEDIA_URL + filename),
                "gradcam": request.build_absolute_uri(settings.MEDIA_URL + cam_filename)
            },
            "meta": {
                "model": "DrishtiAI Ensemble + ResNet18",
                "face_detection": True
            }
        })