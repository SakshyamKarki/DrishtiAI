
from django.urls import path
from detection.views.detect import DetectionResultViewSet

urlpatterns = [
    path(
        "detection/",
        DetectionResultViewSet.as_view({
            "get": "list",
            "post": "create",
        }),
        name="detection-list",
    ),
]