from django.urls import path
from detection.views.detect_v3 import DetectionResultViewSet

urlpatterns = [
    path(
        "detection/",
        DetectionResultViewSet.as_view({
            "get": "list",
            "post": "create",
        }),
        name="detection-list",
    ),
    path(
        "detection/stats/",
        DetectionResultViewSet.as_view({
            "get": "stats",
        }),
        name="detection-stats",
    ),
]
