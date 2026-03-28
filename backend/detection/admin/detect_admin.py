from django.contrib import admin
from detection.models.detect import DetectionResult

@admin.register(DetectionResult)
class DetectAdmin(admin.ModelAdmin):
    list_display = ("status",)