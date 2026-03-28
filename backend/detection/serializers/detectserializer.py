from rest_framework import serializers
from detection.models.detect import DetectionResult


class DetectionResultSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(write_only=True)  
    heatmap_url = serializers.SerializerMethodField()

    class Meta:
        model = DetectionResult
        fields = [
            "id",
            "image",     
            "confidence_score",
            "is_fake",
            "created_at",
            "heatmap_url"
        ]
        read_only_fields = ["confidence_score", "is_fake", "created_at"]



    def get_heatmap_url(self, obj):
        request = self.context.get("request")
        if obj.heatmap:
            return request.build_absolute_uri(obj.heatmap.url)
        return None