from .mixins import TimeStampedModel, UserTrackingModel

class BaseModel(TimeStampedModel, UserTrackingModel):
    class Meta:
        abstract = True
        ordering = ("-created_at",)




