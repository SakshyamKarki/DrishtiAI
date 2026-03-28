from django_currentuser.db.models import CurrentUserField
from django.db import models

class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UserTrackingModel(models.Model):
    created_by = CurrentUserField(
        related_name="%(class)s_created",
        on_update=False
    )
    modified_by = CurrentUserField(
        related_name="%(class)s_modified",
        on_update=True
    )

    class Meta:
        abstract = True