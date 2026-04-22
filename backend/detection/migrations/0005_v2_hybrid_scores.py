"""
Migration: add v2 hybrid scoring fields to DetectionResult
Run with:
    python manage.py migrate
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection", "0004_remove_detectionresult_decision_score_and_more"),
    ]

    operations = [
        # Core v2 fields
        migrations.AddField(
            model_name="detectionresult",
            name="verdict",
            field=models.CharField(
                max_length=20,
                default="UNKNOWN",
                choices=[
                    ("FAKE",       "Fake"),
                    ("REAL",       "Real"),
                    ("SUSPICIOUS", "Suspicious"),
                    ("UNKNOWN",    "Unknown"),
                ],
            ),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="decision_score",
            field=models.FloatField(null=True, blank=True,
                                    help_text="Hybrid weighted score 0-1"),
        ),

        # New classical algorithm scores
        migrations.AddField(
            model_name="detectionresult",
            name="freq_score",
            field=models.FloatField(null=True, blank=True,
                                    help_text="DCT frequency fakeness signal 0-1"),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="lbp_score",
            field=models.FloatField(null=True, blank=True,
                                    help_text="LBP texture realness score 0-1"),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="color_score",
            field=models.FloatField(null=True, blank=True,
                                    help_text="Color statistics realness 0-1"),
        ),

        # Keep/re-add classical scores from v1
        migrations.AddField(
            model_name="detectionresult",
            name="kmeans_variance",
            field=models.FloatField(null=True, blank=True,
                                    help_text="K-Means pixel diversity 0-1"),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="edge_score",
            field=models.FloatField(null=True, blank=True,
                                    help_text="Sobel edge strength 0-1"),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="entropy_score",
            field=models.FloatField(null=True, blank=True,
                                    help_text="Shannon entropy 0-1"),
        ),
    ]
