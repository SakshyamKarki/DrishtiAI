"""
Migration: add hybrid decision engine fields to DetectionResult
Run with:
    python manage.py makemigrations detection --name hybrid_algorithm_fields
    python manage.py migrate
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0002_detectionresult_heatmap'),
    ]

    operations = [
        migrations.AddField(
            model_name='detectionresult',
            name='final_label',
            field=models.CharField(
                max_length=20,
                default='Unknown',
                choices=[
                    ('Real',       'Real'),
                    ('Deepfake',   'Deepfake'),
                    ('Suspicious', 'Suspicious'),
                    ('Unknown',    'Unknown'),
                ],
            ),
        ),
        migrations.AddField(
            model_name='detectionresult',
            name='decision_score',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='detectionresult',
            name='kmeans_variance',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='detectionresult',
            name='edge_score',
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='detectionresult',
            name='entropy_score',
            field=models.FloatField(null=True, blank=True),
        ),
    ]
