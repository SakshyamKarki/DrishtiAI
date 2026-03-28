import torch
from torchvision import models
from collections import OrderedDict
from django.conf import settings
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, "DrishtiAI_AI_Image.pth")

_model = None 

def load_model():
    global _model

    if _model is not None:
        return _model
    model = models.resnet18(weights=None)

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.fc.in_features, 2)
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state)
    model.to(DEVICE).eval()

    _model = model
    return model