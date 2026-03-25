import torch
import timm
from torchvision import models
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    # ===== Load Swin + ResNet50 Ensemble =====
    swin = timm.create_model("swin_tiny_patch4_window7_224", num_classes=2)
    resnet50 = timm.create_model("resnet50", num_classes=2)

    combined = torch.load("models/DrishtiAI_Deepfake_Image.pth", map_location=DEVICE)

    swin_state = {k.replace("swin.",""):v for k,v in combined.items() if k.startswith("swin.")}
    resnet_state = {k.replace("resnet.",""):v for k,v in combined.items() if k.startswith("resnet.")}

    swin.load_state_dict(swin_state)
    resnet50.load_state_dict(resnet_state)

    swin.to(DEVICE).eval()
    resnet50.to(DEVICE).eval()

    # ===== Load ResNet18 AI detector =====
    resnet18 = models.resnet18(weights=None)
    resnet18.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(resnet18.fc.in_features, 2)
    )

    state_dict = torch.load("models/DrishtiAI_AI_Image.pth", map_location=DEVICE)

    new_state = OrderedDict()
    for k,v in state_dict.items():
        new_state[k.replace("module.","")] = v

    resnet18.load_state_dict(new_state)
    resnet18.to(DEVICE).eval()

    return swin, resnet50, resnet18