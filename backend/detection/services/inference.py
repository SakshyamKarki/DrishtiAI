import torch
from .model_loader import load_model
from .preprocess import preprocess

def predict(image_path):
    model = load_model()

    tensor = preprocess(image_path)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probs, 1)

    return {
        "is_fake": bool(predicted.item()),
        "confidence": float(confidence.item())
    }