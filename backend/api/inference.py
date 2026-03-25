import torch
from torchvision import transforms
from PIL import Image
import cv2
from .face_detection import detect_face

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def preprocess(image_path):
    face = detect_face(image_path)  # 🔥 FACE DETECTION

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    return tensor, face_rgb