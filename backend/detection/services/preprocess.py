import torch
from torchvision import transforms
import cv2
from .face_detection import detect_face

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def preprocess(image_path):
    try:
        face = detect_face(image_path)

        if face is None:
            raise ValueError("No face detected")

    except Exception as e:
        print("Face detection failed:", e)

        face = cv2.imread(image_path)

        if face is None:
            raise ValueError("Image could not be read")

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    return tensor, face_rgb