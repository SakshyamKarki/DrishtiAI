import numpy as np
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_cam(model, tensor, target_layer, target_class, save_path):
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    img = tensor.cpu().squeeze().permute(1,2,0).numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + img * 255 * 0.6

    cv2.imwrite(save_path, overlay.astype(np.uint8))

    return save_path