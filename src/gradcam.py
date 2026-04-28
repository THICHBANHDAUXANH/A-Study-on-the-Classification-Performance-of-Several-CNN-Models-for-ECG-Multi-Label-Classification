from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torchvision import transforms

from config import GRADCAM_DIR, IMAGE_DIR, IMAGE_SIZE, MODEL_DIR, REPORT_CLASSES

ECG_ID = 189
TARGET_CLASS_INDEX = 3
WEIGHTS_PATH = MODEL_DIR/"best_resnet50_ecg_model.pth"
IMAGE_PATH = IMAGE_DIR/f"{ECG_ID}.png"

class ECGResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.resnet(x)

def load_image(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    return transform(image).unsqueeze(0), original

class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_module = self._last_conv_layer()
        self.target_module.register_forward_hook(self._save_activation)
        self.target_module.register_full_backward_hook(self._save_gradient)

    def _last_conv_layer(self) -> nn.Module:
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        return last_conv

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int, output_size: tuple[int, int]) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        score = output[0, target_class]
        probability = float(score.item())
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        heatmap = cv2.resize(cam.cpu().numpy(), output_size)
        return heatmap, probability

def overlay_heatmap(heatmap: np.ndarray, original: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color = cv2.resize(color, (original.shape[1], original.shape[0]))
    base = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0)

def main() -> None:
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    class_names = checkpoint.get("class_names", REPORT_CLASSES)
    model = ECGResNet50(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x, original = load_image(IMAGE_PATH)
    x = x.to(device)
    cam = GradCAM(model)
    heatmap, probability = cam(x, TARGET_CLASS_INDEX, (original.shape[1], original.shape[0]))
    overlay = overlay_heatmap(heatmap, original)

    class_name = class_names[TARGET_CLASS_INDEX]
    output_path = GRADCAM_DIR/f"{ECG_ID}_{class_name}_gradcam.png"
    cv2.imwrite(str(output_path), overlay)
    print(f"Saved Grad-CAM: {output_path}")
    print(f"Target class: {class_name}, score={probability:.4f}")

if __name__ == "__main__":
    main()
