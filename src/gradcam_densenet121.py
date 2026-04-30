from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torchvision.models as models
from torchvision import transforms

from config import GRADCAM_DIR, IMAGE_DIR, IMAGE_SIZE, MODEL_DIR, REPORT_CLASSES

THRESHOLD = 0.5
WEIGHTS_PATH = MODEL_DIR/"best_densenet121_ecg_model.pth"
OUTPUT_DIR = GRADCAM_DIR/"densenet121"

class ECGDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.densenet = models.densenet121(weights=None)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.densenet(x)

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

    def __call__(self, x: torch.Tensor, target_class: int, output_size: tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        return cv2.resize(cam.cpu().numpy(), output_size)

def overlay_heatmap(heatmap: np.ndarray, original: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color = cv2.resize(color, (original.shape[1], original.shape[0]))
    base = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0)

def select_target_classes(scores: torch.Tensor) -> list[int]:
    selected = torch.where(scores >= THRESHOLD)[0].tolist()
    if selected:
        return selected
    return [int(torch.argmax(scores).item())]

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    class_names = checkpoint.get("class_names", REPORT_CLASSES)
    model = ECGDenseNet121(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cam = GradCAM(model)
    image_paths = sorted(path for path in IMAGE_DIR.glob("*.png") if path.stem.isdigit())

    for image_path in tqdm(image_paths, desc="Generating DenseNet-121 Grad-CAM", unit="record"):
        x, original = load_image(image_path)
        x = x.to(device)
        with torch.no_grad():
            scores = model(x)[0]
        for class_index in select_target_classes(scores):
            score = float(scores[class_index].item())
            class_name = class_names[class_index]
            heatmap = cam(x, class_index, (original.shape[1], original.shape[0]))
            overlay = overlay_heatmap(heatmap, original)
            output_path = OUTPUT_DIR/f"record_{image_path.stem}_label_{class_name}_score_{score:.4f}_gradcam.png"
            cv2.imwrite(str(output_path), overlay)

if __name__ == "__main__":
    main()
