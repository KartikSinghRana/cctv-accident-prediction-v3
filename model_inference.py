from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


CLASS_NAMES = ["No_Accident", "Minor_Accident", "Severe_Accident"]
MODEL_NAME = "tf_efficientnetv2_s"
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class AccidentCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.175),
            nn.Linear(256, len(CLASS_NAMES)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
        return self.classifier(features)


def load_trained_model(checkpoint_path: str) -> tuple[AccidentCNN, torch.device]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes != CLASS_NAMES:
        raise ValueError(
            "The checkpoint class order does not match the dashboard's expected "
            f"classes: {checkpoint_classes!r}."
        )

    model = AccidentCNN().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, device


def _normalized_map(values: torch.Tensor) -> np.ndarray:
    values = values.detach().squeeze().float().cpu()
    values = F.interpolate(
        values[None, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    minimum = values.min()
    maximum = values.max()
    if maximum <= minimum:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).numpy()


def _heatmap_overlay(image: Image.Image, activation_map: np.ndarray) -> np.ndarray:
    base = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))).astype(
        np.float32
    )
    red = np.clip(1.5 - np.abs(4 * activation_map - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * activation_map - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * activation_map - 1), 0, 1)
    heatmap = np.stack((red, green, blue), axis=-1) * 255
    return np.clip(base * 0.55 + heatmap * 0.45, 0, 255).astype(np.uint8)


def predict_with_explanations(
    model: AccidentCNN,
    device: torch.device,
    image: Image.Image,
    target_class: str | None = None,
) -> dict[str, object]:
    rgb_image = np.asarray(image.convert("RGB"))
    resized_image = cv2.resize(
        rgb_image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    display_image = Image.fromarray(resized_image)
    input_tensor = (
        torch.from_numpy(resized_image.transpose(2, 0, 1).copy())
        .float()
        .div_(255.0)
    )
    mean = torch.tensor(IMAGE_MEAN, dtype=input_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=input_tensor.dtype).view(3, 1, 1)
    input_tensor = ((input_tensor - mean) / std).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    activations: list[torch.Tensor] = []
    target_layer = model.backbone.blocks[-1]
    hook = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        if target_class is None:
            target_index = predicted_index
        else:
            try:
                target_index = CLASS_NAMES.index(target_class)
            except ValueError as error:
                raise ValueError(f"Unsupported explanation class: {target_class}") from error

        activation_gradient, input_gradient = torch.autograd.grad(
            logits[0, target_index],
            (activations[0], input_tensor),
        )
    finally:
        hook.remove()

    activation = activations[0].detach()
    weights = activation_gradient.mean(dim=(2, 3), keepdim=True)
    gradcam_map = F.relu((weights * activation).sum(dim=1, keepdim=True))
    gradcam_map = _normalized_map(gradcam_map)

    saliency_map = input_gradient.detach().abs().mean(dim=1, keepdim=True)
    saliency_map = _normalized_map(saliency_map)
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "display_image": np.asarray(display_image),
        "gradcam_overlay": _heatmap_overlay(display_image, gradcam_map),
        "saliency_overlay": _heatmap_overlay(display_image, saliency_map),
    }
