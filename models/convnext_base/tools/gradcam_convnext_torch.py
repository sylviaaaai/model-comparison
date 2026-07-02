#!/usr/bin/env python3
"""Generate Grad-CAM visualizations for the final ConvNeXt ROI classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PureWindowsPath

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("This script requires timm. Install with: pip install timm") from exc


IMG_SIZE = 224
DEFAULT_EXPERIMENT_DIR = Path("ConvNeXt/results/convnext_base_torch_final_stage3_4_dropout04")
DEFAULT_MODEL_PATH = DEFAULT_EXPERIMENT_DIR / "final/stage3_4/final_model.pt"
DEFAULT_SPLIT_CSV = Path("results/splits/final_80_10_10/final_test_images.csv")
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "gradcam"
DEFAULT_MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
DEFAULT_TARGET_LAYER = "stages.3.blocks.2"
CLASS_NAMES = {0: "Benign", 1: "Cancer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM images for the ConvNeXt final model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--max-per-group", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Create the architecture without downloading pretrained weights before loading the checkpoint.",
    )
    return parser.parse_args()


def normalize_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    if len(raw_path) >= 3 and raw_path[1:3] == ":\\":
        win_path = PureWindowsPath(raw_path)
        drive = win_path.drive.rstrip(":").lower()
        mapped = Path("/mnt") / drive / Path(*win_path.parts[1:])
        if mapped.exists():
            return mapped

    backslash_mapped = Path(raw_path.replace("\\", "/"))
    if backslash_mapped.exists():
        return backslash_mapped

    return path


def read_split(path: Path, limit: int | None) -> list[dict[str, object]]:
    records = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["path"] = normalize_path(row["path"])
            row["label"] = int(row["label"])
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    return records


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_image(path: Path, transform) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    original = np.asarray(image).astype(np.uint8)
    tensor = transform(image).unsqueeze(0)
    return original, tensor


def build_model(model_name: str, model_path: Path, pretrained: bool, device: torch.device) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=1, drop_rate=0.0)
    # The final-training script stores a local trusted checkpoint with metadata
    # such as pathlib.Path values, so PyTorch 2.6+ needs weights_only=False.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_module(model: nn.Module, dotted_name: str) -> nn.Module:
    module: nn.Module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


class GradCam:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_batch: torch.Tensor, device: torch.device, threshold: float) -> tuple[np.ndarray, float, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_batch.to(device)).reshape(-1)
        cancer_logit = logits[0]
        probability = torch.sigmoid(cancer_logit.detach()).item()
        target_class_value = int(probability >= threshold)
        target_score = cancer_logit if target_class_value == 1 else -cancer_logit
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), float(probability), target_class_value


def save_gradcam_panel(
    original: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    alpha: float = 0.40,
) -> None:
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]
    overlay = np.clip((1 - alpha) * (original / 255.0) + alpha * colored, 0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=160)
    panels = [
        ("Original", original / 255.0),
        ("Grad-CAM", heatmap),
        ("Overlay", overlay),
    ]
    for ax, (name, image) in zip(axes, panels):
        ax.imshow(image, cmap="jet" if name == "Grad-CAM" else None)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def group_name(label: int, predicted: int) -> str:
    if label == 1 and predicted == 1:
        return "correct_cancer"
    if label == 0 and predicted == 0:
        return "correct_benign"
    if label == 0 and predicted == 1:
        return "false_positive"
    return "false_negative"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(args.model_name, args.model_path, not args.no_pretrained, device)
    target_layer = get_module(model, args.target_layer)
    gradcam = GradCam(model, target_layer)
    transform = make_transform()
    records = read_split(args.split_csv, args.limit)
    print(f"Using Grad-CAM layer: {args.target_layer}")
    print(f"Device: {device}")

    counts = {
        "correct_cancer": 0,
        "correct_benign": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    summary_rows = []

    try:
        for index, record in enumerate(records, start=1):
            image_path = Path(record["path"])
            if not image_path.exists():
                print(f"Skipping missing image: {image_path}")
                continue

            original, image_batch = load_image(image_path, transform)
            heatmap, probability, predicted = gradcam(image_batch, device, args.threshold)
            label = int(record["label"])
            group = group_name(label, predicted)

            if counts[group] < args.max_per_group:
                counts[group] += 1
                stem = image_path.stem.replace(" ", "_")
                output_path = args.output_dir / group / f"{counts[group]:02d}_{stem}.png"
                title = (
                    f"{group} | true={CLASS_NAMES[label]} | "
                    f"pred={CLASS_NAMES[predicted]} | p(Cancer)={probability:.3f}"
                )
                save_gradcam_panel(original, heatmap, output_path, title)
                print(f"Saved {output_path}")

            summary_rows.append(
                {
                    "index": index,
                    "path": str(image_path),
                    "true_label": label,
                    "true_class": CLASS_NAMES[label],
                    "predicted_label": predicted,
                    "predicted_class": CLASS_NAMES[predicted],
                    "cancer_probability": f"{probability:.8f}",
                    "group": group,
                }
            )

            if all(value >= args.max_per_group for value in counts.values()):
                break
    finally:
        gradcam.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "path",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "cancer_probability",
            "group",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = args.output_dir / "summary_counts.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    print(f"Saved summary: {summary_csv}")
    print(f"Saved counts: {summary_json}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
