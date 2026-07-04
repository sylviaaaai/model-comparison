"""Export raw ConvNeXt-Base Grad-CAM maps for all final test images.

Outputs 224x224 normalized heatmaps as `.npy` plus uint8 PNG previews and
updates `gradcam_mask_overlap/metadata/saliency_manifest.csv` for ConvNeXt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("This script requires timm. Install with: pip install timm") from exc


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "ConvNeXt/results/convnext_base_torch_final_stage3_4_dropout04/final/stage3_4/final_model.pt"
DEFAULT_PREDICTIONS = METADATA_DIR / "model_predictions.csv"
DEFAULT_SALIENCY_MANIFEST = METADATA_DIR / "saliency_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "data/saliency_maps/convnext_base/gradcam_predclass"
DEFAULT_MALIGNANT_OUTPUT_DIR = EXPERIMENT_ROOT / "data/saliency_maps/convnext_base/gradcam_malignant"
RUN_LOG = METADATA_DIR / "runs.jsonl"
IMG_SIZE = 224
DEFAULT_MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
DEFAULT_TARGET_LAYER = "stages.3.blocks.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--saliency-manifest", type=Path, default=DEFAULT_SALIENCY_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--target-mode",
        choices=["predclass", "malignant"],
        default="predclass",
        help=(
            "predclass explains the model's predicted class; malignant always "
            "explains the Cancer logit and is recommended for mask overlap."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_MALIGNANT_OUTPUT_DIR if args.target_mode == "malignant" else DEFAULT_OUTPUT_DIR
    return args


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
    mapped = Path(raw_path.replace("\\", "/"))
    if mapped.exists():
        return mapped
    return path


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_image(path: Path, transform: transforms.Compose) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def build_model(model_name: str, model_path: Path, device: torch.device) -> nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=0.0)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_module(model: nn.Module, dotted_name: str) -> nn.Module:
    module: nn.Module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)  # type: ignore[index]
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

    def __call__(self, image_batch: torch.Tensor, device: torch.device, target_class_value: int) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_batch.to(device)).reshape(-1)
        cancer_logit = logits[0]
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
        probability = torch.sigmoid(cancer_logit.detach()).item()
        return cam.cpu().numpy().astype(np.float32), float(probability)


def image_id_from_path(path: str) -> str:
    return normalize_path(path).stem


def saliency_id_for_prediction(prediction_id: str, target_mode: str) -> str:
    suffix = "gradcam__predclass" if target_mode == "predclass" else "gradcam_malignant__cancer_score"
    return f"{prediction_id}__{suffix}"


def saliency_metadata_for_mode(row: dict[str, str], target_mode: str) -> tuple[str, str, int]:
    if target_mode == "predclass":
        return "gradcam", "predicted_label", int(row["predicted_label"])
    return "gradcam_malignant", "cancer_score", 1


def save_heatmap(heatmap: np.ndarray, output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    npy_path = Path(str(output_base) + ".npy")
    png_path = Path(str(output_base) + ".png")
    np.save(npy_path, heatmap)
    Image.fromarray((heatmap * 255).round().astype(np.uint8), mode="L").save(png_path)
    return npy_path, png_path


def append_run_log(summary: dict[str, object]) -> None:
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    predictions, _ = read_csv(args.predictions_csv)
    convnext_predictions = [row for row in predictions if row["model_name"] == "convnext_base"]
    if args.limit:
        convnext_predictions = convnext_predictions[: args.limit]

    image_paths_by_id: dict[str, Path] = {}
    source_prediction_csvs = sorted({Path(row["source_predictions_csv"]) for row in convnext_predictions})
    for source_csv in source_prediction_csvs:
        source_rows, _ = read_csv(source_csv)
        for source_row in source_rows:
            image_paths_by_id[image_id_from_path(source_row["path"])] = normalize_path(source_row["path"])

    model = build_model(args.model_name, args.model_path, device)
    target_layer = get_module(model, args.target_layer)
    gradcam = GradCam(model, target_layer)
    transform = make_transform()
    print(f"Using Grad-CAM layer: {args.target_layer}")
    print(f"Device: {device}")

    manifest_rows, manifest_fields = read_csv(args.saliency_manifest)
    for extra_field in ["recomputed_cancer_probability", "source_cancer_probability"]:
        if extra_field not in manifest_fields:
            manifest_fields.append(extra_field)
    saliency_method = "gradcam" if args.target_mode == "predclass" else "gradcam_malignant"
    manifest_by_prediction = {
        row["prediction_id"]: row
        for row in manifest_rows
        if row.get("model_name") == "convnext_base" and row.get("saliency_method") == saliency_method
    }
    existing_saliency_ids = {row["saliency_id"] for row in manifest_rows}

    exported = 0
    skipped_existing = 0
    failed = 0
    new_manifest_rows: list[dict[str, object]] = []

    try:
        for index, row in enumerate(convnext_predictions, start=1):
            image_id = row["image_id"]
            prediction_id = row["prediction_id"]
            image_path = image_paths_by_id.get(image_id)
            if image_path is None or not image_path.exists():
                print(f"[{index}/{len(convnext_predictions)}] missing image for {image_id}")
                failed += 1
                continue

            saliency_method, target_class_for_cam, target_class_value = saliency_metadata_for_mode(row, args.target_mode)
            source_probability = float(row["cancer_probability"])
            output_base = args.output_dir / image_id[:4] / image_id
            npy_path = Path(str(output_base) + ".npy")
            png_path = Path(str(output_base) + ".png")
            probability: float | None = None
            if args.skip_existing and npy_path.exists() and png_path.exists():
                skipped_existing += 1
            else:
                heatmap, probability = gradcam(load_image(image_path, transform), device, target_class_value)
                npy_path, png_path = save_heatmap(heatmap, output_base)
                exported += 1
                if exported % 25 == 0 or exported == 1:
                    print(f"Exported {exported}/{len(convnext_predictions)}: {image_id}")

            manifest_row = manifest_by_prediction.get(prediction_id)
            saliency_id = saliency_id_for_prediction(prediction_id, args.target_mode)
            if manifest_row is None:
                manifest_row = {
                    "saliency_id": saliency_id,
                    "image_id": image_id,
                    "prediction_id": prediction_id,
                    "model_name": "convnext_base",
                    "architecture": row.get("architecture", "Modern CNN"),
                    "saliency_method": saliency_method,
                    "target_class_for_cam": target_class_for_cam,
                    "target_class_value": target_class_value,
                    "source_summary_csv": "",
                }
                if saliency_id not in existing_saliency_ids:
                    new_manifest_rows.append(manifest_row)
                    existing_saliency_ids.add(saliency_id)
            manifest_row.update(
                {
                    "saliency_id": saliency_id,
                    "image_id": image_id,
                    "prediction_id": prediction_id,
                    "model_name": "convnext_base",
                    "architecture": row.get("architecture", "Modern CNN"),
                    "saliency_method": saliency_method,
                    "target_class_for_cam": target_class_for_cam,
                    "target_class_value": target_class_value,
                    "saliency_path": str(npy_path),
                    "saliency_width": IMG_SIZE,
                    "saliency_height": IMG_SIZE,
                    "saliency_normalization": "minmax_0_1_after_relu_resized_224",
                    "status": "raw_map_available",
                    "source_cancer_probability": f"{source_probability:.8f}",
                }
            )
            if probability is not None:
                manifest_row["recomputed_cancer_probability"] = f"{probability:.8f}"
    finally:
        gradcam.close()

    if new_manifest_rows:
        manifest_rows.extend(new_manifest_rows)
    write_csv(args.saliency_manifest, manifest_rows, manifest_fields)

    summary = {
        "run_id": "convnext_raw_gradcam_export_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_type": "convnext_raw_gradcam_export",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "target_mode": args.target_mode,
        "gradcam_layer": args.target_layer,
        "processed_predictions": len(convnext_predictions),
        "exported": exported,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }
    append_run_log(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
