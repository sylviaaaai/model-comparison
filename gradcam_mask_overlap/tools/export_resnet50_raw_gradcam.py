"""Export raw ResNet50 Grad-CAM maps for all final test images.

Outputs 224x224 normalized heatmaps as `.npy` plus uint8 PNG previews and
updates `gradcam_mask_overlap/metadata/saliency_manifest.csv` for ResNet50.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "results/resnet50_final_conv3_5_preprocess_dropout04/final/conv3_5/final_model.keras"
DEFAULT_PREDICTIONS = METADATA_DIR / "model_predictions.csv"
DEFAULT_SALIENCY_MANIFEST = METADATA_DIR / "saliency_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "data/saliency_maps/resnet50/gradcam_predclass"
DEFAULT_MALIGNANT_OUTPUT_DIR = EXPERIMENT_ROOT / "data/saliency_maps/resnet50/gradcam_malignant"
RUN_LOG = METADATA_DIR / "runs.jsonl"
IMG_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--saliency-manifest", type=Path, default=DEFAULT_SALIENCY_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--last-conv-layer", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--target-mode",
        choices=["predclass", "malignant"],
        default="predclass",
        help=(
            "predclass explains the model's predicted class; malignant always "
            "explains the Cancer score and is recommended for mask overlap."
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


def load_image(path: Path) -> tf.Tensor:
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)


def find_last_conv_layer(base_model: keras.Model, requested_name: str | None) -> keras.layers.Layer:
    if requested_name:
        return base_model.get_layer(requested_name)
    for layer in reversed(base_model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer
    raise ValueError("Could not find a 4D convolution feature layer in the ResNet50 base model.")


def call_layer(layer: keras.layers.Layer, x: tf.Tensor) -> tf.Tensor:
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def build_grad_model(model: keras.Model, last_conv_layer_name: str | None) -> tuple[keras.Model, str]:
    base_model = model.get_layer("resnet50")
    last_conv_layer = find_last_conv_layer(base_model, last_conv_layer_name)
    feature_extractor = keras.Model(
        base_model.input,
        [last_conv_layer.output, base_model.output],
        name=f"{base_model.name}_{last_conv_layer.name}_features",
    )

    inputs = model.input
    # The final ResNet50 model was trained with keras.applications.resnet50
    # preprocess_input applied inside the Functional graph before augmentation.
    # That TensorFlow op is not listed as a normal Keras layer after reloading,
    # so rebuilding the Grad-CAM path from model.layers must add it explicitly.
    x = keras.applications.resnet50.preprocess_input(inputs)
    conv_output = None
    for layer in model.layers[1:]:
        if layer is base_model:
            conv_output, x = feature_extractor(x, training=False)
        else:
            x = call_layer(layer, x)
    if conv_output is None:
        raise ValueError("Could not connect nested ResNet50 base model for Grad-CAM.")
    return keras.Model(inputs, [conv_output, x], name="resnet50_gradcam_model"), last_conv_layer.name


def make_heatmap(grad_model: keras.Model, image_batch: tf.Tensor, target_class_value: int) -> tuple[np.ndarray, float]:
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(image_batch, training=False)
        cancer_score = prediction[:, 0]
        target_score = cancer_score if target_class_value == 1 else 1.0 - cancer_score

    grads = tape.gradient(target_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + keras.backend.epsilon())
    heatmap_224 = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE, method="bilinear").numpy().squeeze()
    heatmap_224 = np.clip(heatmap_224, 0.0, 1.0).astype(np.float32)
    return heatmap_224, float(cancer_score.numpy()[0])


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
    predictions, _ = read_csv(args.predictions_csv)
    resnet_predictions = [row for row in predictions if row["model_name"] == "resnet50"]
    if args.limit:
        resnet_predictions = resnet_predictions[: args.limit]

    image_paths_by_id: dict[str, Path] = {}
    source_prediction_csvs = sorted({Path(row["source_predictions_csv"]) for row in resnet_predictions})
    for source_csv in source_prediction_csvs:
        source_rows, _ = read_csv(source_csv)
        for source_row in source_rows:
            image_paths_by_id[image_id_from_path(source_row["path"])] = normalize_path(source_row["path"])

    model = keras.models.load_model(args.model_path)
    grad_model, layer_name = build_grad_model(model, args.last_conv_layer)
    print(f"Using Grad-CAM layer: resnet50/{layer_name}")

    manifest_rows, manifest_fields = read_csv(args.saliency_manifest)
    for extra_field in ["recomputed_cancer_probability", "source_cancer_probability"]:
        if extra_field not in manifest_fields:
            manifest_fields.append(extra_field)
    saliency_method = "gradcam" if args.target_mode == "predclass" else "gradcam_malignant"
    manifest_by_prediction = {
        row["prediction_id"]: row
        for row in manifest_rows
        if row.get("model_name") == "resnet50" and row.get("saliency_method") == saliency_method
    }
    existing_saliency_ids = {row["saliency_id"] for row in manifest_rows}

    exported = 0
    skipped_existing = 0
    failed = 0
    new_manifest_rows: list[dict[str, object]] = []

    for index, row in enumerate(resnet_predictions, start=1):
        image_id = row["image_id"]
        prediction_id = row["prediction_id"]
        image_path = image_paths_by_id.get(image_id)
        if image_path is None or not image_path.exists():
            print(f"[{index}/{len(resnet_predictions)}] missing image for {image_id}")
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
            heatmap, probability = make_heatmap(grad_model, load_image(image_path), target_class_value)
            npy_path, png_path = save_heatmap(heatmap, output_base)
            exported += 1
            if exported % 25 == 0 or exported == 1:
                print(f"Exported {exported}/{len(resnet_predictions)}: {image_id}")

        manifest_row = manifest_by_prediction.get(prediction_id)
        saliency_id = saliency_id_for_prediction(prediction_id, args.target_mode)
        if manifest_row is None:
            manifest_row = {
                "saliency_id": saliency_id,
                "image_id": image_id,
                "prediction_id": prediction_id,
                "model_name": "resnet50",
                "architecture": row.get("architecture", "CNN"),
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
                "saliency_path": str(npy_path),
                "saliency_width": IMG_SIZE[0],
                "saliency_height": IMG_SIZE[1],
                "saliency_normalization": "minmax_0_1_after_relu_resized_224",
                "status": "raw_map_available",
                "target_class_for_cam": target_class_for_cam,
                "target_class_value": target_class_value,
                "source_cancer_probability": f"{source_probability:.8f}",
            }
        )
        if probability is not None:
            manifest_row["recomputed_cancer_probability"] = f"{probability:.8f}"

    if new_manifest_rows:
        manifest_rows.extend(new_manifest_rows)
    write_csv(args.saliency_manifest, manifest_rows, manifest_fields)

    summary = {
        "run_id": f"resnet50_raw_gradcam_{args.target_mode}_export_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_type": "resnet50_raw_gradcam_export",
        "target_mode": args.target_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "gradcam_layer": f"resnet50/{layer_name}",
        "processed_predictions": len(resnet_predictions),
        "exported": exported,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }
    append_run_log(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
