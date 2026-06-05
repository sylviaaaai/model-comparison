#!/usr/bin/env python3
"""Generate Grad-CAM visualizations for the final ResNet50 ROI classifier."""

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
import tensorflow as tf
from tensorflow import keras


IMG_SIZE = (224, 224)
DEFAULT_EXPERIMENT_DIR = Path("results/resnet50_final_conv3_5_preprocess_dropout04")
DEFAULT_MODEL_PATH = DEFAULT_EXPERIMENT_DIR / "final/conv3_5/final_model.keras"
DEFAULT_SPLIT_CSV = Path("results/splits/final_80_10_10/final_test_images.csv")
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "gradcam"
CLASS_NAMES = {0: "Benign", 1: "Cancer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM images for the dropout=0.4 ResNet50 final model."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-group", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the Cancer class.",
    )
    parser.add_argument(
        "--last-conv-layer",
        default=None,
        help="Optional ResNet50 convolution layer name. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of test images to scan.",
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


def load_image(path: Path) -> tuple[np.ndarray, tf.Tensor]:
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32)
    return image.numpy().astype(np.uint8), tf.expand_dims(image, axis=0)


def find_last_conv_layer(base_model: keras.Model, requested_name: str | None) -> keras.layers.Layer:
    if requested_name:
        return base_model.get_layer(requested_name)

    for layer in reversed(base_model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            try:
                output_shape = layer.output.shape
            except AttributeError:
                continue
        if len(output_shape) == 4:
            return layer

    raise ValueError("Could not find a 4D convolution feature layer in the ResNet50 base model.")


def call_layer(layer: keras.layers.Layer, x: tf.Tensor) -> tf.Tensor:
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def build_grad_model(model: keras.Model, last_conv_layer_name: str | None) -> keras.Model:
    base_model = model.get_layer("resnet50")
    last_conv_layer = find_last_conv_layer(base_model, last_conv_layer_name)
    feature_extractor = keras.Model(
        base_model.input,
        [last_conv_layer.output, base_model.output],
        name=f"{base_model.name}_{last_conv_layer.name}_features",
    )

    inputs = model.input
    x = inputs
    conv_output = None
    for layer in model.layers[1:]:
        if layer is base_model:
            conv_output, x = feature_extractor(x, training=False)
        else:
            x = call_layer(layer, x)

    if conv_output is None:
        raise ValueError("Could not connect the nested ResNet50 base model for Grad-CAM.")

    grad_model = keras.Model(inputs, [conv_output, x], name="resnet50_gradcam_model")
    print(f"Using Grad-CAM layer: resnet50/{last_conv_layer.name}")
    return grad_model


def make_heatmap(grad_model: keras.Model, image_batch: tf.Tensor) -> tuple[np.ndarray, float]:
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(image_batch, training=False)
        cancer_score = prediction[:, 0]

    grads = tape.gradient(cancer_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + keras.backend.epsilon())
    return heatmap.numpy(), float(cancer_score.numpy()[0])


def save_gradcam_panel(
    original: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    alpha: float = 0.40,
) -> None:
    resized_heatmap = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    cmap = plt.get_cmap("jet")
    colored = cmap(resized_heatmap)[..., :3]
    overlay = np.clip((1 - alpha) * (original / 255.0) + alpha * colored, 0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=160)
    panels = [
        ("Original", original / 255.0),
        ("Grad-CAM", resized_heatmap),
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
    model = keras.models.load_model(args.model_path)
    grad_model = build_grad_model(model, args.last_conv_layer)
    records = read_split(args.split_csv, args.limit)

    counts = {
        "correct_cancer": 0,
        "correct_benign": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    summary_rows = []

    for index, record in enumerate(records, start=1):
        image_path = record["path"]
        if not Path(image_path).exists():
            print(f"Skipping missing image: {image_path}")
            continue

        original, image_batch = load_image(Path(image_path))
        heatmap, probability = make_heatmap(grad_model, image_batch)
        label = int(record["label"])
        predicted = int(probability >= args.threshold)
        group = group_name(label, predicted)

        if counts[group] < args.max_per_group:
            counts[group] += 1
            stem = Path(image_path).stem.replace(" ", "_")
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
