"""Render case-centric smoke QC panels for lesion-mask overlap analysis."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_ANALYSIS = METADATA_DIR / "analysis_ready_overlap_manifest.csv"
DEFAULT_IMAGES = METADATA_DIR / "test_images_manifest.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "qc_figures" / "overlap_check"
ROI_SIZE = 224

PRIMARY_METHODS = {
    "resnet50": "gradcam_malignant",
    "convnext_base": "gradcam_malignant",
    "swin_base": "gradcam_malignant",
    "vit_b16": "attention_rollout",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value <= min_value:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_value) / (max_value - min_value)


def load_gray(path: str) -> np.ndarray:
    image = Image.open(path).convert("L").resize((ROI_SIZE, ROI_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    image = Image.open(path).convert("L").resize((ROI_SIZE, ROI_SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def mask_contour(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
    )
    return mask & ~neighbors


def heat_overlay(base: np.ndarray, heatmap: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    cmap = plt.get_cmap("magma")
    heat_rgb = cmap(normalize01(heatmap))[..., :3]
    base_rgb = np.repeat(base[..., None], 3, axis=2)
    overlay = 0.55 * base_rgb + 0.45 * heat_rgb
    if mask is not None:
        overlay[mask_contour(mask)] = np.array([0.0, 1.0, 1.0])
    return np.clip(overlay, 0, 1)


def binary_saliency(heatmap: np.ndarray, top_fraction: float = 0.10) -> np.ndarray:
    heatmap = normalize01(heatmap)
    if float(heatmap.sum()) <= 1e-12 or float(np.max(heatmap)) <= 1e-12:
        return np.zeros_like(heatmap, dtype=bool)
    flat = heatmap.reshape(-1)
    n_pixels = flat.size
    k = int(np.ceil(top_fraction * n_pixels))
    k = max(1, min(k, n_pixels))
    top_idx = np.argpartition(flat, -k)[-k:]
    binary_flat = np.zeros(n_pixels, dtype=bool)
    binary_flat[top_idx] = True
    return binary_flat.reshape(heatmap.shape)


def binary_mask_overlay(base: np.ndarray, mask: np.ndarray, binary: np.ndarray) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=2)
    rgb[mask] = 0.55 * rgb[mask] + np.array([0.45, 0.0, 0.0])
    rgb[binary] = 0.55 * rgb[binary] + np.array([0.0, 0.45, 0.0])
    rgb[mask & binary] = np.array([1.0, 0.95, 0.0])
    rgb[mask_contour(mask)] = np.array([0.0, 1.0, 1.0])
    return np.clip(rgb, 0, 1)


def primary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("ready_for_overlap") == "True"
        and PRIMARY_METHODS.get(row["model_name"]) == row["saliency_method"]
    ]


def sample_cases(
    rows: list[dict[str, str]],
    reference_model: str,
    sample_per_group: int,
    group_image_ids: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row["model_name"] == reference_model
        and row["saliency_method"] == PRIMARY_METHODS[reference_model]
    ]
    by_group_image = {
        (row["prediction_group"], row["image_id"]): row
        for row in candidates
    }
    sampled: list[dict[str, str]] = []
    for group in ["TP", "TN", "FP", "FN"]:
        if group_image_ids and group in group_image_ids:
            sampled.extend(by_group_image[(group, image_id)] for image_id in group_image_ids[group])
            continue
        group_rows = sorted(
            [row for row in candidates if row["prediction_group"] == group],
            key=lambda row: row["image_id"],
        )
        sampled.extend(group_rows[:sample_per_group])
    return sampled


def parse_group_image_ids(values: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for value in values:
        group, image_ids = value.split(":", 1)
        parsed[group] = [image_id for image_id in image_ids.split(",") if image_id]
    return parsed


def render_case(
    image_id: str,
    group: str,
    rows_by_model: dict[str, dict[str, str]],
    roi_image_path: str,
    output_path: Path,
    top_fraction: float,
) -> None:
    base = load_gray(roi_image_path)
    first_row = next(iter(rows_by_model.values()))
    mask = load_mask(first_row["roi_mask_path"])
    top_percent = int(round(top_fraction * 100))

    columns = [
        "ROI image",
        "lesion mask",
        "raw heatmap",
        "heatmap overlay",
        f"top-{top_percent}% saliency",
        "mask + saliency",
    ]
    fig, axes = plt.subplots(
        len(PRIMARY_METHODS),
        len(columns),
        figsize=(18, 12),
        constrained_layout=True,
    )
    fig.suptitle(f"{group} | {image_id}", fontsize=14)

    for row_index, (model_name, method) in enumerate(PRIMARY_METHODS.items()):
        row = rows_by_model[model_name]
        heatmap = normalize01(np.load(row["saliency_path"]))
        binary = binary_saliency(heatmap, top_fraction=top_fraction)
        panel_values = [
            base,
            mask.astype(np.float32),
            heatmap,
            heat_overlay(base, heatmap, mask),
            binary.astype(np.float32),
            binary_mask_overlay(base, mask, binary),
        ]
        for col_index, value in enumerate(panel_values):
            ax = axes[row_index, col_index]
            if value.ndim == 2:
                ax.imshow(value, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(value)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(columns[col_index], fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{model_name}\n{method}", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-per-group", type=int, default=3)
    parser.add_argument("--reference-model", default="resnet50", choices=sorted(PRIMARY_METHODS))
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--group-image-ids",
        action="append",
        default=[],
        help="Override sampled image IDs for a group, e.g. FN:id1,id2,id3",
    )
    args = parser.parse_args()

    analysis_rows = primary_rows(read_csv(args.analysis_manifest))
    image_rows = {row["image_id"]: row for row in read_csv(args.image_manifest)}
    by_image_model = {
        (row["image_id"], row["model_name"]): row
        for row in analysis_rows
    }

    sampled = sample_cases(
        analysis_rows,
        args.reference_model,
        args.sample_per_group,
        parse_group_image_ids(args.group_image_ids),
    )
    index_rows: list[dict[str, object]] = []
    for sample in sampled:
        image_id = sample["image_id"]
        rows_by_model = {
            model_name: by_image_model[(image_id, model_name)]
            for model_name in PRIMARY_METHODS
        }
        group = sample["prediction_group"]
        output_path = args.output_dir / group / f"{image_id}_overlap_smoke_qc.png"
        render_case(
            image_id=image_id,
            group=group,
            rows_by_model=rows_by_model,
            roi_image_path=image_rows[image_id]["roi_image_path"],
            output_path=output_path,
            top_fraction=args.top_fraction,
        )
        index_rows.append(
            {
                "image_id": image_id,
                "reference_model": args.reference_model,
                "reference_prediction_group": group,
                "output_path": str(output_path),
            }
        )

    write_csv(
        args.output_dir / "sampled_cases_index.csv",
        index_rows,
        ["image_id", "reference_model", "reference_prediction_group", "output_path"],
    )
    print(f"Rendered {len(index_rows)} QC panels to {args.output_dir}")


if __name__ == "__main__":
    main()
