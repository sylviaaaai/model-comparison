"""Compute lesion-mask overlap metrics for primary saliency maps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_INPUT = METADATA_DIR / "analysis_ready_overlap_manifest.csv"
DEFAULT_OUTPUT = METADATA_DIR / "overlap_metrics.csv"
DEFAULT_SUMMARY = METADATA_DIR / "overlap_metrics_summary.json"
ROI_SIZE = 224
EPS = 1e-12

PRIMARY_METHODS = {
    "resnet50": "gradcam_malignant",
    "convnext_base": "gradcam_malignant",
    "swin_base": "gradcam_malignant",
    "vit_b16": "attention_rollout",
}

METHOD_META = {
    "gradcam_malignant": ("Grad-CAM", True),
    "attention_rollout": ("Attention rollout", False),
}

FIXED_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOP_FRACTIONS = [0.10, 0.20, 0.30, 0.40]


BASE_FIELDS = [
    "analysis_id",
    "image_id",
    "prediction_id",
    "model_name",
    "architecture",
    "model_display_name",
    "saliency_method",
    "explanation_family",
    "class_specific",
    "true_label",
    "true_class",
    "predicted_label",
    "predicted_class",
    "prediction_group",
    "prediction_group_name",
    "cancer_probability",
    "manual_annotation_type",
    "manual_experiment_group",
    "mask_id",
    "saliency_id",
    "threshold_type",
    "threshold_value",
    "threshold_label",
    "cam_energy_in_mask",
    "normalized_energy_in_mask",
    "peak_in_mask",
    "peak_x",
    "peak_y",
    "peak_value",
    "peak_distance_to_mask_px",
    "mask_area_px",
    "mask_area_fraction",
    "cam_sum",
    "iou",
    "dice",
    "mask_coverage_by_cam",
    "saliency_precision",
    "cam_area_px",
    "cam_area_fraction",
    "intersection_area_px",
    "union_area_px",
    "quality_status",
    "quality_note",
]


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


def load_mask(path: str) -> np.ndarray:
    image = Image.open(path).convert("L").resize((ROI_SIZE, ROI_SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def selected_rows(rows: list[dict[str, str]], include_all: bool) -> list[dict[str, str]]:
    if include_all:
        return [row for row in rows if row.get("ready_for_overlap") == "True"]
    return [
        row
        for row in rows
        if row.get("ready_for_overlap") == "True"
        and PRIMARY_METHODS.get(row["model_name"]) == row["saliency_method"]
    ]


def continuous_metrics(cam: np.ndarray, mask: np.ndarray) -> tuple[dict[str, object], list[str]]:
    notes: list[str] = []
    cam = np.asarray(cam, dtype=np.float32)
    if not np.isfinite(cam).all():
        notes.append("nonfinite_saliency")
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    zero_saliency = float(cam.sum()) <= EPS or float(cam.max()) <= EPS
    if cam.shape != (ROI_SIZE, ROI_SIZE):
        notes.append(f"saliency_shape_{cam.shape}")
    if mask.shape != (ROI_SIZE, ROI_SIZE):
        notes.append(f"mask_shape_{mask.shape}")
    if not mask.any():
        notes.append("empty_mask")
    if zero_saliency:
        notes.append("zero_saliency")

    cam = normalize01(cam)
    mask_area_px = int(mask.sum())
    mask_area_fraction = mask_area_px / float(mask.size)
    cam_sum = float(cam.sum())
    cam_energy_in_mask = float(cam[mask].sum() / (cam_sum + EPS)) if mask_area_px else 0.0
    normalized_energy = float(cam_energy_in_mask / (mask_area_fraction + EPS))
    if zero_saliency:
        return (
            {
                "cam_energy_in_mask": 0.0,
                "normalized_energy_in_mask": 0.0,
                "peak_in_mask": False,
                "peak_x": "",
                "peak_y": "",
                "peak_value": 0.0,
                "peak_distance_to_mask_px": float("nan"),
                "mask_area_px": mask_area_px,
                "mask_area_fraction": mask_area_fraction,
                "cam_sum": 0.0,
            },
            notes,
        )

    peak_flat = int(np.argmax(cam))
    peak_y, peak_x = np.unravel_index(peak_flat, cam.shape)
    peak_in_mask = bool(mask[peak_y, peak_x]) if mask.size else False
    if mask.any():
        peak_distance = float(ndimage.distance_transform_edt(~mask)[peak_y, peak_x])
    else:
        peak_distance = float("nan")

    return (
        {
            "cam_energy_in_mask": cam_energy_in_mask,
            "normalized_energy_in_mask": normalized_energy,
            "peak_in_mask": peak_in_mask,
            "peak_x": int(peak_x),
            "peak_y": int(peak_y),
            "peak_value": float(cam[peak_y, peak_x]),
            "peak_distance_to_mask_px": peak_distance,
            "mask_area_px": mask_area_px,
            "mask_area_fraction": mask_area_fraction,
            "cam_sum": cam_sum,
        },
        notes,
    )


def threshold_metrics(binary: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    cam_area_px = int(binary.sum())
    mask_area_px = int(mask.sum())
    intersection = int((binary & mask).sum())
    union = int((binary | mask).sum())
    return {
        "iou": intersection / (union + EPS),
        "dice": (2.0 * intersection) / (cam_area_px + mask_area_px + EPS),
        "mask_coverage_by_cam": intersection / (mask_area_px + EPS),
        "saliency_precision": intersection / (cam_area_px + EPS),
        "cam_area_px": cam_area_px,
        "cam_area_fraction": cam_area_px / float(binary.size),
        "intersection_area_px": intersection,
        "union_area_px": union,
    }


def top_fraction_binary(cam: np.ndarray, fraction: float) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    if float(cam.sum()) <= EPS or float(cam.max()) <= EPS:
        return np.zeros_like(cam, dtype=bool)
    flat = cam.reshape(-1)
    n_pixels = flat.size
    k = int(np.ceil(fraction * n_pixels))
    k = max(1, min(k, n_pixels))
    top_idx = np.argpartition(flat, -k)[-k:]
    binary_flat = np.zeros(n_pixels, dtype=bool)
    binary_flat[top_idx] = True
    return binary_flat.reshape(cam.shape)


def format_float(value: object) -> object:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.10g}"
    return value


def metric_rows_for_analysis(row: dict[str, str]) -> tuple[list[dict[str, object]], list[str]]:
    cam = np.load(row["saliency_path"])
    mask = load_mask(row["roi_mask_path"])
    continuous, notes = continuous_metrics(cam, mask)
    cam = normalize01(cam)
    explanation_family, class_specific = METHOD_META.get(row["saliency_method"], ("unknown", ""))
    base = {
        field: row.get(field, "")
        for field in BASE_FIELDS
        if field in row
    }
    base.update(
        {
            "explanation_family": explanation_family,
            "class_specific": class_specific,
            **continuous,
            "quality_status": "ok" if not notes else "check",
            "quality_note": ";".join(notes),
        }
    )

    output_rows: list[dict[str, object]] = []
    for threshold in FIXED_THRESHOLDS:
        binary = cam >= threshold
        metric_row = dict(base)
        metric_row.update(
            {
                "threshold_type": "fixed",
                "threshold_value": threshold,
                "threshold_label": f"fixed_{threshold:.1f}",
                **threshold_metrics(binary, mask),
            }
        )
        output_rows.append({key: format_float(value) for key, value in metric_row.items()})

    for fraction in TOP_FRACTIONS:
        binary = top_fraction_binary(cam, fraction)
        metric_row = dict(base)
        metric_row.update(
            {
                "threshold_type": "top_fraction",
                "threshold_value": fraction,
                "threshold_label": f"top_{int(round(fraction * 100))}pct",
                **threshold_metrics(binary, mask),
            }
        )
        output_rows.append({key: format_float(value) for key, value in metric_row.items()})

    return output_rows, notes


def build_summary(
    source_rows: list[dict[str, str]],
    metric_rows: list[dict[str, object]],
    note_counter: Counter[str],
) -> dict[str, object]:
    image_method_pairs = {
        (row["image_id"], row["model_name"], row["saliency_method"])
        for row in source_rows
    }
    rows_by_model_method = Counter(
        f"{row['model_name']}:{row['saliency_method']}" for row in source_rows
    )
    rows_by_threshold_type = Counter(str(row["threshold_type"]) for row in metric_rows)
    return {
        "run_id": "overlap_metrics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_method_pairs": len(image_method_pairs),
        "source_rows": len(source_rows),
        "metric_rows": len(metric_rows),
        "rows_by_model_method": dict(sorted(rows_by_model_method.items())),
        "rows_by_threshold_type": dict(sorted(rows_by_threshold_type.items())),
        "quality_note_counts": dict(sorted(note_counter.items())),
        "fixed_thresholds": FIXED_THRESHOLDS,
        "top_fractions": TOP_FRACTIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--include-all-methods", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = selected_rows(read_csv(args.input), include_all=args.include_all_methods)
    rows.sort(key=lambda row: (row["model_name"], row["image_id"], row["saliency_method"]))
    if args.limit:
        rows = rows[: args.limit]

    output_rows: list[dict[str, object]] = []
    note_counter: Counter[str] = Counter()
    for row in rows:
        metric_rows, notes = metric_rows_for_analysis(row)
        output_rows.extend(metric_rows)
        if notes:
            note_counter.update(notes)
        else:
            note_counter.update(["ok"])

    write_csv(args.output, output_rows, BASE_FIELDS)
    summary = build_summary(rows, output_rows, note_counter)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
