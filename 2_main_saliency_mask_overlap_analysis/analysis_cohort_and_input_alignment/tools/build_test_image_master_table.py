"""Build one-row-per-test-image master table for overlap analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"

DEFAULT_IMAGES = METADATA_DIR / "test_images_manifest.csv"
DEFAULT_MASKS = METADATA_DIR / "test_roi_mask_manifest.csv"
DEFAULT_PREDICTIONS = METADATA_DIR / "model_predictions.csv"
DEFAULT_ANALYSIS = METADATA_DIR / "analysis_ready_overlap_manifest.csv"
DEFAULT_METRICS = METADATA_DIR / "overlap_selected_metrics_wide.csv"
DEFAULT_MANUAL_REVIEW = METADATA_DIR / "manual_mask_review_template.csv"
DEFAULT_MARKER_ATTENTION = (
    EXPERIMENT_ROOT / "limitation_analysis" / "bright_dot_marker" / "bright_dot_attention_review.csv"
)
DEFAULT_OUTPUT = METADATA_DIR / "test_image_master_table.csv"

PRIMARY_METHODS = {
    "resnet50": ("ResNet50", "gradcam_malignant"),
    "vit_b16": ("ViT_B16", "attention_rollout"),
    "convnext_base": ("ConvNeXt_Base", "gradcam_malignant"),
    "swin_base": ("Swin_Base", "gradcam_malignant"),
}

MODEL_PREFIXES = [prefix for prefix, _method in PRIMARY_METHODS.values()]


BASE_COLUMNS = [
    "image_id",
    "patient_id",
    "image_path",
    "class_name",
    "true_label",
    "roi_mask_path",
    "mask_area_fraction",
    "manual_mask_type",
    "mask_qc_type",
    "bright_dot_or_marker",
    "usable_for_overlap",
    "manual_review_notes",
    "mask_qc_status",
    "mask_qc_note",
]

MODEL_COLUMN_SUFFIXES = [
    "prob",
    "pred_label",
    "pred_type",
    "heatmap_path",
    "energy_inside_mask",
    "normalized_energy",
    "peak_in_mask",
    "peak_distance_px",
    "mask_coverage_top20",
    "iou_top20",
    "dice_top20",
]

MARKER_ATTENTION_COLUMNS = [f"{prefix}_attends_marker" for prefix in MODEL_PREFIXES]

STATUS_COLUMNS = [
    "has_mask",
    "has_prediction_all_models",
    "has_heatmap_all_models",
    "quality_note_any_model",
    "zero_saliency_any_model",
    "included_in_main_overlap_analysis",
    "missing_reason",
]

FIELDNAMES = (
    BASE_COLUMNS
    + [f"{prefix}_{suffix}" for suffix in MODEL_COLUMN_SUFFIXES for prefix in MODEL_PREFIXES]
    + MARKER_ATTENTION_COLUMNS
    + STATUS_COLUMNS
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def true_label_name(value: str) -> str:
    return "malignant" if str(value) == "1" else "benign"


def pred_label_name(value: str) -> str:
    return "malignant" if str(value) == "1" else "benign"


def is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def index_primary_analysis(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        method = PRIMARY_METHODS.get(row["model_name"], ("", ""))[1]
        if row.get("ready_for_overlap") == "True" and row.get("saliency_method") == method:
            indexed[(row["image_id"], row["model_name"])] = row
    return indexed


def collect_missing_reasons(
    has_mask: bool,
    has_prediction_all_models: bool,
    has_heatmap_all_models: bool,
    usable_for_overlap: str,
) -> str:
    reasons: list[str] = []
    if not has_mask:
        reasons.append("missing_or_empty_mask")
    if not has_prediction_all_models:
        reasons.append("missing_prediction")
    if not has_heatmap_all_models:
        reasons.append("missing_heatmap")
    if usable_for_overlap == "no":
        reasons.append("not_usable_for_overlap")
    elif usable_for_overlap == "cautious":
        reasons.append("cautious_sensitivity_only")
    elif usable_for_overlap not in {"yes", "cautious"}:
        reasons.append("manual_usable_for_overlap_pending")
    return ";".join(reasons)


def build_master_table(args: argparse.Namespace) -> list[dict[str, object]]:
    images = read_csv(args.image_manifest)
    masks = {row["image_id"]: row for row in read_csv(args.mask_manifest)}
    predictions = {
        (row["image_id"], row["model_name"]): row
        for row in read_csv(args.predictions)
        if row["model_name"] in PRIMARY_METHODS
    }
    analysis = index_primary_analysis(read_csv(args.analysis_manifest))
    metrics = {
        (row["image_id"], row["model_name"]): row
        for row in read_csv(args.selected_metrics)
        if row["model_name"] in PRIMARY_METHODS
    }
    manual_review = {row["image_id"]: row for row in read_csv(args.manual_review)}
    marker_attention = {row["image_id"]: row for row in read_csv(args.marker_attention)}

    output_rows: list[dict[str, object]] = []
    for image in sorted(images, key=lambda row: row["image_id"]):
        image_id = image["image_id"]
        mask = masks.get(image_id, {})
        review = manual_review.get(image_id, {})

        has_mask = is_true(mask.get("has_mask", "")) and not is_true(mask.get("mask_empty", ""))
        usable_for_overlap = review.get("usable_for_overlap", "")

        row: dict[str, object] = {
            "image_id": image_id,
            "patient_id": image["patient_id"],
            "image_path": image["roi_image_path"],
            "class_name": image["class_name"],
            "true_label": true_label_name(image["label"]),
            "roi_mask_path": mask.get("roi_mask_path", ""),
            "mask_area_fraction": mask.get("mask_area_roi_fraction", ""),
            "manual_mask_type": review.get("manual_mask_type", ""),
            "mask_qc_type": review.get("mask_qc_type", ""),
            "bright_dot_or_marker": review.get("bright_dot_or_marker", ""),
            "usable_for_overlap": usable_for_overlap,
            "manual_review_notes": review.get("notes", ""),
            "mask_qc_status": mask.get("qc_status", ""),
            "mask_qc_note": mask.get("qc_note", ""),
            "has_mask": has_mask,
        }

        quality_notes: list[str] = []
        for model_name, (prefix, _method) in PRIMARY_METHODS.items():
            prediction = predictions.get((image_id, model_name), {})
            analysis_row = analysis.get((image_id, model_name), {})
            metric = metrics.get((image_id, model_name), {})

            row[f"{prefix}_prob"] = prediction.get("cancer_probability", "")
            row[f"{prefix}_pred_label"] = pred_label_name(prediction["predicted_label"]) if prediction else ""
            row[f"{prefix}_pred_type"] = prediction.get("prediction_group", "")
            row[f"{prefix}_heatmap_path"] = analysis_row.get("saliency_path", "")
            row[f"{prefix}_energy_inside_mask"] = metric.get("cam_energy_in_mask", "")
            row[f"{prefix}_normalized_energy"] = metric.get("normalized_energy_in_mask", "")
            row[f"{prefix}_peak_in_mask"] = metric.get("peak_in_mask", "")
            row[f"{prefix}_peak_distance_px"] = metric.get("peak_distance_to_mask_px", "")
            row[f"{prefix}_mask_coverage_top20"] = metric.get("top_20pct_mask_coverage_by_cam", "")
            row[f"{prefix}_iou_top20"] = metric.get("top_20pct_iou", "")
            row[f"{prefix}_dice_top20"] = metric.get("top_20pct_dice", "")

            if metric.get("quality_note"):
                quality_notes.append(f"{prefix}:{metric['quality_note']}")

        marker_row = marker_attention.get(image_id, {})
        for prefix in MODEL_PREFIXES:
            row[f"{prefix}_attends_marker"] = marker_row.get(f"{prefix}_attends_marker", "")

        has_prediction_all = all((image_id, model_name) in predictions for model_name in PRIMARY_METHODS)
        has_heatmap_all = all(bool(analysis.get((image_id, model_name), {}).get("saliency_path")) for model_name in PRIMARY_METHODS)
        missing_reason = collect_missing_reasons(has_mask, has_prediction_all, has_heatmap_all, usable_for_overlap)

        row["has_prediction_all_models"] = has_prediction_all
        row["has_heatmap_all_models"] = has_heatmap_all
        row["quality_note_any_model"] = ";".join(quality_notes)
        row["zero_saliency_any_model"] = any(note.endswith(":zero_saliency") for note in quality_notes)
        row["included_in_main_overlap_analysis"] = not missing_reason
        row["missing_reason"] = missing_reason
        output_rows.append(row)

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--mask-manifest", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--selected-metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_MANUAL_REVIEW)
    parser.add_argument("--marker-attention", type=Path, default=DEFAULT_MARKER_ATTENTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_master_table(args)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} image-level rows to {args.output}")


if __name__ == "__main__":
    main()
