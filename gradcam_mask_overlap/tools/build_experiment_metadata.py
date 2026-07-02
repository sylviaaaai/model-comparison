"""Build manifest tables for the Grad-CAM/lesion-mask overlap experiment.

This script only creates metadata tables. It does not generate ROI-space masks
or overlap metrics; those later steps should append/update tables using the
same stable image_id keys created here.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_ROOT.parent
REPO_ROOT = WORKSPACE_ROOT / "model-comparison-repo"
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DATA_DIR = EXPERIMENT_ROOT / "data"
TEST_SPLIT = REPO_ROOT / "data_splits" / "final_80_10_10" / "final_test_images.csv"

MODEL_SUMMARIES = {
    "resnet50": {
        "architecture": "CNN",
        "saliency_method": "gradcam",
        "summary_path": REPO_ROOT / "models" / "resnet50" / "final_results" / "gradcam" / "summary.csv",
    },
    "convnext_base": {
        "architecture": "Modern CNN",
        "saliency_method": "gradcam",
        "summary_path": REPO_ROOT / "models" / "convnext_base" / "final_results" / "gradcam" / "summary.csv",
    },
    "swin_base": {
        "architecture": "Hierarchical Transformer",
        "saliency_method": "gradcam",
        "summary_path": REPO_ROOT / "models" / "swin_base" / "final_results" / "gradcam" / "summary.csv",
    },
    "vit_b16": {
        "architecture": "Transformer",
        "saliency_method": "attention_rollout",
        "summary_path": REPO_ROOT
        / "models"
        / "vit_b16"
        / "final_results"
        / "attention_rollout"
        / "summary.csv",
    },
}

FULL_PREDICTION_FILES = {
    "resnet50": {
        "architecture": "CNN",
        "path": WORKSPACE_ROOT / "results" / "patient_level_bootstrap_ci" / "resnet50_test_predictions.csv",
    },
    "convnext_base": {
        "architecture": "Modern CNN",
        "path": WORKSPACE_ROOT / "results" / "patient_level_bootstrap_ci" / "convnext_base_test_predictions.csv",
    },
    "swin_base": {
        "architecture": "Hierarchical Transformer",
        "path": WORKSPACE_ROOT / "results" / "patient_level_bootstrap_ci" / "swin_base_test_predictions.csv",
    },
    "vit_b16": {
        "architecture": "Transformer",
        "path": WORKSPACE_ROOT / "results" / "patient_level_bootstrap_ci" / "vit_b16_test_predictions.csv",
    },
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


def normalize_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("E:/"):
        return "/mnt/e/" + normalized[3:]
    return normalized


def image_id_from_path(path: str) -> str:
    return Path(normalize_path(path)).stem


def class_dir_name(class_name: str) -> str:
    return "Benign" if class_name.lower() == "benign" else "Cancer"


def split_roi_filename(image_id: str) -> tuple[str, str]:
    case_id, original_stem = image_id.split("_", 1)
    return case_id, original_stem + ".jpg"


def derive_original_paths(row: dict[str, str]) -> dict[str, str]:
    image_id = image_id_from_path(row["path"])
    case_id, original_name = split_roi_filename(image_id)
    class_name = row.get("class_name") or row.get("true_class") or row.get("label_name")
    class_dir = class_dir_name(class_name)
    case_dir = Path("/mnt/e/DDSM") / class_dir / case_id
    original_image_path = case_dir / original_name
    original_mask_path = case_dir / original_name.replace(".jpg", "_Mask.jpg")
    original_mask2_path = case_dir / original_name.replace(".jpg", "_MASK2.jpg")
    return {
        "original_image_path": str(original_image_path),
        "original_mask_path": str(original_mask_path),
        "original_mask2_path": str(original_mask2_path),
        "has_original_image": original_image_path.exists(),
        "has_original_mask": original_mask_path.exists(),
        "has_original_mask2": original_mask2_path.exists(),
    }


def parse_view_laterality(image_id: str) -> tuple[str, str]:
    # Example: 0217_C_0217_1.RIGHT_CC
    last = image_id.rsplit(".", 1)[-1]
    laterality, view = last.split("_", 1)
    return laterality, view


def prediction_group(true_label: str, predicted_label: str) -> str:
    true_int = int(float(true_label))
    pred_int = int(float(predicted_label))
    if true_int == 1 and pred_int == 1:
        return "TP"
    if true_int == 0 and pred_int == 0:
        return "TN"
    if true_int == 0 and pred_int == 1:
        return "FP"
    return "FN"


def build_test_images_manifest(test_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in test_rows:
        image_id = image_id_from_path(row["path"])
        laterality, view = parse_view_laterality(image_id)
        paths = derive_original_paths(row)
        rows.append(
            {
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "class_name": row["class_name"],
                "label": row["label"],
                "view": view,
                "laterality": laterality,
                "roi_image_path": normalize_path(row["path"]),
                "original_image_path": paths["original_image_path"],
                "has_original_image": paths["has_original_image"],
                "split": "test",
            }
        )
    return rows


def build_mask_manifest(test_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in test_rows:
        image_id = image_id_from_path(row["path"])
        paths = derive_original_paths(row)
        class_name = row["class_name"]
        class_slug = class_name.lower()
        roi_mask_path = DATA_DIR / "test_roi_masks" / class_slug / f"{image_id}_mask.png"
        qc_overlay_path = DATA_DIR / "qc_overlays" / class_slug / f"{image_id}_qc.png"
        mask_id = f"{image_id}__mask_primary"
        num_source_masks = int(bool(paths["has_original_mask"])) + int(bool(paths["has_original_mask2"]))
        rows.append(
            {
                "mask_id": mask_id,
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "class_name": class_name,
                "label": row["label"],
                "roi_image_path": normalize_path(row["path"]),
                "original_image_path": paths["original_image_path"],
                "original_mask_path": paths["original_mask_path"],
                "original_mask2_path": paths["original_mask2_path"] if paths["has_original_mask2"] else "",
                "roi_mask_path": str(roi_mask_path),
                "qc_overlay_path": str(qc_overlay_path),
                "has_mask": paths["has_original_mask"],
                "has_mask2": paths["has_original_mask2"],
                "num_source_masks": num_source_masks,
                "mask_threshold": 127,
                "lesion_bbox_x": "",
                "lesion_bbox_y": "",
                "lesion_bbox_w": "",
                "lesion_bbox_h": "",
                "crop_x1": "",
                "crop_y1": "",
                "crop_x2": "",
                "crop_y2": "",
                "crop_width": "",
                "crop_height": "",
                "padding": 0.1,
                "crop_rule": "boundingRect(_Mask.jpg) -> square crop around bbox center -> clip to image bounds -> resize to 224",
                "image_resize_interpolation": "INTER_LINEAR",
                "mask_resize_interpolation": "INTER_NEAREST",
                "mask_area_original_px": "",
                "mask_area_roi_px": "",
                "mask_area_roi_fraction": "",
                "multi_mask_rule": "primary analysis uses _Mask.jpg only to match original ROI generation",
                "mask_fill_method": "pending_generation",
                "qc_status": "pending_generation",
                "qc_note": "",
            }
        )
    return rows


def build_prediction_manifest() -> list[dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []

    for model_name, meta in FULL_PREDICTION_FILES.items():
        prediction_path = meta["path"]
        if not prediction_path.exists():
            continue
        for row in read_csv(prediction_path):
            image_id = image_id_from_path(row["path"])
            true_label = row["y_true"]
            predicted_label = row["y_pred"]
            true_class = row["class_name"]
            predicted_class = "Cancer" if int(float(predicted_label)) == 1 else "Benign"
            compact_group = prediction_group(true_label, predicted_label)
            group_name = {
                "TP": "correct_cancer",
                "TN": "correct_benign",
                "FP": "false_positive",
                "FN": "false_negative",
            }[compact_group]
            prediction_id = f"{model_name}__{image_id}"

            prediction_rows.append(
                {
                    "prediction_id": prediction_id,
                    "image_id": image_id,
                    "model_name": model_name,
                    "architecture": meta["architecture"],
                    "model_display_name": row["model"],
                    "fine_tuning_setting": row["setting"],
                    "true_label": true_label,
                    "true_class": true_class,
                    "predicted_label": predicted_label,
                    "predicted_class": predicted_class,
                    "cancer_probability": row["y_score"],
                    "prediction_group": compact_group,
                    "prediction_group_name": group_name,
                    "threshold": 0.5,
                    "split": "test",
                    "source_predictions_csv": str(prediction_path),
                    "source_coverage": "full_test_predictions_for_patient_level_bootstrap_ci",
                }
            )

    return prediction_rows


def build_saliency_manifest() -> list[dict[str, object]]:
    saliency_rows: list[dict[str, object]] = []

    for model_name, meta in MODEL_SUMMARIES.items():
        summary_path = meta["summary_path"]
        if not summary_path.exists():
            continue
        for row in read_csv(summary_path):
            image_id = image_id_from_path(row["path"])
            predicted_label = row.get("predicted_label") or row.get("predicted")
            prediction_id = f"{model_name}__{image_id}"
            saliency_id = f"{model_name}__{image_id}__{meta['saliency_method']}__predclass"
            saliency_rows.append(
                {
                    "saliency_id": saliency_id,
                    "image_id": image_id,
                    "prediction_id": prediction_id,
                    "model_name": model_name,
                    "architecture": meta["architecture"],
                    "saliency_method": meta["saliency_method"],
                    "target_class_for_cam": "predicted_label",
                    "target_class_value": predicted_label,
                    "saliency_path": "",
                    "saliency_width": "",
                    "saliency_height": "",
                    "saliency_normalization": "pending_raw_map_export",
                    "source_summary_csv": str(summary_path),
                    "status": "visualization_exists_raw_map_pending",
                }
            )

    return saliency_rows


def write_run_metadata() -> None:
    run = {
        "run_id": "metadata_bootstrap_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_type": "metadata_bootstrap",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_split": str(TEST_SPLIT),
        "roi_size": 224,
        "mask_threshold": 127,
        "padding": 0.1,
        "crop_rule": "Reproduce original ROI script: threshold _Mask.jpg, boundingRect, square crop with padding=0.1, resize to 224.",
        "primary_mask_rule": "Use _Mask.jpg only to match the original ROI generation script.",
        "multi_mask_rule": "_MASK2 is recorded but ignored in the primary analysis.",
        "image_resize_interpolation": "INTER_LINEAR",
        "mask_resize_interpolation": "INTER_NEAREST",
        "notes": "Prediction manifest uses the full patient-level bootstrap CI prediction files. Saliency manifest uses existing Grad-CAM/attention summary CSVs and may be a subset until raw maps are fully generated.",
    }
    run_path = METADATA_DIR / "mask_generation_run.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    runs_jsonl = METADATA_DIR / "runs.jsonl"
    with runs_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = read_csv(TEST_SPLIT)

    test_images = build_test_images_manifest(test_rows)
    masks = build_mask_manifest(test_rows)
    predictions = build_prediction_manifest()
    saliency = build_saliency_manifest()

    write_csv(
        METADATA_DIR / "test_images_manifest.csv",
        test_images,
        [
            "image_id",
            "patient_id",
            "class_name",
            "label",
            "view",
            "laterality",
            "roi_image_path",
            "original_image_path",
            "has_original_image",
            "split",
        ],
    )
    write_csv(
        METADATA_DIR / "test_roi_mask_manifest.csv",
        masks,
        [
            "mask_id",
            "image_id",
            "patient_id",
            "class_name",
            "label",
            "roi_image_path",
            "original_image_path",
            "original_mask_path",
            "original_mask2_path",
            "roi_mask_path",
            "qc_overlay_path",
            "has_mask",
            "has_mask2",
            "num_source_masks",
            "mask_threshold",
            "lesion_bbox_x",
            "lesion_bbox_y",
            "lesion_bbox_w",
            "lesion_bbox_h",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "crop_width",
            "crop_height",
            "padding",
            "crop_rule",
            "image_resize_interpolation",
            "mask_resize_interpolation",
            "mask_area_original_px",
            "mask_area_roi_px",
            "mask_area_roi_fraction",
            "multi_mask_rule",
            "mask_fill_method",
            "qc_status",
            "qc_note",
        ],
    )
    write_csv(
        METADATA_DIR / "model_predictions.csv",
        predictions,
        [
            "prediction_id",
            "image_id",
            "model_name",
            "architecture",
            "model_display_name",
            "fine_tuning_setting",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "cancer_probability",
            "prediction_group",
            "prediction_group_name",
            "threshold",
            "split",
            "source_predictions_csv",
            "source_coverage",
        ],
    )
    write_csv(
        METADATA_DIR / "saliency_manifest.csv",
        saliency,
        [
            "saliency_id",
            "image_id",
            "prediction_id",
            "model_name",
            "architecture",
            "saliency_method",
            "target_class_for_cam",
            "target_class_value",
            "saliency_path",
            "saliency_width",
            "saliency_height",
            "saliency_normalization",
            "source_summary_csv",
            "status",
        ],
    )
    write_csv(
        METADATA_DIR / "overlap_metrics.csv",
        [],
        [
            "overlap_id",
            "image_id",
            "mask_id",
            "saliency_id",
            "model_name",
            "threshold_type",
            "threshold_value",
            "cam_area_px",
            "mask_area_px",
            "intersection_area_px",
            "union_area_px",
            "iou",
            "dice",
            "cam_energy_in_mask",
            "mask_coverage_by_cam",
            "peak_x",
            "peak_y",
            "peak_value",
            "peak_in_mask",
            "peak_distance_to_mask_px",
            "qc_status",
        ],
    )
    write_run_metadata()

    print(f"Wrote {len(test_images)} test image rows")
    print(f"Wrote {len(masks)} mask manifest rows")
    print(f"Wrote {len(predictions)} prediction rows")
    print(f"Wrote {len(saliency)} saliency rows")


if __name__ == "__main__":
    main()
