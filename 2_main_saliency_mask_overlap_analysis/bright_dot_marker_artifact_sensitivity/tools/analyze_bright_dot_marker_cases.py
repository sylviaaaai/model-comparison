"""Analyze bright-dot / marker cases and render top-20% four-model QC panels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from render_overlap_smoke_qc import (
    PRIMARY_METHODS,
    primary_rows,
    read_csv,
    render_case,
    write_csv,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_REVIEW = METADATA_DIR / "manual_mask_review_template.csv"
DEFAULT_MASTER = METADATA_DIR / "test_image_master_table.csv"
DEFAULT_ANALYSIS = METADATA_DIR / "analysis_ready_overlap_manifest.csv"
DEFAULT_IMAGES = METADATA_DIR / "test_images_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "limitation_analysis" / "bright_dot_marker"

MODEL_PREFIXES = {
    "resnet50": "ResNet50",
    "vit_b16": "ViT_B16",
    "convnext_base": "ConvNeXt_Base",
    "swin_base": "Swin_Base",
}

MODEL_ORDER = ["resnet50", "vit_b16", "convnext_base", "swin_base"]
PRED_TYPES = ["TP", "FN", "FP", "TN"]


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    write_csv(path, rows, fieldnames)


def marker_cases(review_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in review_rows
        if row.get("bright_dot_or_marker") and row["bright_dot_or_marker"] != "none"
    ]


def build_case_rows(
    bright_rows: list[dict[str, str]],
    master_by_image: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for review in bright_rows:
        image_id = review["image_id"]
        master = master_by_image[image_id]
        row: dict[str, object] = {
            "image_id": image_id,
            "patient_id": review["patient_id"],
            "patient_num": review["patient_num"],
            "label": review["label"],
            "class_name": review["class_name"],
            "laterality": review["laterality"],
            "view": review["view"],
            "bright_dot_or_marker": review["bright_dot_or_marker"],
            "manual_mask_type": review["manual_mask_type"],
            "mask_qc_type": review["mask_qc_type"],
            "notes": review["notes"],
            "roi_image_path": review["roi_image_path"],
            "roi_mask_path": review["roi_mask_path"],
            "qc_overlay_path": review["qc_overlay_path"],
        }
        for model_name in MODEL_ORDER:
            prefix = MODEL_PREFIXES[model_name]
            row[f"{prefix}_pred_type"] = master[f"{prefix}_pred_type"]
            row[f"{prefix}_pred_label"] = master[f"{prefix}_pred_label"]
            row[f"{prefix}_prob"] = master[f"{prefix}_prob"]
            row[f"{prefix}_iou_top20"] = master[f"{prefix}_iou_top20"]
            row[f"{prefix}_mask_coverage_top20"] = master[f"{prefix}_mask_coverage_top20"]
        rows.append(row)
    return rows


def build_model_rows(
    bright_rows: list[dict[str, str]],
    master_by_image: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for review in bright_rows:
        image_id = review["image_id"]
        master = master_by_image[image_id]
        for model_name in MODEL_ORDER:
            prefix = MODEL_PREFIXES[model_name]
            rows.append(
                {
                    "image_id": image_id,
                    "patient_id": review["patient_id"],
                    "label": review["label"],
                    "bright_dot_or_marker": review["bright_dot_or_marker"],
                    "manual_mask_type": review["manual_mask_type"],
                    "mask_qc_type": review["mask_qc_type"],
                    "model_name": model_name,
                    "model_prefix": prefix,
                    "pred_type": master[f"{prefix}_pred_type"],
                    "pred_label": master[f"{prefix}_pred_label"],
                    "cancer_probability": master[f"{prefix}_prob"],
                    "iou_top20": master[f"{prefix}_iou_top20"],
                    "mask_coverage_top20": master[f"{prefix}_mask_coverage_top20"],
                }
            )
    return rows


def build_summary_rows(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        selected = [row for row in model_rows if row["model_name"] == model_name]
        counter = Counter(str(row["pred_type"]) for row in selected)
        rows.append(
            {
                "scope": "all_non_none_marker_cases",
                "bright_dot_or_marker": "any_non_none",
                "model_name": model_name,
                "n_images": len(selected),
                **{pred_type: counter.get(pred_type, 0) for pred_type in PRED_TYPES},
            }
        )
        for marker in sorted({str(row["bright_dot_or_marker"]) for row in selected}):
            marker_rows = [row for row in selected if row["bright_dot_or_marker"] == marker]
            marker_counter = Counter(str(row["pred_type"]) for row in marker_rows)
            rows.append(
                {
                    "scope": "by_marker_location",
                    "bright_dot_or_marker": marker,
                    "model_name": model_name,
                    "n_images": len(marker_rows),
                    **{pred_type: marker_counter.get(pred_type, 0) for pred_type in PRED_TYPES},
                }
            )
    return rows


def build_simple_confusion_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "model_name": row["model_name"],
            "n_bright_dot_images": row["n_images"],
            "TP": row["TP"],
            "FN": row["FN"],
            "FP": row["FP"],
            "TN": row["TN"],
        }
        for row in summary_rows
        if row["scope"] == "all_non_none_marker_cases"
    ]


def build_figure_index_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "image_id": row["image_id"],
            "label": row["label"],
            "bright_dot_or_marker": row["bright_dot_or_marker"],
            "ResNet50_pred_type": row["ResNet50_pred_type"],
            "ViT_B16_pred_type": row["ViT_B16_pred_type"],
            "ConvNeXt_Base_pred_type": row["ConvNeXt_Base_pred_type"],
            "Swin_Base_pred_type": row["Swin_Base_pred_type"],
            "top20_figure_path": row["top20_figure_path"],
        }
        for row in case_rows
    ]


def render_top20_figures(
    bright_rows: list[dict[str, str]],
    master_by_image: dict[str, dict[str, str]],
    analysis_manifest: Path,
    image_manifest: Path,
    output_dir: Path,
) -> dict[str, str]:
    analysis_rows = primary_rows(read_csv(analysis_manifest))
    image_rows = {row["image_id"]: row for row in read_csv(image_manifest)}
    by_image_model = {
        (row["image_id"], row["model_name"]): row
        for row in analysis_rows
    }
    figure_paths: dict[str, str] = {}
    for review in bright_rows:
        image_id = review["image_id"]
        master = master_by_image[image_id]
        rows_by_model = {
            model_name: by_image_model[(image_id, model_name)]
            for model_name in PRIMARY_METHODS
        }
        marker = review["bright_dot_or_marker"]
        title_group = (
            f"marker={marker} | "
            f"R={master['ResNet50_pred_type']} "
            f"V={master['ViT_B16_pred_type']} "
            f"C={master['ConvNeXt_Base_pred_type']} "
            f"S={master['Swin_Base_pred_type']}"
        )
        output_path = output_dir / marker / f"{image_id}_top20_four_model_qc.png"
        render_case(
            image_id=image_id,
            group=title_group,
            rows_by_model=rows_by_model,
            roi_image_path=image_rows[image_id]["roi_image_path"],
            output_path=output_path,
            top_fraction=0.20,
        )
        figure_paths[image_id] = str(output_path)
    return figure_paths


def write_report(
    path: Path,
    case_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    marker_counts = Counter(str(row["bright_dot_or_marker"]) for row in case_rows)
    lines = [
        "# Bright-Dot / Marker Limitation Analysis",
        "",
        "This folder contains image-level cases where `bright_dot_or_marker != none`.",
        "",
        "## Files",
        "",
        "- `bright_dot_cases_confusion_wide.csv`: one row per marker case, with four-model prediction groups and top-20% metrics.",
        "- `bright_dot_cases_by_model_long.csv`: one row per image-model pair.",
        "- `bright_dot_confusion_summary_by_model.csv`: TP/FN/FP/TN counts by model and marker location.",
        "- `bright_dot_confusion_counts_simple.csv`: simple TP/FN/FP/TN counts by model for all marker cases.",
        "- `bright_dot_image_figure_index.csv`: one row per marker case with the rendered figure path.",
        "- `figures_top20/`: one four-model top-20% QC panel per marker case.",
        "",
        "## Marker Counts",
        "",
    ]
    for marker, count in sorted(marker_counts.items()):
        lines.append(f"- `{marker}`: {count}")
    lines += [
        "",
        "## Overall Confusion Counts",
        "",
        "| Model | N | TP | FN | FP | TN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["scope"] != "all_non_none_marker_cases":
            continue
        lines.append(
            f"| `{row['model_name']}` | {row['n_images']} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} |"
        )
    lines += [
        "",
        "## Interpretation Note",
        "",
        "These cases are collected for limitation and sensitivity review. The marker location does not by itself prove model shortcut learning; it flags images where a small bright dot or marker-like artifact could complicate visual interpretation of saliency/overlap.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-table", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--master-table", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--image-manifest", type=Path, default=METADATA_DIR / "test_images_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    review_rows = read_csv(args.review_table)
    master_rows = read_csv(args.master_table)
    master_by_image = {row["image_id"]: row for row in master_rows}
    bright_rows = marker_cases(review_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = render_top20_figures(
        bright_rows,
        master_by_image,
        args.analysis_manifest,
        args.image_manifest,
        args.output_dir / "figures_top20",
    )

    case_rows = build_case_rows(bright_rows, master_by_image)
    for row in case_rows:
        row["top20_figure_path"] = figure_paths[str(row["image_id"])]
    model_rows = build_model_rows(bright_rows, master_by_image)
    summary_rows = build_summary_rows(model_rows)
    simple_rows = build_simple_confusion_rows(summary_rows)

    base_fields = [
        "image_id",
        "patient_id",
        "patient_num",
        "label",
        "class_name",
        "laterality",
        "view",
        "bright_dot_or_marker",
        "manual_mask_type",
        "mask_qc_type",
        "notes",
        "roi_image_path",
        "roi_mask_path",
        "qc_overlay_path",
        "top20_figure_path",
    ]
    model_fields = []
    for model_name in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_name]
        model_fields.extend(
            [
                f"{prefix}_pred_type",
                f"{prefix}_pred_label",
                f"{prefix}_prob",
                f"{prefix}_iou_top20",
                f"{prefix}_mask_coverage_top20",
            ]
        )
    save_csv(args.output_dir / "bright_dot_cases_confusion_wide.csv", case_rows, base_fields + model_fields)
    save_csv(
        args.output_dir / "bright_dot_cases_by_model_long.csv",
        model_rows,
        [
            "image_id",
            "patient_id",
            "label",
            "bright_dot_or_marker",
            "manual_mask_type",
            "mask_qc_type",
            "model_name",
            "model_prefix",
            "pred_type",
            "pred_label",
            "cancer_probability",
            "iou_top20",
            "mask_coverage_top20",
        ],
    )
    save_csv(
        args.output_dir / "bright_dot_confusion_summary_by_model.csv",
        summary_rows,
        ["scope", "bright_dot_or_marker", "model_name", "n_images", *PRED_TYPES],
    )
    save_csv(
        args.output_dir / "bright_dot_confusion_counts_simple.csv",
        simple_rows,
        ["model_name", "n_bright_dot_images", *PRED_TYPES],
    )
    save_csv(
        args.output_dir / "bright_dot_image_figure_index.csv",
        build_figure_index_rows(case_rows),
        [
            "image_id",
            "label",
            "bright_dot_or_marker",
            "ResNet50_pred_type",
            "ViT_B16_pred_type",
            "ConvNeXt_Base_pred_type",
            "Swin_Base_pred_type",
            "top20_figure_path",
        ],
    )
    write_report(args.output_dir / "README.md", case_rows, summary_rows)
    print(f"Wrote bright-dot limitation analysis for {len(case_rows)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
