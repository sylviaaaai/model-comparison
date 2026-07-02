"""Build compact overlap tables for primary reporting and plotting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_METRICS = METADATA_DIR / "overlap_metrics.csv"
DEFAULT_WIDE = METADATA_DIR / "overlap_selected_metrics_wide.csv"
DEFAULT_MODEL = METADATA_DIR / "overlap_selected_model_summary.csv"
DEFAULT_SUMMARY = METADATA_DIR / "overlap_selected_summary.json"

SELECTED_THRESHOLDS = [
    ("fixed", "fixed_0.1"),
    ("top_fraction", "top_10pct"),
    ("top_fraction", "top_20pct"),
]

ID_FIELDS = [
    "analysis_id",
    "image_id",
    "prediction_id",
    "model_name",
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
    "quality_status",
    "quality_note",
]

CONTINUOUS_FIELDS = [
    "cam_energy_in_mask",
    "normalized_energy_in_mask",
    "peak_in_mask",
    "peak_distance_to_mask_px",
    "mask_area_px",
    "mask_area_fraction",
    "cam_sum",
]

THRESHOLD_METRICS = [
    "iou",
    "dice",
    "mask_coverage_by_cam",
    "saliency_precision",
    "cam_area_fraction",
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


def as_float(value: str) -> float:
    if value in {"", "nan", "NaN"}:
        return float("nan")
    return float(value)


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.10g}"


def pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["image_id"], row["model_name"], row["saliency_method"]


def selected_key(row: dict[str, str]) -> str | None:
    key = (row["threshold_type"], row["threshold_label"])
    if key not in SELECTED_THRESHOLDS:
        return None
    return row["threshold_label"]


def build_wide_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_pair: dict[tuple[str, str, str], dict[str, str]] = {}
    selected_by_pair: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)

    for row in rows:
        key = pair_key(row)
        by_pair.setdefault(key, row)
        label = selected_key(row)
        if label:
            selected_by_pair[key][label] = row

    output_rows: list[dict[str, object]] = []
    for key in sorted(by_pair):
        first = by_pair[key]
        output: dict[str, object] = {field: first.get(field, "") for field in ID_FIELDS}
        output.update({field: first.get(field, "") for field in CONTINUOUS_FIELDS})
        for _, label in SELECTED_THRESHOLDS:
            selected = selected_by_pair[key].get(label, {})
            for metric in THRESHOLD_METRICS:
                output[f"{label}_{metric}"] = selected.get(metric, "")
        output_rows.append(output)
    return output_rows


def metric_summary(rows: list[dict[str, object]], field: str) -> dict[str, str]:
    values = [as_float(str(row[field])) for row in rows if str(row.get(field, "")) != ""]
    values = [value for value in values if value == value]
    if not values:
        return {f"{field}_mean": "nan", f"{field}_median": "nan", f"{field}_sd": "nan"}
    return {
        f"{field}_mean": fmt(mean(values)),
        f"{field}_median": fmt(median(values)),
        f"{field}_sd": fmt(stdev(values) if len(values) > 1 else 0.0),
    }


def build_model_summary(wide_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in wide_rows:
        key = (
            str(row["model_name"]),
            str(row["model_display_name"]),
            str(row["saliency_method"]),
            str(row["explanation_family"]),
        )
        groups[key].append(row)

    summary_fields = [
        "cam_energy_in_mask",
        "normalized_energy_in_mask",
        "peak_distance_to_mask_px",
        "mask_area_fraction",
    ]
    for _, label in SELECTED_THRESHOLDS:
        for metric in THRESHOLD_METRICS:
            summary_fields.append(f"{label}_{metric}")

    output_rows: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        model_name, model_display_name, saliency_method, explanation_family = key
        output: dict[str, object] = {
            "model_name": model_name,
            "model_display_name": model_display_name,
            "saliency_method": saliency_method,
            "explanation_family": explanation_family,
            "n_image_method_pairs": len(group_rows),
            "peak_in_mask_rate": fmt(mean(str(row["peak_in_mask"]) == "True" for row in group_rows)),
            "zero_saliency_pairs": sum(str(row["quality_note"]) == "zero_saliency" for row in group_rows),
        }
        for field in summary_fields:
            output.update(metric_summary(group_rows, field))
        output_rows.append(output)
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--wide-output", type=Path, default=DEFAULT_WIDE)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_csv(args.metrics)
    wide_rows = build_wide_rows(rows)
    model_rows = build_model_summary(wide_rows)

    wide_fields = ID_FIELDS + CONTINUOUS_FIELDS
    for _, label in SELECTED_THRESHOLDS:
        wide_fields.extend(f"{label}_{metric}" for metric in THRESHOLD_METRICS)

    model_fields = [
        "model_name",
        "model_display_name",
        "saliency_method",
        "explanation_family",
        "n_image_method_pairs",
        "peak_in_mask_rate",
        "zero_saliency_pairs",
    ]
    for field in [
        "cam_energy_in_mask",
        "normalized_energy_in_mask",
        "peak_distance_to_mask_px",
        "mask_area_fraction",
        *[
            f"{label}_{metric}"
            for _, label in SELECTED_THRESHOLDS
            for metric in THRESHOLD_METRICS
        ],
    ]:
        model_fields.extend([f"{field}_mean", f"{field}_median", f"{field}_sd"])

    write_csv(args.wide_output, wide_rows, wide_fields)
    write_csv(args.model_output, model_rows, model_fields)

    summary = {
        "run_id": "overlap_selected_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metric_rows": len(rows),
        "wide_rows": len(wide_rows),
        "model_summary_rows": len(model_rows),
        "selected_thresholds": [label for _, label in SELECTED_THRESHOLDS],
        "outputs": {
            "wide": str(args.wide_output),
            "model": str(args.model_output),
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
