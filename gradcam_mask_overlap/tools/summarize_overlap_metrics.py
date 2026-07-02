"""Summarize primary lesion-mask overlap metrics at model level."""

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
DEFAULT_CONTINUOUS = METADATA_DIR / "overlap_continuous_model_summary.csv"
DEFAULT_THRESHOLD = METADATA_DIR / "overlap_threshold_model_summary.csv"
DEFAULT_BEST = METADATA_DIR / "overlap_best_fixed_thresholds.csv"
DEFAULT_SUMMARY = METADATA_DIR / "overlap_model_summary.json"

CONTINUOUS_METRICS = [
    "cam_energy_in_mask",
    "normalized_energy_in_mask",
    "peak_distance_to_mask_px",
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


def finite_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values = [as_float(row[metric]) for row in rows]
    return [value for value in values if value == value]


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.10g}"


def metric_summary(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    values = finite_values(rows, metric)
    if not values:
        return {
            f"{metric}_mean": "nan",
            f"{metric}_median": "nan",
            f"{metric}_sd": "nan",
        }
    return {
        f"{metric}_mean": fmt(mean(values)),
        f"{metric}_median": fmt(median(values)),
        f"{metric}_sd": fmt(stdev(values) if len(values) > 1 else 0.0),
    }


def pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["image_id"], row["model_name"], row["saliency_method"]


def model_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["model_name"],
        row["model_display_name"],
        row["saliency_method"],
        row["explanation_family"],
    )


def build_continuous_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    first_by_pair: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        first_by_pair.setdefault(pair_key(row), row)

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in first_by_pair.values():
        groups[model_key(row)].append(row)

    output_rows: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        model_name, model_display_name, saliency_method, explanation_family = key
        output: dict[str, object] = {
            "model_name": model_name,
            "model_display_name": model_display_name,
            "saliency_method": saliency_method,
            "explanation_family": explanation_family,
            "n_image_method_pairs": len(group_rows),
            "peak_in_mask_rate": fmt(mean(row["peak_in_mask"] == "True" for row in group_rows)),
            "zero_saliency_pairs": sum(row["quality_note"] == "zero_saliency" for row in group_rows),
        }
        for metric in CONTINUOUS_METRICS:
            output.update(metric_summary(group_rows, metric))
        output_rows.append(output)
    return output_rows


def build_threshold_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["model_name"],
            row["model_display_name"],
            row["saliency_method"],
            row["threshold_type"],
            row["threshold_value"],
            row["threshold_label"],
        )
        groups[key].append(row)

    output_rows: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        model_name, model_display_name, saliency_method, threshold_type, threshold_value, threshold_label = key
        output: dict[str, object] = {
            "model_name": model_name,
            "model_display_name": model_display_name,
            "saliency_method": saliency_method,
            "threshold_type": threshold_type,
            "threshold_value": threshold_value,
            "threshold_label": threshold_label,
            "n_image_method_pairs": len(group_rows),
        }
        for metric in THRESHOLD_METRICS:
            output.update(metric_summary(group_rows, metric))
        output_rows.append(output)
    return output_rows


def build_best_fixed_thresholds(threshold_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_model: dict[str, dict[str, object]] = {}
    for row in threshold_rows:
        if row["threshold_type"] != "fixed":
            continue
        model_name = str(row["model_name"])
        mean_iou = as_float(str(row["iou_mean"]))
        current = best_by_model.get(model_name)
        if current is None or mean_iou > as_float(str(current["iou_mean"])):
            best_by_model[model_name] = row
    return [best_by_model[key] for key in sorted(best_by_model)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--continuous-output", type=Path, default=DEFAULT_CONTINUOUS)
    parser.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD)
    parser.add_argument("--best-fixed-output", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_csv(args.metrics)
    continuous_rows = build_continuous_summary(rows)
    threshold_rows = build_threshold_summary(rows)
    best_rows = build_best_fixed_thresholds(threshold_rows)

    continuous_fields = [
        "model_name",
        "model_display_name",
        "saliency_method",
        "explanation_family",
        "n_image_method_pairs",
        "peak_in_mask_rate",
        "zero_saliency_pairs",
    ]
    for metric in CONTINUOUS_METRICS:
        continuous_fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_sd"])

    threshold_fields = [
        "model_name",
        "model_display_name",
        "saliency_method",
        "threshold_type",
        "threshold_value",
        "threshold_label",
        "n_image_method_pairs",
    ]
    for metric in THRESHOLD_METRICS:
        threshold_fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_sd"])

    write_csv(args.continuous_output, continuous_rows, continuous_fields)
    write_csv(args.threshold_output, threshold_rows, threshold_fields)
    write_csv(args.best_fixed_output, best_rows, threshold_fields)

    summary = {
        "run_id": "overlap_model_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_rows": len(rows),
        "continuous_summary_rows": len(continuous_rows),
        "threshold_summary_rows": len(threshold_rows),
        "best_fixed_threshold_rows": len(best_rows),
        "outputs": {
            "continuous": str(args.continuous_output),
            "threshold": str(args.threshold_output),
            "best_fixed": str(args.best_fixed_output),
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
