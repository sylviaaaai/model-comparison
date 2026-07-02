"""Update analysis-ready overlap manifest from available saliency rows.

This preserves existing manual annotation and ROI-mask columns, then expands
the selected model to one row per available saliency map in saliency_manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_ANALYSIS = METADATA_DIR / "analysis_ready_overlap_manifest.csv"
DEFAULT_SALIENCY = METADATA_DIR / "saliency_manifest.csv"
DEFAULT_SUMMARY = METADATA_DIR / "analysis_ready_overlap_manifest_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--saliency-manifest", type=Path, default=DEFAULT_SALIENCY)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


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


def ready_for_overlap(row: dict[str, str]) -> bool:
    return bool(row.get("saliency_path")) and row.get("status") == "raw_map_available"


def make_analysis_row(base: dict[str, str], saliency: dict[str, str]) -> dict[str, object]:
    method = saliency["saliency_method"]
    row = dict(base)
    row.update(
        {
            "analysis_id": f"{base['prediction_id']}__{method}",
            "saliency_id": saliency["saliency_id"],
            "saliency_method": method,
            "saliency_path": saliency["saliency_path"],
            "saliency_status": saliency["status"],
            "ready_for_overlap": str(ready_for_overlap(saliency)),
        }
    )
    return row


def build_summary(
    rows: list[dict[str, object]],
    saliency_rows: list[dict[str, str]],
) -> dict[str, object]:
    ready_rows = [row for row in rows if row.get("ready_for_overlap") == "True"]
    ready_by_model_method = Counter(
        f"{row['model_name']}:{row['saliency_method']}" for row in ready_rows
    )
    rows_by_model = Counter(str(row["model_name"]) for row in rows)
    saliency_status_counts = Counter(str(row.get("saliency_status", "")) for row in rows)
    model_prediction_ids = {str(row["prediction_id"]) for row in rows}
    image_ids = {str(row["image_id"]) for row in rows}
    return {
        "test_images": len(image_ids),
        "manual_label_images": len(image_ids),
        "roi_masks": len(image_ids),
        "model_prediction_rows": len(model_prediction_ids),
        "saliency_manifest_rows": len(saliency_rows),
        "analysis_manifest_rows": len(rows),
        "ready_for_overlap_rows": len(ready_rows),
        "ready_by_model_method": dict(sorted(ready_by_model_method.items())),
        "rows_by_model": dict(sorted(rows_by_model.items())),
        "saliency_status_counts": dict(sorted(saliency_status_counts.items())),
    }


def main() -> None:
    args = parse_args()
    analysis_rows, analysis_fields = read_csv(args.analysis_manifest)
    saliency_rows, _ = read_csv(args.saliency_manifest)

    base_by_prediction: dict[str, dict[str, str]] = {}
    for row in analysis_rows:
        if row["model_name"] == args.model_name and row["prediction_id"] not in base_by_prediction:
            base_by_prediction[row["prediction_id"]] = row

    selected_saliency = [
        row
        for row in saliency_rows
        if row["model_name"] == args.model_name and row.get("status") == "raw_map_available"
    ]
    selected_saliency.sort(key=lambda row: (row["prediction_id"], row["saliency_method"], row["saliency_id"]))

    updated_model_rows = []
    for saliency in selected_saliency:
        base = base_by_prediction.get(saliency["prediction_id"])
        if base is None:
            continue
        updated_model_rows.append(make_analysis_row(base, saliency))

    kept_rows = [row for row in analysis_rows if row["model_name"] != args.model_name]
    output_rows = kept_rows + updated_model_rows
    write_csv(args.analysis_manifest, output_rows, analysis_fields)

    summary = build_summary(output_rows, saliency_rows)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Kept rows: {len(kept_rows)}")
    print(f"Updated {args.model_name} rows: {len(updated_model_rows)}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
