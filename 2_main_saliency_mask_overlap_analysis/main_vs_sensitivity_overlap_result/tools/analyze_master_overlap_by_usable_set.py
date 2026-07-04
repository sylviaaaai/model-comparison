"""Summarize overlap metrics from the image-level master table by usable set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = EXPERIMENT_ROOT / "metadata" / "test_image_master_table.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "overlap_analysis" / "master_table_usable_sets"

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]
METRICS = [
    "energy_inside_mask",
    "normalized_energy",
    "peak_in_mask",
    "peak_distance_px",
    "mask_coverage_top20",
    "iou_top20",
    "dice_top20",
]
LOWER_IS_BETTER = {"peak_distance_px"}
PEAK_DISTANCE_INVALID_REASON = "Peak distance was computed only for valid non-zero saliency maps."


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


def as_float(value: str) -> float | None:
    value = str(value).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    if lowered in {"nan", "none", "null"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def subset_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        "main_yes": [row for row in rows if row["usable_for_overlap"] == "yes"],
        "sensitivity_yes_cautious": [
            row for row in rows if row["usable_for_overlap"] in {"yes", "cautious"}
        ],
        "excluded_no": [row for row in rows if row["usable_for_overlap"] == "no"],
        "all_images": rows,
    }


def summarize_counts(subsets: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        usable_counts = Counter(row["usable_for_overlap"] for row in rows)
        marker_positive = sum(
            bool(row.get("bright_dot_or_marker", "").strip())
            and row.get("bright_dot_or_marker", "").strip() != "none"
            for row in rows
        )
        out.append(
            {
                "scope": scope,
                "n_images": len(rows),
                "n_malignant": sum(row["true_label"] == "malignant" for row in rows),
                "n_benign": sum(row["true_label"] == "benign" for row in rows),
                "usable_yes": usable_counts["yes"],
                "usable_cautious": usable_counts["cautious"],
                "usable_no": usable_counts["no"],
                "marker_positive": marker_positive,
                "marker_negative": len(rows) - marker_positive,
            }
        )
    return out


def summarize_metrics(subsets: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for model in MODELS:
            for metric in METRICS:
                col = f"{model}_{metric}"
                values = [as_float(row.get(col, "")) for row in rows]
                clean = [value for value in values if value is not None]
                out.append(
                    {
                        "scope": scope,
                        "model": model,
                        "metric": metric,
                        "n_images": len(rows),
                        "n_valid": len(clean),
                        "mean": fmt_float(mean(clean)),
                        "median": fmt_float(median(clean) if clean else None),
                        "sd": fmt_float(sample_sd(clean)),
                        "min": fmt_float(min(clean) if clean else None),
                        "max": fmt_float(max(clean) if clean else None),
                    }
                )
    return out


def rank_models(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in metric_rows:
        grouped.setdefault((str(row["scope"]), str(row["metric"])), []).append(row)

    out: list[dict[str, object]] = []
    for (scope, metric), rows in sorted(grouped.items()):
        scored = []
        for row in rows:
            value = as_float(str(row["mean"]))
            if value is not None:
                scored.append((value, str(row["model"]), row))
        reverse = metric not in LOWER_IS_BETTER
        for rank, (_value, _model, row) in enumerate(
            sorted(scored, key=lambda item: item[0], reverse=reverse), start=1
        ):
            out.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "rank": rank,
                    "model": row["model"],
                    "mean": row["mean"],
                    "median": row["median"],
                    "n_valid": row["n_valid"],
                    "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                }
            )
    return out


def summarize_peak_distance_validity(
    subsets: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for model in MODELS:
            col = f"{model}_peak_distance_px"
            n_valid = sum(as_float(row.get(col, "")) is not None for row in rows)
            n_invalid = len(rows) - n_valid
            out.append(
                {
                    "model": model,
                    "scope": scope,
                    "n_images": len(rows),
                    "n_valid_peak_distance": n_valid,
                    "n_invalid": n_invalid,
                    "reason": PEAK_DISTANCE_INVALID_REASON if n_invalid else "",
                }
            )
    return out


def write_markdown_summary(
    path: Path,
    count_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    peak_distance_validity_rows: list[dict[str, object]],
) -> None:
    count_by_scope = {row["scope"]: row for row in count_rows}
    rank_by_key = {(row["scope"], row["metric"], row["rank"]): row for row in rank_rows}
    validity_by_key = {
        (row["scope"], row["model"]): row for row in peak_distance_validity_rows
    }

    lines = [
        "# Master-Table Overlap Analysis by Usable Set",
        "",
        "Source table: `metadata/test_image_master_table.csv`.",
        "",
        "Scopes:",
        "",
        "- `main_yes`: `usable_for_overlap == yes`.",
        "- `sensitivity_yes_cautious`: `usable_for_overlap in {yes, cautious}`.",
        "- `excluded_no`: `usable_for_overlap == no`.",
        "- `all_images`: full 290-image table for context only.",
        "",
        "## Scope Counts",
        "",
        "| Scope | N | Malignant | Benign | Yes | Cautious | No | Marker-positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in ["main_yes", "sensitivity_yes_cautious", "excluded_no", "all_images"]:
        row = count_by_scope[scope]
        lines.append(
            f"| `{scope}` | {row['n_images']} | {row['n_malignant']} | {row['n_benign']} | "
            f"{row['usable_yes']} | {row['usable_cautious']} | {row['usable_no']} | {row['marker_positive']} |"
        )

    lines += [
        "",
        "## Main Set Model Ranking",
        "",
        "Ranking is by mean value; higher is better except `peak_distance_px`, where lower is better.",
        "",
        "| Metric | Rank 1 | Rank 2 | Rank 3 | Rank 4 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        cells = []
        for rank in range(1, 5):
            row = rank_by_key.get(("main_yes", metric, rank))
            cells.append("" if row is None else f"`{row['model']}` ({row['mean']})")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Sensitivity Set Model Ranking",
        "",
        "| Metric | Rank 1 | Rank 2 | Rank 3 | Rank 4 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        cells = []
        for rank in range(1, 5):
            row = rank_by_key.get(("sensitivity_yes_cautious", metric, rank))
            cells.append("" if row is None else f"`{row['model']}` ({row['mean']})")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Peak Distance Validity",
        "",
        "`peak_distance_px` is summarized only for rows with a valid saliency peak.",
        "",
        "| Scope | Model | N images | Valid peak distance | Invalid | Reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for scope in ["main_yes", "sensitivity_yes_cautious", "excluded_no", "all_images"]:
        for model in MODELS:
            row = validity_by_key[(scope, model)]
            lines.append(
                f"| `{scope}` | `{model}` | {row['n_images']} | "
                f"{row['n_valid_peak_distance']} | {row['n_invalid']} | "
                f"{row['reason']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = read_csv(args.master)
    subsets = subset_rows(rows)
    count_rows = summarize_counts(subsets)
    metric_rows = summarize_metrics(subsets)
    rank_rows = rank_models(metric_rows)
    peak_distance_validity_rows = summarize_peak_distance_validity(subsets)

    write_csv(
        args.output_dir / "overlap_usable_scope_counts.csv",
        count_rows,
        list(count_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "overlap_usable_scope_model_metric_summary.csv",
        metric_rows,
        list(metric_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "overlap_usable_scope_model_metric_ranks.csv",
        rank_rows,
        list(rank_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "overlap_peak_distance_validity_by_scope.csv",
        peak_distance_validity_rows,
        list(peak_distance_validity_rows[0].keys()),
    )
    write_markdown_summary(
        args.output_dir / "overlap_usable_scope_summary.md",
        count_rows,
        rank_rows,
        peak_distance_validity_rows,
    )

    summary = {
        "run_id": "master_overlap_usable_sets_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_master": str(args.master),
        "n_source_rows": len(rows),
        "outputs": {
            "counts": str(args.output_dir / "overlap_usable_scope_counts.csv"),
            "metric_summary": str(args.output_dir / "overlap_usable_scope_model_metric_summary.csv"),
            "metric_ranks": str(args.output_dir / "overlap_usable_scope_model_metric_ranks.csv"),
            "peak_distance_validity": str(args.output_dir / "overlap_peak_distance_validity_by_scope.csv"),
            "markdown": str(args.output_dir / "overlap_usable_scope_summary.md"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlap_usable_scope_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
