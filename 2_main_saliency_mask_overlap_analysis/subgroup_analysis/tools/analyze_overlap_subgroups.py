"""Summarize overlap metrics by true label and model confusion group."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = EXPERIMENT_ROOT / "metadata" / "test_image_master_table.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "overlap_analysis" / "subgroup_analyses"

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]
SCOPES = {
    "main_yes": {"yes"},
    "sensitivity_yes_cautious": {"yes", "cautious"},
}
METRICS = [
    "energy_inside_mask",
    "normalized_energy",
    "peak_in_mask",
    "peak_distance_px",
    "mask_coverage_top20",
    "iou_top20",
    "dice_top20",
]
TRUE_LABEL_ORDER = {"malignant": 0, "benign": 1}
CONFUSION_GROUP_ORDER = {"TP": 0, "FN": 1, "FP": 2, "TN": 3}


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


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def subset_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        scope: [row for row in rows if row.get("usable_for_overlap", "") in allowed]
        for scope, allowed in SCOPES.items()
    }


def is_marker_positive(row: dict[str, str]) -> bool:
    marker = row.get("bright_dot_or_marker", "").strip()
    return bool(marker) and marker != "none"


def summarize_group(
    rows: list[dict[str, str]],
    scope: str,
    model: str,
    metric: str,
    extra: dict[str, object],
) -> dict[str, object]:
    values = [as_float(row.get(f"{model}_{metric}", "")) for row in rows]
    clean = [value for value in values if value is not None]
    pred_counter = Counter(row.get(f"{model}_pred_type", "") for row in rows)
    return {
        "scope": scope,
        **extra,
        "model": model,
        "metric": metric,
        "n_images": len(rows),
        "n_valid": len(clean),
        "n_malignant": sum(row["true_label"] == "malignant" for row in rows),
        "n_benign": sum(row["true_label"] == "benign" for row in rows),
        "TP": pred_counter["TP"],
        "FN": pred_counter["FN"],
        "FP": pred_counter["FP"],
        "TN": pred_counter["TN"],
        "marker_positive": sum(is_marker_positive(row) for row in rows),
        "zero_saliency_any_model": sum(row.get("zero_saliency_any_model", "") == "True" for row in rows),
        "mean": fmt_float(mean(clean) if clean else None),
        "median": fmt_float(median(clean) if clean else None),
        "sd": fmt_float(sample_sd(clean)),
        "min": fmt_float(min(clean) if clean else None),
        "max": fmt_float(max(clean) if clean else None),
    }


def build_true_label_summary(subsets: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        labels = sorted({row["true_label"] for row in rows}, key=lambda item: TRUE_LABEL_ORDER.get(item, 99))
        for true_label in labels:
            group_rows = [row for row in rows if row["true_label"] == true_label]
            for model in MODELS:
                for metric in METRICS:
                    out.append(
                        summarize_group(
                            group_rows,
                            scope,
                            model,
                            metric,
                            {"true_label": true_label},
                        )
                    )
    return out


def build_confusion_group_summary(subsets: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for model in MODELS:
            groups = sorted(
                {row[f"{model}_pred_type"] for row in rows},
                key=lambda item: CONFUSION_GROUP_ORDER.get(item, 99),
            )
            for confusion_group in groups:
                group_rows = [row for row in rows if row[f"{model}_pred_type"] == confusion_group]
                for metric in METRICS:
                    out.append(
                        summarize_group(
                            group_rows,
                            scope,
                            model,
                            metric,
                            {"confusion_group": confusion_group},
                        )
                    )
    return out


def build_true_label_confusion_group_summary(
    subsets: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        labels = sorted({row["true_label"] for row in rows}, key=lambda item: TRUE_LABEL_ORDER.get(item, 99))
        for model in MODELS:
            groups = sorted(
                {row[f"{model}_pred_type"] for row in rows},
                key=lambda item: CONFUSION_GROUP_ORDER.get(item, 99),
            )
            for true_label in labels:
                for confusion_group in groups:
                    group_rows = [
                        row
                        for row in rows
                        if row["true_label"] == true_label
                        and row[f"{model}_pred_type"] == confusion_group
                    ]
                    if not group_rows:
                        continue
                    for metric in METRICS:
                        out.append(
                            summarize_group(
                                group_rows,
                                scope,
                                model,
                                metric,
                                {
                                    "true_label": true_label,
                                    "confusion_group": confusion_group,
                                },
                            )
                        )
    return out


def build_counts(
    true_label_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    true_label_confusion_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    def add_from(row: dict[str, object], stratifier: str, fields: list[str]) -> None:
        key = (stratifier, *(row.get(field, "") for field in fields), row["model"])
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "stratifier": stratifier,
                "scope": row["scope"],
                "model": row["model"],
                "true_label": row.get("true_label", ""),
                "confusion_group": row.get("confusion_group", ""),
                "n_images": row["n_images"],
                "n_malignant": row["n_malignant"],
                "n_benign": row["n_benign"],
                "TP": row["TP"],
                "FN": row["FN"],
                "FP": row["FP"],
                "TN": row["TN"],
                "marker_positive": row["marker_positive"],
                "zero_saliency_any_model": row["zero_saliency_any_model"],
            }
        )

    for row in true_label_rows:
        add_from(row, "true_label", ["scope", "true_label"])
    for row in confusion_rows:
        add_from(row, "confusion_group", ["scope", "confusion_group"])
    for row in true_label_confusion_rows:
        add_from(row, "true_label_and_confusion_group", ["scope", "true_label", "confusion_group"])
    return out


def filter_rows(
    rows: list[dict[str, object]],
    scope: str,
    metric: str,
    **criteria: str,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if row["scope"] != scope or row["metric"] != metric:
            continue
        if all(row.get(key, "") == value for key, value in criteria.items()):
            out.append(row)
    return out


def write_markdown_summary(
    path: Path,
    true_label_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    counts_rows: list[dict[str, object]],
) -> None:
    def value(rows: list[dict[str, object]], model: str, field: str = "mean") -> str:
        for row in rows:
            if row["model"] == model:
                return str(row[field])
        return ""

    lines = [
        "# Overlap Subgroup Analysis",
        "",
        "This is an exploratory subgroup analysis. It explains where the main overlap signal is coming from; it does not replace the primary `main_yes` and `yes_cautious` model comparisons.",
        "",
        "## Output Tables",
        "",
        "- `overlap_by_true_label_summary.csv`: scope x true_label x model x metric.",
        "- `overlap_by_confusion_group_summary.csv`: scope x model-specific TP/FN/FP/TN x metric.",
        "- `overlap_by_true_label_and_confusion_group_summary.csv`: scope x true_label x model-specific confusion group x metric.",
        "- `overlap_subgroup_counts.csv`: denominators for the subgroup tables.",
        "",
        "## True-Label Subgroups",
        "",
        "Mean top-20% Dice by true label:",
        "",
        "| Scope | True label | ResNet50 | ViT_B16 | ConvNeXt_Base | Swin_Base |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in ["main_yes", "sensitivity_yes_cautious"]:
        for true_label in ["malignant", "benign"]:
            rows = filter_rows(true_label_rows, scope, "dice_top20", true_label=true_label)
            lines.append(
                f"| `{scope}` | `{true_label}` | "
                f"{value(rows, 'ResNet50')} | {value(rows, 'ViT_B16')} | "
                f"{value(rows, 'ConvNeXt_Base')} | {value(rows, 'Swin_Base')} |"
            )

    lines += [
        "",
        "## Model-Specific Confusion Groups",
        "",
        "Mean top-20% Dice by model-specific confusion group:",
        "",
        "| Scope | Model | TP | FN | FP | TN |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in ["main_yes", "sensitivity_yes_cautious"]:
        for model in MODELS:
            cells = []
            for group in ["TP", "FN", "FP", "TN"]:
                rows = filter_rows(confusion_rows, scope, "dice_top20", confusion_group=group)
                cells.append(value(rows, model))
            lines.append(f"| `{scope}` | `{model}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- `true_label` subgrouping asks whether overlap behavior differs between malignant and benign images.",
        "- `confusion_group` subgrouping is model-specific: each model has its own TP/FN/FP/TN labels.",
        "- TP/FN are malignant-image groups; FP/TN are benign-image groups.",
        "- Small FP/FN subgroup denominators should be interpreted cautiously.",
        "- ResNet50 zero-saliency rows can reduce valid `peak_distance_px` counts in TN/FN-heavy subgroups.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = read_csv(args.master)
    subsets = subset_rows(rows)
    true_label_rows = build_true_label_summary(subsets)
    confusion_rows = build_confusion_group_summary(subsets)
    true_label_confusion_rows = build_true_label_confusion_group_summary(subsets)
    counts_rows = build_counts(true_label_rows, confusion_rows, true_label_confusion_rows)

    write_csv(
        args.output_dir / "overlap_by_true_label_summary.csv",
        true_label_rows,
        list(true_label_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "overlap_by_confusion_group_summary.csv",
        confusion_rows,
        list(confusion_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "overlap_by_true_label_and_confusion_group_summary.csv",
        true_label_confusion_rows,
        list(true_label_confusion_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "overlap_subgroup_counts.csv",
        counts_rows,
        list(counts_rows[0].keys()),
    )
    write_markdown_summary(
        args.output_dir / "overlap_subgroup_analysis_summary.md",
        true_label_rows,
        confusion_rows,
        counts_rows,
    )

    summary = {
        "run_id": "overlap_subgroup_analysis_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_master": str(args.master),
        "n_source_rows": len(rows),
        "scopes": {scope: len(scope_rows) for scope, scope_rows in subsets.items()},
        "outputs": {
            "true_label_summary": str(args.output_dir / "overlap_by_true_label_summary.csv"),
            "confusion_group_summary": str(args.output_dir / "overlap_by_confusion_group_summary.csv"),
            "true_label_confusion_group_summary": str(
                args.output_dir / "overlap_by_true_label_and_confusion_group_summary.csv"
            ),
            "counts": str(args.output_dir / "overlap_subgroup_counts.csv"),
            "markdown": str(args.output_dir / "overlap_subgroup_analysis_summary.md"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlap_subgroup_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
