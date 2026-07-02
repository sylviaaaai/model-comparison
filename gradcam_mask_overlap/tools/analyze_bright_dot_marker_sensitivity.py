"""Summarize bright-dot / marker sensitivity on the image-level master table."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = EXPERIMENT_ROOT / "metadata" / "test_image_master_table.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "limitation_analysis" / "bright_dot_marker"
USABLE_SCOPES = {
    "all": None,
    "main_yes": {"yes"},
    "yes_cautious": {"yes", "cautious"},
}

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]
PRED_TYPES = ["TP", "FN", "FP", "TN"]
OVERLAP_METRICS = [
    "energy_inside_mask",
    "normalized_energy",
    "peak_in_mask",
    "peak_distance_px",
    "mask_coverage_top20",
    "iou_top20",
    "dice_top20",
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


def is_marker_positive(row: dict[str, str]) -> bool:
    marker = row.get("bright_dot_or_marker", "").strip()
    return bool(marker) and marker != "none"


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


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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
    marker_positive = [row for row in rows if is_marker_positive(row)]
    marker_negative = [row for row in rows if not is_marker_positive(row)]
    return {
        "all_images": rows,
        "marker_negative": marker_negative,
        "marker_positive": marker_positive,
        "marker_positive_benign": [row for row in marker_positive if row["true_label"] == "benign"],
        "marker_positive_malignant": [row for row in marker_positive if row["true_label"] == "malignant"],
    }


def filter_by_usable_scope(rows: list[dict[str, str]], usable_scope: str) -> list[dict[str, str]]:
    allowed = USABLE_SCOPES[usable_scope]
    if allowed is None:
        return rows
    return [row for row in rows if row.get("usable_for_overlap", "") in allowed]


def summarize_classification(
    subsets: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for model in MODELS:
            pred_col = f"{model}_pred_type"
            counter = Counter(row[pred_col] for row in rows)
            tp = counter["TP"]
            fn = counter["FN"]
            fp = counter["FP"]
            tn = counter["TN"]
            n = len(rows)
            benign_n = sum(1 for row in rows if row["true_label"] == "benign")
            malignant_n = sum(1 for row in rows if row["true_label"] == "malignant")
            out.append(
                {
                    "scope": scope,
                    "model": model,
                    "n_images": n,
                    "n_benign": benign_n,
                    "n_malignant": malignant_n,
                    "TP": tp,
                    "FN": fn,
                    "FP": fp,
                    "TN": tn,
                    "accuracy": fmt_float(rate(tp + tn, n)),
                    "sensitivity": fmt_float(rate(tp, tp + fn)),
                    "specificity": fmt_float(rate(tn, tn + fp)),
                    "benign_fp_rate": fmt_float(rate(fp, fp + tn)),
                    "malignant_fn_rate": fmt_float(rate(fn, tp + fn)),
                }
            )
    return out


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(score, label) for label, score in zip(labels, scores, strict=True)]
    n_pos = sum(label == 1 for _score, label in pairs)
    n_neg = sum(label == 0 for _score, label in pairs)
    if n_pos == 0 or n_neg == 0:
        return None

    sorted_pairs = sorted(pairs, key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        for _score, label in sorted_pairs[i:j]:
            if label == 1:
                rank_sum_pos += avg_rank
        rank += j - i
        i = j

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarize_classification_metrics(
    subsets: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for model in MODELS:
            pred_col = f"{model}_pred_type"
            prob_col = f"{model}_prob"
            counter = Counter(row[pred_col] for row in rows)
            tp = counter["TP"]
            fn = counter["FN"]
            fp = counter["FP"]
            tn = counter["TN"]
            n = len(rows)
            labels: list[int] = []
            scores: list[float] = []
            for row in rows:
                score = as_float(row.get(prob_col, ""))
                if score is None:
                    continue
                labels.append(1 if row["true_label"] == "malignant" else 0)
                scores.append(score)
            precision = rate(tp, tp + fp)
            recall = rate(tp, tp + fn)
            f1 = None
            if precision is not None and recall is not None and precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            out.append(
                {
                    "scope": scope,
                    "model": model,
                    "n_images": n,
                    "n_benign": sum(1 for row in rows if row["true_label"] == "benign"),
                    "n_malignant": sum(1 for row in rows if row["true_label"] == "malignant"),
                    "n_auc_valid": len(scores),
                    "AUC": fmt_float(roc_auc(labels, scores)),
                    "accuracy": fmt_float(rate(tp + tn, n)),
                    "sensitivity_recall": fmt_float(recall),
                    "specificity": fmt_float(rate(tn, tn + fp)),
                    "precision": fmt_float(precision),
                    "F1": fmt_float(f1),
                    "TP": tp,
                    "FN": fn,
                    "FP": fp,
                    "TN": tn,
                }
            )
    return out


def summarize_overlap(
    subsets: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for model in MODELS:
            for metric in OVERLAP_METRICS:
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


def keyed(rows: list[dict[str, object]], key_fields: list[str]) -> dict[tuple[object, ...], dict[str, object]]:
    return {tuple(row[field] for field in key_fields): row for row in rows}


def diff_numeric(after: dict[str, object], before: dict[str, object], field: str) -> str:
    after_value = as_float(str(after.get(field, "")))
    before_value = as_float(str(before.get(field, "")))
    if after_value is None or before_value is None:
        return ""
    return fmt_float(after_value - before_value)


def build_exclusion_delta(
    classification_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    class_by_key = keyed(classification_rows, ["scope", "model"])
    overlap_by_key = keyed(overlap_rows, ["scope", "model", "metric"])
    out: list[dict[str, object]] = []

    for model in MODELS:
        all_class = class_by_key[("all_images", model)]
        marker_negative_class = class_by_key[("marker_negative", model)]
        marker_positive_class = class_by_key[("marker_positive", model)]
        out.append(
            {
                "comparison": "marker_negative_minus_all_images",
                "model": model,
                "metric_family": "classification",
                "metric": "benign_fp_rate",
                "all_images": all_class["benign_fp_rate"],
                "marker_negative": marker_negative_class["benign_fp_rate"],
                "marker_positive": marker_positive_class["benign_fp_rate"],
                "delta_after_excluding_marker": diff_numeric(
                    marker_negative_class, all_class, "benign_fp_rate"
                ),
                "marker_positive_minus_marker_negative": diff_numeric(
                    marker_positive_class, marker_negative_class, "benign_fp_rate"
                ),
            }
        )
        out.append(
            {
                "comparison": "marker_negative_minus_all_images",
                "model": model,
                "metric_family": "classification",
                "metric": "specificity",
                "all_images": all_class["specificity"],
                "marker_negative": marker_negative_class["specificity"],
                "marker_positive": marker_positive_class["specificity"],
                "delta_after_excluding_marker": diff_numeric(
                    marker_negative_class, all_class, "specificity"
                ),
                "marker_positive_minus_marker_negative": diff_numeric(
                    marker_positive_class, marker_negative_class, "specificity"
                ),
            }
        )
        for metric in OVERLAP_METRICS:
            all_overlap = overlap_by_key[("all_images", model, metric)]
            marker_negative_overlap = overlap_by_key[("marker_negative", model, metric)]
            marker_positive_overlap = overlap_by_key[("marker_positive", model, metric)]
            out.append(
                {
                    "comparison": "marker_negative_minus_all_images",
                    "model": model,
                    "metric_family": "overlap",
                    "metric": metric,
                    "all_images": all_overlap["mean"],
                    "marker_negative": marker_negative_overlap["mean"],
                    "marker_positive": marker_positive_overlap["mean"],
                    "delta_after_excluding_marker": diff_numeric(
                        marker_negative_overlap, all_overlap, "mean"
                    ),
                    "marker_positive_minus_marker_negative": diff_numeric(
                        marker_positive_overlap, marker_negative_overlap, "mean"
                    ),
                }
            )
    return out


def build_classification_metric_delta(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = keyed(metric_rows, ["scope", "model"])
    metrics = ["AUC", "accuracy", "sensitivity_recall", "specificity", "precision", "F1"]
    out: list[dict[str, object]] = []
    for model in MODELS:
        all_row = by_key[("all_images", model)]
        marker_negative_row = by_key[("marker_negative", model)]
        marker_positive_row = by_key[("marker_positive", model)]
        for metric in metrics:
            out.append(
                {
                    "comparison": "marker_negative_minus_all_images",
                    "model": model,
                    "metric": metric,
                    "all_images": all_row[metric],
                    "marker_negative": marker_negative_row[metric],
                    "marker_positive": marker_positive_row[metric],
                    "delta_after_excluding_marker": diff_numeric(
                        marker_negative_row, all_row, metric
                    ),
                    "marker_positive_minus_marker_negative": diff_numeric(
                        marker_positive_row, marker_negative_row, metric
                    ),
                }
            )
    return out


def write_markdown_summary(
    path: Path,
    classification_rows: list[dict[str, object]],
    classification_metric_rows: list[dict[str, object]],
    classification_metric_delta_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    usable_scope: str,
    source_rows: int,
    analyzed_rows: int,
) -> None:
    class_by_key = keyed(classification_rows, ["scope", "model"])
    delta_by_key = {
        (row["model"], row["metric"]): row
        for row in delta_rows
        if row["comparison"] == "marker_negative_minus_all_images"
    }
    metric_by_key = keyed(classification_metric_rows, ["scope", "model"])
    metric_delta_by_key = keyed(classification_metric_delta_rows, ["model", "metric"])

    lines = [
        "# Bright-Dot / Marker Sensitivity Summary",
        "",
        "This analysis compares marker-positive and marker-negative images using the current image-level master table.",
        "",
        "Important scope note:",
        "",
        "```text",
        "This is the bright-dot / marker artifact track. It does not use",
        "bright_dot_or_marker to decide mask usability.",
        f"usable_scope = {usable_scope}",
        f"source_rows = {source_rows}",
        f"analyzed_rows = {analyzed_rows}",
        "```",
        "",
        "## Files",
        "",
        "- `bright_dot_marker_classification_sensitivity_by_model.csv`",
        "- `bright_dot_marker_classification_performance_by_model.csv`",
        "- `bright_dot_marker_classification_performance_delta_by_model.csv`",
        "- `bright_dot_marker_overlap_sensitivity_by_model.csv`",
        "- `bright_dot_marker_exclusion_delta_by_model.csv`",
        "",
        "## Classification Summary",
        "",
        "| Model | Scope | N | TP | FN | FP | TN | Benign FP rate | Specificity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for scope in ["all_images", "marker_negative", "marker_positive"]:
            row = class_by_key[(scope, model)]
            lines.append(
                f"| `{model}` | `{scope}` | {row['n_images']} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} | {row['benign_fp_rate']} | {row['specificity']} |"
            )

    lines += [
        "",
        "## Classification Performance",
        "",
        "| Model | Scope | AUC | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for scope in ["all_images", "marker_negative", "marker_positive"]:
            row = metric_by_key[(scope, model)]
            lines.append(
                f"| `{model}` | `{scope}` | {row['AUC']} | {row['accuracy']} | {row['sensitivity_recall']} | {row['specificity']} | {row['precision']} | {row['F1']} |"
            )

    lines += [
        "",
        "## Classification Performance Delta",
        "",
        "Delta is computed as `marker_negative - all_images`.",
        "",
        "| Model | AUC delta | Accuracy delta | Sensitivity delta | Specificity delta | Precision delta | F1 delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        lines.append(
            f"| `{model}` | "
            f"{metric_delta_by_key[(model, 'AUC')]['delta_after_excluding_marker']} | "
            f"{metric_delta_by_key[(model, 'accuracy')]['delta_after_excluding_marker']} | "
            f"{metric_delta_by_key[(model, 'sensitivity_recall')]['delta_after_excluding_marker']} | "
            f"{metric_delta_by_key[(model, 'specificity')]['delta_after_excluding_marker']} | "
            f"{metric_delta_by_key[(model, 'precision')]['delta_after_excluding_marker']} | "
            f"{metric_delta_by_key[(model, 'F1')]['delta_after_excluding_marker']} |"
        )

    lines += [
        "",
        "## Marker Exclusion Delta",
        "",
        "Delta is computed as `marker_negative - all_images`, so a negative FP-rate delta means FP rate decreases after marker-positive images are excluded.",
        "",
        "| Model | FP-rate delta | Specificity delta | Mean iou_top20 delta | Mean dice_top20 delta | Mean normalized_energy delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        fp = delta_by_key[(model, "benign_fp_rate")]
        spec = delta_by_key[(model, "specificity")]
        iou = delta_by_key[(model, "iou_top20")]
        dice = delta_by_key[(model, "dice_top20")]
        energy = delta_by_key[(model, "normalized_energy")]
        lines.append(
            f"| `{model}` | {fp['delta_after_excluding_marker']} | {spec['delta_after_excluding_marker']} | {iou['delta_after_excluding_marker']} | {dice['delta_after_excluding_marker']} | {energy['delta_after_excluding_marker']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The marker-positive subset is small and imbalanced, so the analysis should be treated as a sensitivity / limitation check rather than a definitive causal test.",
        "",
        "ConvNeXt_Base remains the strongest artifact-sensitivity concern because it combines frequent marker attention with a high benign marker false-positive rate.",
        "",
        "ViT_B16 also frequently attends to markers, but the marker-positive benign false-positive count is lower than ConvNeXt_Base.",
        "",
        "ResNet50 and Swin_Base show lower overall marker-attention rates. However, marker attention is still present in their marker-positive false-positive cases, so the issue is relevant across architectures.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--usable-scope",
        choices=sorted(USABLE_SCOPES),
        default="all",
        help="Filter rows by usable_for_overlap before marker sensitivity analysis.",
    )
    args = parser.parse_args()

    source_rows = read_csv(args.master)
    rows = filter_by_usable_scope(source_rows, args.usable_scope)
    subsets = subset_rows(rows)
    classification_rows = summarize_classification(subsets)
    classification_metric_rows = summarize_classification_metrics(subsets)
    classification_metric_delta_rows = build_classification_metric_delta(classification_metric_rows)
    overlap_rows = summarize_overlap(subsets)
    delta_rows = build_exclusion_delta(classification_rows, overlap_rows)

    write_csv(
        args.output_dir / "bright_dot_marker_classification_sensitivity_by_model.csv",
        classification_rows,
        list(classification_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "bright_dot_marker_classification_performance_by_model.csv",
        classification_metric_rows,
        list(classification_metric_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "bright_dot_marker_classification_performance_delta_by_model.csv",
        classification_metric_delta_rows,
        list(classification_metric_delta_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "bright_dot_marker_overlap_sensitivity_by_model.csv",
        overlap_rows,
        list(overlap_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "bright_dot_marker_exclusion_delta_by_model.csv",
        delta_rows,
        list(delta_rows[0].keys()),
    )
    write_markdown_summary(
        args.output_dir / "bright_dot_marker_sensitivity_summary.md",
        classification_rows,
        classification_metric_rows,
        classification_metric_delta_rows,
        overlap_rows,
        delta_rows,
        args.usable_scope,
        len(source_rows),
        len(rows),
    )

    print(f"Read {len(source_rows)} rows from {args.master}")
    print(f"Analyzed {len(rows)} rows for usable_scope={args.usable_scope}")
    print(f"Marker-positive rows: {len(subsets['marker_positive'])}")
    print(f"Marker-negative rows: {len(subsets['marker_negative'])}")
    print(f"Wrote sensitivity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
