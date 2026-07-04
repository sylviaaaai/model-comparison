"""Paired model comparisons for master-table overlap metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = EXPERIMENT_ROOT / "metadata" / "test_image_master_table.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "overlap_analysis" / "paired_model_comparisons"

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]
REFERENCE_MODEL = "ConvNeXt_Base"
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
LOWER_IS_BETTER = {"peak_distance_px"}


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


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float],
    iterations: int,
    rng: random.Random,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1 or iterations <= 0:
        return values[0], values[0]
    n = len(values)
    boot_means = []
    for _ in range(iterations):
        boot_means.append(sum(values[rng.randrange(n)] for _i in range(n)) / n)
    boot_means.sort()
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def subset_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        scope: [row for row in rows if row.get("usable_for_overlap", "") in allowed]
        for scope, allowed in SCOPES.items()
    }


def metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"


def effect_delta(metric: str, model_a_value: float, model_b_value: float) -> float:
    raw_delta = model_a_value - model_b_value
    if metric in LOWER_IS_BETTER:
        return -raw_delta
    return raw_delta


def build_long_rows(
    subsets: dict[str, list[dict[str, str]]],
    reference_model: str,
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    comparators = [model for model in MODELS if model != reference_model]
    for scope, rows in subsets.items():
        for row in rows:
            for metric in METRICS:
                ref_value = as_float(row.get(f"{reference_model}_{metric}", ""))
                for comparator in comparators:
                    comp_value = as_float(row.get(f"{comparator}_{metric}", ""))
                    valid_pair = ref_value is not None and comp_value is not None
                    raw_delta = None
                    signed_delta = None
                    if valid_pair:
                        raw_delta = ref_value - comp_value
                        signed_delta = effect_delta(metric, ref_value, comp_value)
                    rows_out.append(
                        {
                            "scope": scope,
                            "image_id": row["image_id"],
                            "metric": metric,
                            "direction": metric_direction(metric),
                            "reference_model": reference_model,
                            "comparator_model": comparator,
                            "reference_value": fmt_float(ref_value),
                            "comparator_value": fmt_float(comp_value),
                            "raw_delta_reference_minus_comparator": fmt_float(raw_delta),
                            "effect_delta_positive_favors_reference": fmt_float(signed_delta),
                            "valid_pair": valid_pair,
                        }
                    )
    return rows_out


def build_pair_rows(
    subsets: dict[str, list[dict[str, str]]],
    model_a: str,
    model_b: str,
    metric: str,
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        for row in rows:
            a_value = as_float(row.get(f"{model_a}_{metric}", ""))
            b_value = as_float(row.get(f"{model_b}_{metric}", ""))
            if a_value is None or b_value is None:
                continue
            raw_delta = a_value - b_value
            signed_delta = effect_delta(metric, a_value, b_value)
            rows_out.append(
                {
                    "scope": scope,
                    "image_id": row["image_id"],
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_value": a_value,
                    "model_b_value": b_value,
                    "raw_delta_model_a_minus_model_b": raw_delta,
                    "effect_delta_positive_favors_model_a": signed_delta,
                }
            )
    return rows_out


def summarize_pair(
    scope: str,
    metric: str,
    model_a: str,
    model_b: str,
    n_scope_images: int,
    pair_rows: list[dict[str, object]],
    bootstrap_iterations: int,
    rng: random.Random,
) -> dict[str, object]:
    n_pairs = len(pair_rows)
    a_values = [float(row["model_a_value"]) for row in pair_rows]
    b_values = [float(row["model_b_value"]) for row in pair_rows]
    raw_deltas = [float(row["raw_delta_model_a_minus_model_b"]) for row in pair_rows]
    effect_deltas = [float(row["effect_delta_positive_favors_model_a"]) for row in pair_rows]
    ci_low, ci_high = bootstrap_mean_ci(effect_deltas, bootstrap_iterations, rng)
    model_a_better = sum(delta > 0 for delta in effect_deltas)
    model_b_better = sum(delta < 0 for delta in effect_deltas)
    ties = sum(delta == 0 for delta in effect_deltas)
    return {
        "scope": scope,
        "metric": metric,
        "direction": metric_direction(metric),
        "model_a": model_a,
        "model_b": model_b,
        "n_scope_images": n_scope_images,
        "n_pairs": n_pairs,
        "n_invalid_pairs": n_scope_images - n_pairs,
        "model_a_mean": fmt_float(mean(a_values) if a_values else None),
        "model_b_mean": fmt_float(mean(b_values) if b_values else None),
        "raw_mean_delta_model_a_minus_model_b": fmt_float(mean(raw_deltas) if raw_deltas else None),
        "effect_mean_delta_positive_favors_model_a": fmt_float(
            mean(effect_deltas) if effect_deltas else None
        ),
        "effect_median_delta_positive_favors_model_a": fmt_float(
            median(effect_deltas) if effect_deltas else None
        ),
        "effect_bootstrap_ci_low": fmt_float(ci_low),
        "effect_bootstrap_ci_high": fmt_float(ci_high),
        "model_a_better_n": model_a_better,
        "model_b_better_n": model_b_better,
        "tie_n": ties,
        "model_a_better_rate": fmt_float(model_a_better / n_pairs if n_pairs else None),
        "pair_validity_note": (
            "Paired comparison uses rows where both models have finite metric values."
            if n_scope_images != n_pairs
            else ""
        ),
    }


def build_summary_rows(
    subsets: dict[str, list[dict[str, str]]],
    model_pairs: list[tuple[str, str]],
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for scope, rows in subsets.items():
        n_scope_images = len(rows)
        for metric in METRICS:
            for model_a, model_b in model_pairs:
                pair_rows = [
                    row for row in build_pair_rows({scope: rows}, model_a, model_b, metric)
                ]
                out.append(
                    summarize_pair(
                        scope,
                        metric,
                        model_a,
                        model_b,
                        n_scope_images,
                        pair_rows,
                        bootstrap_iterations,
                        rng,
                    )
                )
    return out


def keyed(rows: list[dict[str, object]], fields: list[str]) -> dict[tuple[object, ...], dict[str, object]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def write_markdown_summary(
    path: Path,
    convnext_rows: list[dict[str, object]],
    bootstrap_iterations: int,
    seed: int,
) -> None:
    by_key = keyed(convnext_rows, ["scope", "metric", "model_b"])
    lines = [
        "# Paired Overlap Model Comparison",
        "",
        "Reference model: `ConvNeXt_Base`.",
        "",
        "Effect-delta convention:",
        "",
        "```text",
        "For higher-is-better metrics:",
        "  effect_delta = ConvNeXt_Base - comparator",
        "",
        "For peak_distance_px:",
        "  effect_delta = comparator - ConvNeXt_Base",
        "",
        "Therefore, positive effect_delta always favors ConvNeXt_Base.",
        "```",
        "",
        f"Bootstrap mean-delta CI: {bootstrap_iterations} resamples, seed = {seed}.",
        "",
    ]

    for scope in ["main_yes", "sensitivity_yes_cautious"]:
        lines += [
            f"## {scope}",
            "",
            "| Metric | Comparator | N pairs | Mean effect delta | 95% bootstrap CI | ConvNeXt better rate | Note |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
        for metric in METRICS:
            for comparator in ["ResNet50", "Swin_Base", "ViT_B16"]:
                row = by_key[(scope, metric, comparator)]
                ci = f"{row['effect_bootstrap_ci_low']} to {row['effect_bootstrap_ci_high']}"
                lines.append(
                    f"| `{metric}` | `{comparator}` | {row['n_pairs']} | "
                    f"{row['effect_mean_delta_positive_favors_model_a']} | {ci} | "
                    f"{row['model_a_better_rate']} | {row['pair_validity_note']} |"
                )
        lines.append("")

    lines += [
        "## Interpretation Notes",
        "",
        "- Positive effect deltas favor ConvNeXt_Base for every metric.",
        "- `peak_distance_px` uses the same positive-favors-ConvNeXt convention by reversing the raw delta.",
        "- Rows with invalid metric values are excluded pairwise. This mainly affects ResNet50 `peak_distance_px` because of zero-saliency maps.",
        "- Use the paired rows to support statements about consistency across the same image set, not just model-level mean ranking.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    rows = read_csv(args.master)
    subsets = subset_rows(rows)
    convnext_pairs = [(REFERENCE_MODEL, model) for model in MODELS if model != REFERENCE_MODEL]
    all_model_pairs = list(combinations(MODELS, 2))

    convnext_summary_rows = build_summary_rows(
        subsets,
        convnext_pairs,
        args.bootstrap_iterations,
        args.seed,
    )
    all_pair_summary_rows = build_summary_rows(
        subsets,
        all_model_pairs,
        args.bootstrap_iterations,
        args.seed,
    )
    convnext_long_rows = build_long_rows(subsets, REFERENCE_MODEL)

    write_csv(
        args.output_dir / "paired_overlap_convnext_vs_others_summary.csv",
        convnext_summary_rows,
        list(convnext_summary_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "paired_overlap_all_model_pairs_summary.csv",
        all_pair_summary_rows,
        list(all_pair_summary_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "paired_overlap_convnext_vs_others_long.csv",
        convnext_long_rows,
        list(convnext_long_rows[0].keys()),
    )
    write_markdown_summary(
        args.output_dir / "paired_overlap_model_comparison_summary.md",
        convnext_summary_rows,
        args.bootstrap_iterations,
        args.seed,
    )

    summary = {
        "run_id": "paired_overlap_model_comparison_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_master": str(args.master),
        "n_source_rows": len(rows),
        "scopes": {scope: len(scope_rows) for scope, scope_rows in subsets.items()},
        "reference_model": REFERENCE_MODEL,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "outputs": {
            "convnext_vs_others_summary": str(
                args.output_dir / "paired_overlap_convnext_vs_others_summary.csv"
            ),
            "all_model_pairs_summary": str(
                args.output_dir / "paired_overlap_all_model_pairs_summary.csv"
            ),
            "convnext_vs_others_long": str(
                args.output_dir / "paired_overlap_convnext_vs_others_long.csv"
            ),
            "markdown": str(args.output_dir / "paired_overlap_model_comparison_summary.md"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_overlap_model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
