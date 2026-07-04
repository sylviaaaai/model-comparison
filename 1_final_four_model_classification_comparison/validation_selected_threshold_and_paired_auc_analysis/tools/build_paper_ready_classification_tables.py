"""Build compact human-readable classification tables for paper drafting."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFICATION_DIR = EXPERIMENT_ROOT / "classification_analysis" / "threshold_auc"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "paper_draft_materials" / "tables"


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


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_threshold(value: str | float) -> str:
    return f"{float(value):.4f}"


def fmt_ci(row: dict[str, str], metric: str) -> str:
    return (
        f"{fmt(row[metric])} "
        f"[{fmt(row[f'{metric}_ci95_low'])}, {fmt(row[f'{metric}_ci95_high'])}]"
    )


def fmt_delta_ci(row: dict[str, str]) -> str:
    return (
        f"{fmt(row['delta_auc_a_minus_b'])} "
        f"[{fmt(row['delta_auc_bootstrap_ci95_low'])}, "
        f"{fmt(row['delta_auc_bootstrap_ci95_high'])}]"
    )


def confusion(row: dict[str, str]) -> str:
    return f"{int(float(row['tp']))}/{int(float(row['tn']))}/{int(float(row['fp']))}/{int(float(row['fn']))}"


def direction(row: dict[str, str]) -> str:
    low = float(row["delta_auc_bootstrap_ci95_low"])
    high = float(row["delta_auc_bootstrap_ci95_high"])
    delta = float(row["delta_auc_a_minus_b"])
    model_a = row["model_a_display"]
    model_b = row["model_b_display"]
    if low > 0:
        return f"{model_a} higher"
    if high < 0:
        return f"{model_b} higher"
    if delta > 0:
        return f"{model_a} numerically higher; CI crosses 0"
    if delta < 0:
        return f"{model_b} numerically higher; CI crosses 0"
    return "No numerical difference; CI crosses 0"


def build_validation_threshold_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    output = []
    for row in rows:
        output.append(
            {
                "Model": row["model_display"],
                "Selected threshold": fmt_threshold(row["selected_threshold"]),
                "Validation AUC": fmt(row["auc"]),
                "Sensitivity": fmt(row["sensitivity"]),
                "Specificity": fmt(row["specificity"]),
                "F1": fmt(row["f1"]),
                "TP/TN/FP/FN": confusion(row),
            }
        )
    fields = ["Model", "Selected threshold", "Validation AUC", "Sensitivity", "Specificity", "F1", "TP/TN/FP/FN"]
    return output, fields


def build_test_performance_table(rows: list[dict[str, str]], include_threshold: bool) -> tuple[list[dict[str, str]], list[str]]:
    output = []
    for row in rows:
        readable = {
            "Model": row["model_display"],
            "AUC (95% CI)": fmt_ci(row, "auc"),
            "Accuracy (95% CI)": fmt_ci(row, "accuracy"),
            "Sensitivity (95% CI)": fmt_ci(row, "sensitivity"),
            "Specificity (95% CI)": fmt_ci(row, "specificity"),
            "Precision (95% CI)": fmt_ci(row, "precision"),
            "F1 (95% CI)": fmt_ci(row, "f1"),
            "TP/TN/FP/FN": confusion(row),
        }
        if include_threshold:
            readable = {"Model": row["model_display"], "Threshold": fmt_threshold(row["threshold"]), **{k: v for k, v in readable.items() if k != "Model"}}
        output.append(readable)

    fields = [
        "Model",
        "AUC (95% CI)",
        "Accuracy (95% CI)",
        "Sensitivity (95% CI)",
        "Specificity (95% CI)",
        "Precision (95% CI)",
        "F1 (95% CI)",
        "TP/TN/FP/FN",
    ]
    if include_threshold:
        fields.insert(1, "Threshold")
    return output, fields


def build_paired_auc_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    output = []
    for row in rows:
        output.append(
            {
                "Comparison": f"{row['model_a_display']} - {row['model_b_display']}",
                "AUC model A": fmt(row["auc_a"]),
                "AUC model B": fmt(row["auc_b"]),
                "Delta AUC (A-B, 95% CI)": fmt_delta_ci(row),
                "Interpretation": direction(row),
            }
        )
    fields = ["Comparison", "AUC model A", "AUC model B", "Delta AUC (A-B, 95% CI)", "Interpretation"]
    return output, fields


def markdown_table(title: str, rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-dir", type=Path, default=DEFAULT_CLASSIFICATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    selected = read_csv(args.classification_dir / "classification_validation_selected_thresholds.csv")
    locked = read_csv(args.classification_dir / "classification_test_locked_threshold_performance.csv")
    fixed = read_csv(args.classification_dir / "classification_test_fixed_0p5_performance.csv")
    pairs = read_csv(args.classification_dir / "classification_test_auc_pairwise_bootstrap.csv")

    validation_rows, validation_fields = build_validation_threshold_table(selected)
    locked_rows, locked_fields = build_test_performance_table(locked, include_threshold=True)
    fixed_rows, fixed_fields = build_test_performance_table(fixed, include_threshold=False)
    pair_rows, pair_fields = build_paired_auc_table(pairs)

    outputs = {
        "validation_thresholds_csv": args.output_dir / "tableS_classification_validation_thresholds_readable.csv",
        "locked_thresholds_csv": args.output_dir / "tableS_classification_test_locked_thresholds_readable.csv",
        "fixed_0p5_csv": args.output_dir / "tableS_classification_test_fixed_0p5_readable.csv",
        "paired_auc_csv": args.output_dir / "tableS_classification_paired_auc_readable.csv",
        "markdown": args.output_dir / "tableS_classification_readable_tables.md",
        "summary_json": args.output_dir / "paper_ready_classification_tables_summary.json",
    }

    write_csv(outputs["validation_thresholds_csv"], validation_rows, validation_fields)
    write_csv(outputs["locked_thresholds_csv"], locked_rows, locked_fields)
    write_csv(outputs["fixed_0p5_csv"], fixed_rows, fixed_fields)
    write_csv(outputs["paired_auc_csv"], pair_rows, pair_fields)

    md_lines = ["# Classification Analysis Readable Tables", ""]
    md_lines.extend(markdown_table("Validation-Selected Thresholds", validation_rows, validation_fields))
    md_lines.extend(markdown_table("Locked Test Performance", locked_rows, locked_fields))
    md_lines.extend(markdown_table("Fixed 0.5 Test Performance", fixed_rows, fixed_fields))
    md_lines.extend(markdown_table("Paired Test AUC Differences", pair_rows, pair_fields))
    outputs["markdown"].write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "run_id": "paper_ready_classification_tables_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_classification_dir": str(args.classification_dir),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
