"""Build compact paper-facing subgroup tables from subgroup analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBGROUP_DIR = EXPERIMENT_ROOT / "overlap_analysis" / "subgroup_analyses"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "paper_draft_materials" / "tables"

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]
KEY_TRUE_LABEL_METRICS = [
    "normalized_energy",
    "peak_in_mask",
    "iou_top20",
    "dice_top20",
]
CONFUSION_GROUPS = ["TP", "FN", "FP", "TN"]


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


def fmt(value: str) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def keyed(rows: list[dict[str, str]], fields: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def build_true_label_key_metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_key = keyed(rows, ["scope", "true_label", "model", "metric"])
    scopes = ["main_yes", "sensitivity_yes_cautious"]
    labels = ["malignant", "benign"]
    output: list[dict[str, object]] = []
    for scope in scopes:
        for true_label in labels:
            first = by_key[(scope, true_label, MODELS[0], KEY_TRUE_LABEL_METRICS[0])]
            for metric in KEY_TRUE_LABEL_METRICS:
                row: dict[str, object] = {
                    "scope": scope,
                    "true_label": true_label,
                    "n_images": first["n_images"],
                    "metric": metric,
                }
                for model in MODELS:
                    source = by_key[(scope, true_label, model, metric)]
                    row[f"{model}_mean"] = fmt(source["mean"])
                    row[f"{model}_median"] = fmt(source["median"])
                output.append(row)
    return output


def build_confusion_group_dice(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_key = keyed(rows, ["scope", "confusion_group", "model", "metric"])
    scopes = ["main_yes", "sensitivity_yes_cautious"]
    output: list[dict[str, object]] = []
    for scope in scopes:
        for model in MODELS:
            row: dict[str, object] = {
                "scope": scope,
                "model": model,
                "metric": "dice_top20",
            }
            for group in CONFUSION_GROUPS:
                source = by_key[(scope, group, model, "dice_top20")]
                row[f"{group}_n"] = source["n_images"]
                row[f"{group}_mean"] = fmt(source["mean"])
                row[f"{group}_median"] = fmt(source["median"])
            output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgroup-dir", type=Path, default=DEFAULT_SUBGROUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    true_label_rows = read_csv(args.subgroup_dir / "overlap_by_true_label_summary.csv")
    confusion_rows = read_csv(args.subgroup_dir / "overlap_by_confusion_group_summary.csv")

    true_label_key_metrics = build_true_label_key_metrics(true_label_rows)
    confusion_group_dice = build_confusion_group_dice(confusion_rows)

    true_label_fields = ["scope", "true_label", "n_images", "metric"]
    for model in MODELS:
        true_label_fields.extend([f"{model}_mean", f"{model}_median"])

    confusion_fields = ["scope", "model", "metric"]
    for group in CONFUSION_GROUPS:
        confusion_fields.extend([f"{group}_n", f"{group}_mean", f"{group}_median"])

    true_label_output = args.output_dir / "tableS_subgroup_true_label_key_metrics.csv"
    confusion_output = args.output_dir / "tableS_subgroup_confusion_group_dice.csv"
    write_csv(true_label_output, true_label_key_metrics, true_label_fields)
    write_csv(confusion_output, confusion_group_dice, confusion_fields)

    summary = {
        "run_id": "paper_ready_subgroup_tables_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_subgroup_dir": str(args.subgroup_dir),
        "outputs": {
            "true_label_key_metrics": str(true_label_output),
            "confusion_group_dice": str(confusion_output),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_ready_subgroup_tables_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
