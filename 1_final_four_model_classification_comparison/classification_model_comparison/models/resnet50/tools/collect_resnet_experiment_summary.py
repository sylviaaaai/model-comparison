#!/usr/bin/env python3
"""Collect ResNet experiment directories into one overview table."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "resnet50_experiment_overview.csv"
OUT_JSON = RESULTS_DIR / "resnet50_experiment_overview.json"
OUT_MD = RESULTS_DIR / "resnet50_experiment_overview.md"
REPEATS_CSV = RESULTS_DIR / "resnet50_final_repeats_summary.csv"
REPEATS_MD = RESULTS_DIR / "resnet50_final_repeats_summary.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str | float | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def infer_config(exp_dir: Path) -> dict[str, str]:
    name = exp_dir.name
    config = {
        "experiment_dir": str(exp_dir),
        "experiment_name": name,
        "batch_size": "32" if "bs32" in name else "16",
        "preprocess_input": "yes" if "preprocess" in name else "no",
        "dropout_rate": "0.5",
        "stage2_lr": "1e-5",
        "mode": "final" if name.startswith("resnet50_final") else "cv5",
    }

    if "dropout03" in name:
        config["dropout_rate"] = "0.3"
    elif "dropout04" in name:
        config["dropout_rate"] = "0.4"

    if "lr5e6" in name:
        config["stage2_lr"] = "5e-6"

    return config


def collect_cv_rows(exp_dir: Path) -> list[dict[str, str]]:
    summary_path = exp_dir / "unfreeze_auc_summary.csv"
    if not summary_path.exists():
        return []

    config = infer_config(exp_dir)
    rows = []
    for row in read_csv(summary_path):
        rows.append(
            {
                **config,
                "unfreeze_name": row["unfreeze_name"],
                "folds_completed": row["folds_completed"],
                "mean_val_auc": fmt_float(row["mean_val_auc"]),
                "std_val_auc": fmt_float(row["std_val_auc"]),
                "min_val_auc": fmt_float(row["min_val_auc"]),
                "max_val_auc": fmt_float(row["max_val_auc"]),
                "final_val_auc": "",
                "final_test_auc": "",
                "trainable_base_layers": row["trainable_base_layers"],
                "train_images": "",
                "val_images": "",
                "test_images": "",
                "notes": row.get("description", ""),
            }
        )
    return rows


def collect_final_rows(exp_dir: Path) -> list[dict[str, str]]:
    final_path = exp_dir / "final_results.csv"
    if not final_path.exists():
        return []

    config = infer_config(exp_dir)
    rows = []
    for row in read_csv(final_path):
        rows.append(
            {
                **config,
                "batch_size": row.get("batch_size", config["batch_size"]),
                "preprocess_input": "yes"
                if row.get("use_resnet_preprocess_input") == "True"
                else "no",
                "dropout_rate": row.get("dropout_rate", config["dropout_rate"]),
                "stage2_lr": row.get("stage2_lr", config["stage2_lr"]),
                "unfreeze_name": row["unfreeze_name"],
                "folds_completed": "",
                "mean_val_auc": "",
                "std_val_auc": "",
                "min_val_auc": "",
                "max_val_auc": "",
                "final_val_auc": fmt_float(row["val_auc"]),
                "final_test_auc": fmt_float(row["test_auc"]),
                "trainable_base_layers": row["trainable_base_layers"],
                "train_images": row["train_images"],
                "val_images": row["val_images"],
                "test_images": row["test_images"],
                "run_seed": row.get("run_seed", ""),
                "notes": f"final_split_dir={row.get('final_split_dir', '')}",
            }
        )
    return rows


def sort_key(row: dict[str, str]) -> tuple[int, float]:
    if row["mode"] == "final":
        return (0, -float(row["final_test_auc"] or 0))
    return (1, -float(row["mean_val_auc"] or 0))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "mode",
        "experiment_name",
        "unfreeze_name",
        "batch_size",
        "preprocess_input",
        "dropout_rate",
        "stage2_lr",
        "folds_completed",
        "mean_val_auc",
        "std_val_auc",
        "min_val_auc",
        "max_val_auc",
        "final_val_auc",
        "final_test_auc",
        "trainable_base_layers",
        "train_images",
        "val_images",
        "test_images",
        "run_seed",
        "experiment_dir",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    display_fields = [
        "mode",
        "experiment_name",
        "unfreeze_name",
        "run_seed",
        "batch_size",
        "preprocess_input",
        "dropout_rate",
        "stage2_lr",
        "mean_val_auc",
        "std_val_auc",
        "final_val_auc",
        "final_test_auc",
    ]
    lines = [
        "# ResNet50 Experiment Overview",
        "",
        "| " + " | ".join(display_fields) + " |",
        "| " + " | ".join(["---"] * len(display_fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in display_fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def repeat_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["unfreeze_name"],
        row["batch_size"],
        row["preprocess_input"],
        row["dropout_rate"],
        row["stage2_lr"],
    )


def write_repeat_summary(rows: list[dict[str, str]]) -> None:
    final_rows = [row for row in rows if row["mode"] == "final" and row["final_test_auc"]]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in final_rows:
        grouped[repeat_key(row)].append(row)

    summary_rows = []
    for key, items in sorted(grouped.items()):
        test_aucs = [float(row["final_test_auc"]) for row in items]
        val_aucs = [float(row["final_val_auc"]) for row in items if row["final_val_auc"]]
        seeds = ", ".join(row["run_seed"] for row in items if row.get("run_seed"))
        summary_rows.append(
            {
                "unfreeze_name": key[0],
                "batch_size": key[1],
                "preprocess_input": key[2],
                "dropout_rate": key[3],
                "stage2_lr": key[4],
                "n_runs": str(len(items)),
                "seeds": seeds,
                "mean_final_test_auc": fmt_float(statistics.mean(test_aucs)),
                "std_final_test_auc": fmt_float(statistics.stdev(test_aucs) if len(test_aucs) > 1 else 0.0),
                "mean_final_val_auc": fmt_float(statistics.mean(val_aucs)) if val_aucs else "",
                "std_final_val_auc": fmt_float(statistics.stdev(val_aucs) if len(val_aucs) > 1 else 0.0) if val_aucs else "",
            }
        )

    fields = [
        "unfreeze_name",
        "batch_size",
        "preprocess_input",
        "dropout_rate",
        "stage2_lr",
        "n_runs",
        "seeds",
        "mean_final_test_auc",
        "std_final_test_auc",
        "mean_final_val_auc",
        "std_final_val_auc",
    ]
    with REPEATS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# ResNet50 Final Repeats Summary",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |")
    REPEATS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = []
    for exp_dir in sorted(RESULTS_DIR.glob("resnet50*")):
        if not exp_dir.is_dir():
            continue
        rows.extend(collect_final_rows(exp_dir))
        rows.extend(collect_cv_rows(exp_dir))

    rows = sorted(rows, key=sort_key)
    write_csv(OUT_CSV, rows)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(OUT_MD, rows)
    write_repeat_summary(rows)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {REPEATS_CSV}")
    print(f"Wrote {REPEATS_MD}")
    print(f"Collected {len(rows)} rows")


if __name__ == "__main__":
    main()
