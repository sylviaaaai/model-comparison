#!/usr/bin/env python3
"""Validation-selected thresholds and paired test AUC comparisons.

Thresholds are selected only on the final validation split, then applied
unchanged to the final test split. AUC comparisons use patient-level paired
bootstrap resampling on the final test split.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_DIR = ROOT / "classification_analysis" / "validation_predictions_raw"
DEFAULT_TEST_DIR = ROOT.parent / "results" / "patient_level_bootstrap_ci"
DEFAULT_OUTPUT_DIR = ROOT / "classification_analysis" / "threshold_auc"

MODEL_FILES = {
    "resnet50": "resnet50_test_predictions.csv",
    "vit_b16": "vit_b16_test_predictions.csv",
    "convnext_base": "convnext_base_test_predictions.csv",
    "swin_base": "swin_base_test_predictions.csv",
}

MODEL_DISPLAY = {
    "resnet50": "ResNet50",
    "vit_b16": "ViT-B/16",
    "convnext_base": "ConvNeXt-Base",
    "swin_base": "Swin-Base",
}

PAIR_ORDER = ["resnet50", "vit_b16", "convnext_base", "swin_base"]


@dataclass(frozen=True)
class BootstrapConfig:
    n_bootstrap: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def normalize_image_key(path: str) -> str:
    return str(path).strip().replace("\\", "/")


def read_prediction_file(path: Path, split: str, model_key: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {"model_key", "model", "setting", "path", "patient_id", "y_true", "class_name", "y_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["split"] = split
    frame["model_key"] = model_key
    frame["model_display"] = MODEL_DISPLAY[model_key]
    frame["image_key"] = frame["path"].map(normalize_image_key)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["y_score"] = frame["y_score"].astype(float)
    return frame[
        [
            "split",
            "model_key",
            "model_display",
            "model",
            "setting",
            "image_key",
            "path",
            "patient_id",
            "y_true",
            "class_name",
            "y_score",
        ]
    ]


def read_predictions(directory: Path, split: str) -> pd.DataFrame:
    frames = []
    for model_key, filename in MODEL_FILES.items():
        frames.append(read_prediction_file(directory / filename, split, model_key))
    combined = pd.concat(frames, ignore_index=True)
    validate_prediction_alignment(combined, split)
    return combined


def validate_prediction_alignment(frame: pd.DataFrame, split: str) -> None:
    counts = frame.groupby("model_key")["image_key"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"{split}: models do not have the same number of images: {counts.to_dict()}")

    label_counts = frame.groupby("image_key")["y_true"].nunique()
    if int(label_counts.max()) != 1:
        bad = label_counts[label_counts > 1].head().index.tolist()
        raise ValueError(f"{split}: inconsistent labels across models for images: {bad}")

    patient_counts = frame.groupby("image_key")["patient_id"].nunique()
    if int(patient_counts.max()) != 1:
        bad = patient_counts[patient_counts > 1].head().index.tolist()
        raise ValueError(f"{split}: inconsistent patient IDs across models for images: {bad}")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    youden_j = sensitivity + specificity - 1 if not math.isnan(sensitivity) and not math.isnan(specificity) else math.nan
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else math.nan

    return {
        "threshold": float(threshold),
        "n_images": int(len(y_true)),
        "n_malignant": int(np.sum(y_true == 1)),
        "n_benign": int(np.sum(y_true == 0)),
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
        **counts,
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.concatenate([scores.astype(float), np.array([0.0, 0.5, 1.0])]))
    return np.clip(values, 0.0, 1.0)


def build_threshold_sweep(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, group in validation.groupby("model_key", sort=False):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_score = group["y_score"].to_numpy(dtype=float)
        for threshold in threshold_grid(y_score):
            metrics = metrics_at_threshold(y_true, y_score, float(threshold))
            rows.append(
                {
                    "split": "validation",
                    "model_key": model_key,
                    "model_display": MODEL_DISPLAY[model_key],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for model_key, group in sweep.groupby("model_key", sort=False):
        ranked = group.sort_values(
            by=["youden_j", "f1", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, True],
        )
        chosen = ranked.iloc[0].copy()
        chosen["selection_rule"] = "max_validation_youden_j_tie_f1_balanced_accuracy_low_threshold"
        chosen["selected_threshold"] = float(chosen["threshold"])
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows)
    columns = [
        "split",
        "model_key",
        "model_display",
        "selection_rule",
        "selected_threshold",
        "n_images",
        "n_malignant",
        "n_benign",
        "auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "f1",
        "balanced_accuracy",
        "youden_j",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    return selected[columns]


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    threshold: float,
    config: BootstrapConfig,
    include_auc: bool = True,
) -> dict[str, float | int]:
    patient_groups = [group.index.to_numpy(dtype=int) for _, group in frame.groupby("patient_id", sort=False)]
    rng = np.random.default_rng(config.seed)
    rows = []
    attempts = 0
    while len(rows) < config.n_bootstrap and attempts < config.n_bootstrap * 10:
        attempts += 1
        sampled = rng.integers(0, len(patient_groups), size=len(patient_groups))
        indices = np.concatenate([patient_groups[index] for index in sampled])
        sample = frame.loc[indices]
        y_true = sample["y_true"].to_numpy(dtype=int)
        y_score = sample["y_score"].to_numpy(dtype=float)
        metrics = metrics_at_threshold(y_true, y_score, threshold)
        if include_auc and math.isnan(float(metrics["auc"])):
            continue
        rows.append(metrics)

    if not rows:
        raise RuntimeError("No valid bootstrap replicates were generated")

    boot = pd.DataFrame(rows)
    intervals: dict[str, float | int] = {"n_bootstrap_valid": int(len(boot))}
    metric_names = [
        "auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "f1",
        "balanced_accuracy",
        "youden_j",
    ]
    for metric in metric_names:
        values = boot[metric].dropna()
        if len(values) == 0:
            intervals[f"{metric}_ci95_low"] = math.nan
            intervals[f"{metric}_ci95_high"] = math.nan
        else:
            intervals[f"{metric}_ci95_low"] = float(values.quantile(0.025))
            intervals[f"{metric}_ci95_high"] = float(values.quantile(0.975))
    return intervals


def evaluate_test_thresholds(
    test: pd.DataFrame,
    selected: pd.DataFrame,
    fixed_threshold: float,
    config: BootstrapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_model = dict(zip(selected["model_key"], selected["selected_threshold"], strict=True))
    locked_rows = []
    fixed_rows = []
    for model_key, group in test.groupby("model_key", sort=False):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_score = group["y_score"].to_numpy(dtype=float)

        for threshold_source, threshold, rows in [
            ("validation_selected_youden_j", float(selected_by_model[model_key]), locked_rows),
            ("fixed_0.5", fixed_threshold, fixed_rows),
        ]:
            point = metrics_at_threshold(y_true, y_score, threshold)
            intervals = bootstrap_metric_intervals(group, threshold, config)
            rows.append(
                {
                    "split": "test",
                    "model_key": model_key,
                    "model_display": MODEL_DISPLAY[model_key],
                    "threshold_source": threshold_source,
                    "threshold": threshold,
                    "n_patients": int(group["patient_id"].nunique()),
                    **point,
                    **intervals,
                }
            )
    return pd.DataFrame(locked_rows), pd.DataFrame(fixed_rows)


def build_test_wide(test: pd.DataFrame) -> pd.DataFrame:
    base = (
        test[["image_key", "patient_id", "y_true", "class_name"]]
        .drop_duplicates()
        .sort_values("image_key")
        .reset_index(drop=True)
    )
    score_wide = test.pivot(index="image_key", columns="model_key", values="y_score").reset_index()
    wide = base.merge(score_wide, on="image_key", how="inner")
    missing_models = [model_key for model_key in MODEL_FILES if model_key not in wide.columns]
    if missing_models:
        raise ValueError(f"Missing model columns in wide test table: {missing_models}")
    return wide


def bootstrap_auc_by_model(wide: pd.DataFrame, config: BootstrapConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_keys = [model_key for model_key in PAIR_ORDER if model_key in wide.columns]
    patient_groups = [group.index.to_numpy(dtype=int) for _, group in wide.groupby("patient_id", sort=False)]
    rng = np.random.default_rng(config.seed)

    point_auc = {
        model_key: float(roc_auc_score(wide["y_true"].to_numpy(dtype=int), wide[model_key].to_numpy(dtype=float)))
        for model_key in model_keys
    }

    boot_rows = []
    attempts = 0
    while len(boot_rows) < config.n_bootstrap and attempts < config.n_bootstrap * 10:
        attempts += 1
        sampled = rng.integers(0, len(patient_groups), size=len(patient_groups))
        indices = np.concatenate([patient_groups[index] for index in sampled])
        sample = wide.loc[indices]
        y_true = sample["y_true"].to_numpy(dtype=int)
        if len(np.unique(y_true)) < 2:
            continue
        boot_rows.append(
            {
                model_key: float(roc_auc_score(y_true, sample[model_key].to_numpy(dtype=float)))
                for model_key in model_keys
            }
        )

    if not boot_rows:
        raise RuntimeError("No valid AUC bootstrap replicates were generated")

    boot = pd.DataFrame(boot_rows)
    auc_rows = []
    for model_key in model_keys:
        auc_rows.append(
            {
                "split": "test",
                "model_key": model_key,
                "model_display": MODEL_DISPLAY[model_key],
                "n_images": int(len(wide)),
                "n_patients": int(wide["patient_id"].nunique()),
                "n_malignant": int(np.sum(wide["y_true"].to_numpy(dtype=int) == 1)),
                "n_benign": int(np.sum(wide["y_true"].to_numpy(dtype=int) == 0)),
                "auc": point_auc[model_key],
                "auc_ci95_low": float(boot[model_key].quantile(0.025)),
                "auc_ci95_high": float(boot[model_key].quantile(0.975)),
                "n_bootstrap_valid": int(len(boot)),
            }
        )

    pair_rows = []
    for i, model_a in enumerate(model_keys):
        for model_b in model_keys[i + 1 :]:
            delta = boot[model_a] - boot[model_b]
            pair_rows.append(
                {
                    "split": "test",
                    "model_a_key": model_a,
                    "model_a_display": MODEL_DISPLAY[model_a],
                    "model_b_key": model_b,
                    "model_b_display": MODEL_DISPLAY[model_b],
                    "auc_a": point_auc[model_a],
                    "auc_b": point_auc[model_b],
                    "delta_auc_a_minus_b": point_auc[model_a] - point_auc[model_b],
                    "delta_auc_bootstrap_ci95_low": float(delta.quantile(0.025)),
                    "delta_auc_bootstrap_ci95_high": float(delta.quantile(0.975)),
                    "n_bootstrap_valid": int(len(boot)),
                    "interpretation": "positive_delta_favors_model_a",
                }
            )

    return pd.DataFrame(auc_rows), pd.DataFrame(pair_rows)


def format_ci(row: pd.Series, metric: str) -> str:
    return f"{row[metric]:.3f} [{row[f'{metric}_ci95_low']:.3f}, {row[f'{metric}_ci95_high']:.3f}]"


def write_summary(
    path: Path,
    selected: pd.DataFrame,
    locked: pd.DataFrame,
    fixed: pd.DataFrame,
    auc_by_model: pd.DataFrame,
    auc_pairs: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    lines = [
        "# Classification Threshold and Paired AUC Analysis",
        "",
        "## Design",
        "",
        "- Thresholds were selected only on the final validation split.",
        "- The selected thresholds were then locked and applied to the final test split.",
        "- Test-set AUC CIs and paired AUC differences used patient-level bootstrap resampling.",
        "- Classification analysis uses all final test images and is separate from the mask-overlap usable set.",
        "",
        "## Validation-Selected Thresholds",
        "",
        "| Model | Threshold | Val AUC | Val sensitivity | Val specificity | Val F1 | Val TP/TN/FP/FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_display"],
                    f"{row['selected_threshold']:.4f}",
                    f"{row['auc']:.3f}",
                    f"{row['sensitivity']:.3f}",
                    f"{row['specificity']:.3f}",
                    f"{row['f1']:.3f}",
                    f"{int(row['tp'])}/{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Locked Test Performance",
            "",
            "| Model | Threshold | Test AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | Test TP/TN/FP/FN |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in locked.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_display"],
                    f"{row['threshold']:.4f}",
                    format_ci(row, "auc"),
                    format_ci(row, "accuracy"),
                    format_ci(row, "sensitivity"),
                    format_ci(row, "specificity"),
                    format_ci(row, "f1"),
                    f"{int(row['tp'])}/{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Fixed 0.5 Comparator",
            "",
            "| Model | Test AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | Test TP/TN/FP/FN |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in fixed.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_display"],
                    format_ci(row, "auc"),
                    format_ci(row, "accuracy"),
                    format_ci(row, "sensitivity"),
                    format_ci(row, "specificity"),
                    format_ci(row, "f1"),
                    f"{int(row['tp'])}/{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired Test AUC",
            "",
            "| Model | Test AUC (95% CI) |",
            "| --- | --- |",
        ]
    )
    for _, row in auc_by_model.iterrows():
        lines.append(f"| {row['model_display']} | {format_ci(row, 'auc')} |")

    lines.extend(
        [
            "",
            "Pairwise paired-bootstrap AUC differences are in:",
            "",
            "```text",
            str(metadata["auc_pairwise_csv"]),
            "```",
            "",
            "Positive pairwise deltas favor `model_a`.",
            "",
            "## Notes",
            "",
            "- Validation CSV filenames retain the source script's `_test_predictions.csv` suffix, but the directory and split column identify them as validation predictions.",
            "- The ResNet50 validation predictions were exported with TensorFlow CPU; TensorFlow reported no available GPU devices.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = BootstrapConfig(n_bootstrap=args.n_bootstrap, seed=args.seed)

    validation = read_predictions(args.validation_dir, "validation")
    test = read_predictions(args.test_dir, "test")

    validation_long_csv = args.output_dir / "classification_validation_predictions_long.csv"
    test_long_csv = args.output_dir / "classification_test_predictions_long.csv"
    validation.to_csv(validation_long_csv, index=False)
    test.to_csv(test_long_csv, index=False)

    sweep = build_threshold_sweep(validation)
    selected = select_thresholds(sweep)
    locked, fixed = evaluate_test_thresholds(test, selected, 0.5, config)
    auc_by_model, auc_pairs = bootstrap_auc_by_model(build_test_wide(test), config)

    sweep_csv = args.output_dir / "classification_validation_threshold_sweep.csv"
    selected_csv = args.output_dir / "classification_validation_selected_thresholds.csv"
    locked_csv = args.output_dir / "classification_test_locked_threshold_performance.csv"
    fixed_csv = args.output_dir / "classification_test_fixed_0p5_performance.csv"
    auc_by_model_csv = args.output_dir / "classification_test_auc_by_model.csv"
    auc_pairs_csv = args.output_dir / "classification_test_auc_pairwise_bootstrap.csv"
    summary_md = args.output_dir / "classification_threshold_auc_summary.md"
    summary_json = args.output_dir / "classification_threshold_auc_summary.json"

    sweep.to_csv(sweep_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    locked.to_csv(locked_csv, index=False)
    fixed.to_csv(fixed_csv, index=False)
    auc_by_model.to_csv(auc_by_model_csv, index=False)
    auc_pairs.to_csv(auc_pairs_csv, index=False)

    metadata = {
        "analysis": "classification_threshold_and_paired_auc",
        "threshold_policy": "select_threshold_on_final_validation_split_by_max_youden_j_then_lock_for_final_test",
        "test_auc_comparison": "patient_level_paired_bootstrap",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "validation_dir": str(args.validation_dir),
        "test_dir": str(args.test_dir),
        "output_dir": str(args.output_dir),
        "validation_predictions_long_csv": str(validation_long_csv),
        "test_predictions_long_csv": str(test_long_csv),
        "threshold_sweep_csv": str(sweep_csv),
        "selected_thresholds_csv": str(selected_csv),
        "locked_test_performance_csv": str(locked_csv),
        "fixed_0p5_test_performance_csv": str(fixed_csv),
        "auc_by_model_csv": str(auc_by_model_csv),
        "auc_pairwise_csv": str(auc_pairs_csv),
        "summary_md": str(summary_md),
        "validation_n_images": int(validation.groupby("model_key")["image_key"].nunique().iloc[0]),
        "test_n_images": int(test.groupby("model_key")["image_key"].nunique().iloc[0]),
        "validation_n_patients": int(validation.groupby("model_key")["patient_id"].nunique().iloc[0]),
        "test_n_patients": int(test.groupby("model_key")["patient_id"].nunique().iloc[0]),
        "models": MODEL_DISPLAY,
    }
    summary_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(summary_md, selected, locked, fixed, auc_by_model, auc_pairs, metadata)

    print("Saved:", validation_long_csv)
    print("Saved:", test_long_csv)
    print("Saved:", sweep_csv)
    print("Saved:", selected_csv)
    print("Saved:", locked_csv)
    print("Saved:", fixed_csv)
    print("Saved:", auc_by_model_csv)
    print("Saved:", auc_pairs_csv)
    print("Saved:", summary_md)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
