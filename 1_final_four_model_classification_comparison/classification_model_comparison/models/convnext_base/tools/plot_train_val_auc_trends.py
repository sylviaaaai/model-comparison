#!/usr/bin/env python3
"""Plot ConvNeXt train/validation AUC trends for selected unfreezing settings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT_DIR = Path("ConvNeXt") / "results" / "convnext_base_torch_unfreeze_5fold_dropout04"
DEFAULT_SETTINGS = ["stage3_4", "all_non_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/val AUC trends from ConvNeXt training logs.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <experiment-dir>/plots/stage3_4_all_non_norm_train_val_auc_trends.png.",
    )
    return parser.parse_args()


def read_fold_log(setting_dir: Path, fold: int) -> pd.DataFrame:
    parts = []
    for log_path in [setting_dir / f"fold_{fold}" / "stage1" / "training_log.csv", setting_dir / f"fold_{fold}" / "training_log.csv"]:
        if log_path.exists():
            parts.append(pd.read_csv(log_path))
    if not parts:
        raise FileNotFoundError(f"No training logs found for {setting_dir.name} fold {fold}")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    df["fold"] = fold
    return df


def read_setting_logs(experiment_dir: Path, setting: str) -> pd.DataFrame:
    setting_dir = experiment_dir / setting
    fold_dirs = sorted(setting_dir.glob("fold_*"), key=lambda path: int(path.name.split("_")[-1]))
    frames = []
    for fold_dir in fold_dirs:
        fold = int(fold_dir.name.split("_")[-1])
        frames.append(read_fold_log(setting_dir, fold))
    if not frames:
        raise FileNotFoundError(f"No fold directories found for {setting_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("epoch", as_index=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        val_auc_mean=("val_auc", "mean"),
        val_auc_std=("val_auc", "std"),
        gap_mean=("auc", lambda values: float(np.mean(values))),
        folds=("fold", "nunique"),
    )
    gap = df.assign(gap=df["auc"] - df["val_auc"]).groupby("epoch", as_index=False).agg(
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
    )
    grouped = grouped.drop(columns=["gap_mean"]).merge(gap, on="epoch")
    return grouped.fillna(0.0)


def setting_label(setting: str) -> str:
    return {
        "stage3_4": "Stage 3-4",
        "all_non_norm": "All non-Norm",
    }.get(setting, setting)


def plot_setting(ax_curve, ax_gap, df: pd.DataFrame, summary: pd.DataFrame, label: str) -> None:
    for fold, fold_df in df.groupby("fold"):
        ax_curve.plot(fold_df["epoch"], fold_df["auc"], color="#4f79a7", alpha=0.18, linewidth=1)
        ax_curve.plot(fold_df["epoch"], fold_df["val_auc"], color="#c4513c", alpha=0.18, linewidth=1)
        ax_gap.plot(fold_df["epoch"], fold_df["auc"] - fold_df["val_auc"], color="#6f6f6f", alpha=0.18, linewidth=1)

    epochs = summary["epoch"].to_numpy()
    train_mean = summary["auc_mean"].to_numpy()
    train_std = summary["auc_std"].to_numpy()
    val_mean = summary["val_auc_mean"].to_numpy()
    val_std = summary["val_auc_std"].to_numpy()
    gap_mean = summary["gap_mean"].to_numpy()
    gap_std = summary["gap_std"].to_numpy()

    ax_curve.plot(epochs, train_mean, color="#1f5a95", marker="o", markersize=3, linewidth=2.2, label="Train AUC mean")
    ax_curve.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="#1f5a95", alpha=0.12)
    ax_curve.plot(epochs, val_mean, color="#b53d2d", marker="o", markersize=3, linewidth=2.2, label="Val AUC mean")
    ax_curve.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="#b53d2d", alpha=0.12)
    ax_curve.axvline(8.5, color="#7d7d7d", linestyle="--", linewidth=1, alpha=0.7)
    ax_curve.set_title(label)
    ax_curve.set_ylabel("AUC")
    ax_curve.set_ylim(0.58, 1.0)
    ax_curve.grid(axis="y", alpha=0.25)
    ax_curve.legend(loc="lower right", fontsize=9)

    ax_gap.plot(epochs, gap_mean, color="#4b4b4b", marker="o", markersize=3, linewidth=2.0, label="Train - Val AUC")
    ax_gap.fill_between(epochs, gap_mean - gap_std, gap_mean + gap_std, color="#4b4b4b", alpha=0.12)
    ax_gap.axhline(0, color="#9a9a9a", linewidth=1)
    ax_gap.axvline(8.5, color="#7d7d7d", linestyle="--", linewidth=1, alpha=0.7)
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("AUC gap")
    ax_gap.grid(axis="y", alpha=0.25)


def main() -> None:
    args = parse_args()
    output = args.output or args.experiment_dir / "plots" / "stage3_4_all_non_norm_train_val_auc_trends.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(args.settings), figsize=(7 * len(args.settings), 8), sharex=False)
    if len(args.settings) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, setting in enumerate(args.settings):
        df = read_setting_logs(args.experiment_dir, setting)
        summary = summarize_by_epoch(df)
        plot_setting(axes[0, col], axes[1, col], df, summary, setting_label(setting))

    fig.suptitle("ConvNeXt-Base train/validation AUC trends across 5 folds", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
