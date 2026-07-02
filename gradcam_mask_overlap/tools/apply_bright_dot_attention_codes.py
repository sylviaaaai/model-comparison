"""Apply manual bright-dot attention codes and summarize them.

Codes are image-row ordered against bright_dot_attention_review.csv.
`1` means the model's top-20% saliency visibly attends to the bright dot /
marker; `0` means it does not.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
LIMITATION_DIR = EXPERIMENT_ROOT / "limitation_analysis" / "bright_dot_marker"
ATTENTION_REVIEW = LIMITATION_DIR / "bright_dot_attention_review.csv"
SUMMARY_BY_MODEL = LIMITATION_DIR / "bright_dot_attention_summary_by_model.csv"
SUMMARY_BY_CONFUSION = LIMITATION_DIR / "bright_dot_attention_by_confusion.csv"

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]

FIELDNAMES = [
    "image_id",
    "label",
    "bright_dot_or_marker",
    "top20_figure_path",
    *[f"{model}_pred_type" for model in MODELS],
    *[f"{model}_attends_marker" for model in MODELS],
]

# Row order follows bright_dot_attention_review.csv.
MANUAL_CODES = [
    ("TN", "TN", "FP", "TN", 1, 1, 1, 1),
    ("TN", "FP", "FP", "FP", 1, 1, 1, 1),
    ("FP", "TN", "FP", "FP", 1, 1, 1, 1),
    ("TN", "FP", "TN", "TN", 0, 1, 1, 0),
    ("FP", "TN", "FP", "TN", 1, 1, 1, 1),
    ("FP", "TN", "FP", "TN", 1, 0, 1, 1),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 0),
    ("TP", "TP", "TP", "TP", 0, 0, 1, 0),
    ("FN", "FN", "FN", "FN", 1, 1, 1, 0),
    ("FN", "FN", "FN", "FN", 0, 1, 1, 1),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 0),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 1),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 0),
    ("TP", "TP", "TP", "TP", 0, 0, 0, 1),
    ("TP", "TP", "TP", "TP", 0, 0, 1, 0),
    ("TP", "TP", "TP", "TP", 0, 1, 0, 0),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 0),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 0),
    ("TP", "TP", "TP", "TP", 1, 1, 1, 1),
    ("TP", "TP", "TP", "TP", 1, 1, 1, 1),
    ("TP", "TP", "TP", "TP", 1, 1, 1, 0),
    ("TP", "TP", "TP", "TP", 0, 1, 1, 1),
    ("FN", "TP", "TP", "TP", 1, 1, 1, 1),
    ("TP", "FN", "TP", "TP", 1, 1, 1, 1),
    ("TP", "TP", "TP", "TP", 1, 1, 1, 0),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{100 * numerator / denominator:.1f}"


def summarize_by_model(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for model in MODELS:
        pred_col = f"{model}_pred_type"
        att_col = f"{model}_attends_marker"
        n = len(rows)
        attends = sum(1 for row in rows if row[att_col] == "1")
        benign = [row for row in rows if row["label"] == "benign"]
        malignant = [row for row in rows if row["label"] == "malignant"]
        benign_attends = sum(1 for row in benign if row[att_col] == "1")
        malignant_attends = sum(1 for row in malignant if row[att_col] == "1")
        pred_counts = Counter(row[pred_col] for row in rows)
        benign_fp = [row for row in benign if row[pred_col] == "FP"]
        benign_tn = [row for row in benign if row[pred_col] == "TN"]
        malignant_tp = [row for row in malignant if row[pred_col] == "TP"]
        malignant_fn = [row for row in malignant if row[pred_col] == "FN"]
        benign_fp_attends = sum(1 for row in benign_fp if row[att_col] == "1")
        malignant_fn_attends = sum(1 for row in malignant_fn if row[att_col] == "1")
        out.append(
            {
                "model": model,
                "n_marker_images": n,
                "attends_marker": attends,
                "does_not_attend_marker": n - attends,
                "attention_rate_percent": pct(attends, n),
                "TP": pred_counts["TP"],
                "FN": pred_counts["FN"],
                "FP": pred_counts["FP"],
                "TN": pred_counts["TN"],
                "benign_marker_n": len(benign),
                "benign_marker_attends": benign_attends,
                "benign_marker_attention_rate_percent": pct(benign_attends, len(benign)),
                "benign_FP": len(benign_fp),
                "benign_TN": len(benign_tn),
                "benign_FP_attends_marker": benign_fp_attends,
                "benign_FP_attention_rate_percent": pct(benign_fp_attends, len(benign_fp)),
                "malignant_marker_n": len(malignant),
                "malignant_marker_attends": malignant_attends,
                "malignant_marker_attention_rate_percent": pct(malignant_attends, len(malignant)),
                "malignant_TP": len(malignant_tp),
                "malignant_FN": len(malignant_fn),
                "malignant_FN_attends_marker": malignant_fn_attends,
                "malignant_FN_attention_rate_percent": pct(malignant_fn_attends, len(malignant_fn)),
            }
        )
    return out


def summarize_by_confusion(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for model in MODELS:
        pred_col = f"{model}_pred_type"
        att_col = f"{model}_attends_marker"
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row[pred_col]].append(row)
        for pred_type in ["TP", "FN", "FP", "TN"]:
            group = grouped[pred_type]
            attends = sum(1 for row in group if row[att_col] == "1")
            out.append(
                {
                    "model": model,
                    "pred_type": pred_type,
                    "n": len(group),
                    "attends_marker": attends,
                    "does_not_attend_marker": len(group) - attends,
                    "attention_rate_percent": pct(attends, len(group)),
                }
            )
    return out


def main() -> None:
    rows = read_csv(ATTENTION_REVIEW)
    if len(rows) != len(MANUAL_CODES):
        raise ValueError(f"Expected {len(MANUAL_CODES)} rows, found {len(rows)}")

    for row, code in zip(rows, MANUAL_CODES, strict=True):
        expected_pred = code[:4]
        actual_pred = tuple(row[f"{model}_pred_type"] for model in MODELS)
        if actual_pred != expected_pred:
            raise ValueError(
                f"Prediction row mismatch for {row['image_id']}: "
                f"expected {expected_pred}, found {actual_pred}"
            )
        for model, value in zip(MODELS, code[4:], strict=True):
            row[f"{model}_attends_marker"] = str(value)

    write_csv(ATTENTION_REVIEW, rows, FIELDNAMES)

    model_summary = summarize_by_model(rows)
    confusion_summary = summarize_by_confusion(rows)
    write_csv(SUMMARY_BY_MODEL, model_summary, list(model_summary[0].keys()))
    write_csv(SUMMARY_BY_CONFUSION, confusion_summary, list(confusion_summary[0].keys()))

    print(f"Updated {ATTENTION_REVIEW}")
    print(f"Wrote {SUMMARY_BY_MODEL}")
    print(f"Wrote {SUMMARY_BY_CONFUSION}")


if __name__ == "__main__":
    main()
