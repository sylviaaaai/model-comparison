"""Build a manual review template for model attention to bright-dot markers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "limitation_analysis" / "bright_dot_marker" / "bright_dot_image_figure_index.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "limitation_analysis" / "bright_dot_marker" / "bright_dot_attention_review.csv"

MODELS = ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"]

FIELDNAMES = [
    "image_id",
    "label",
    "bright_dot_or_marker",
    "top20_figure_path",
    *[f"{model}_pred_type" for model in MODELS],
    *[f"{model}_attends_marker" for model in MODELS],
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    for source in read_csv(args.input):
        row: dict[str, object] = {
            "image_id": source["image_id"],
            "label": source["label"],
            "bright_dot_or_marker": source["bright_dot_or_marker"],
            "top20_figure_path": source["top20_figure_path"],
        }
        for model in MODELS:
            row[f"{model}_pred_type"] = source[f"{model}_pred_type"]
            row[f"{model}_attends_marker"] = ""
        rows.append(row)

    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} bright-dot attention review rows to {args.output}")


if __name__ == "__main__":
    main()
