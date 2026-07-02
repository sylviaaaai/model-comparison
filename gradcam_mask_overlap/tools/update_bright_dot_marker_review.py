"""Update bright-dot / marker annotations in the manual mask review table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = EXPERIMENT_ROOT / "metadata" / "manual_mask_review_template.csv"

BRIGHT_DOT_BY_IMAGE = {
    "0217_C_0217_1.RIGHT_MLO": "on_border",
    "0248_C_0248_1.LEFT_CC": "inside_mask",
    "0248_C_0248_1.LEFT_MLO": "inside_mask",
    "0249_C_0249_1.LEFT_CC": "outside_mask",
    "0330_C_0330_1.RIGHT_MLO": "inside_mask",
    "1771_A_1771_1.RIGHT_MLO": "inside_mask",
    "0026_C_0026_1.LEFT_CC": "outside_mask",
    "0026_C_0026_1.LEFT_MLO": "inside_mask",
    "0064_C_0064_1.LEFT_CC": "outside_mask",
    "0064_C_0064_1.LEFT_MLO": "inside_mask",
    "0064_C_0064_1.RIGHT_CC": "outside_mask",
    "0064_C_0064_1.RIGHT_MLO": "outside_mask",
    "0069_C_0069_1.RIGHT_CC": "on_border",
    "0069_C_0069_1.RIGHT_MLO": "outside_mask",
    "0166_C_0166_1.RIGHT_CC": "outside_mask",
    "0182_C_0182_1.RIGHT_CC": "outside_mask",
    "0189_C_0189_1.RIGHT_CC": "inside_mask",
    "0189_C_0189_1.RIGHT_MLO": "outside_mask",
    "0418_C_0418_1.RIGHT_MLO": "outside_mask",
    "3024_B_3024_1.LEFT_CC": "outside_mask",
    "3024_B_3024_1.LEFT_MLO": "inside_mask",
    "3042_B_3042_1.LEFT_CC": "outside_mask",
    "3042_B_3042_1.LEFT_MLO": "outside_mask",
    "3069_B_3069_1.RIGHT_CC": "uncertain",
    "3109_B_3109_1.RIGHT_CC": "inside_mask",
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-table", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    rows, fieldnames = read_csv(args.review_table)
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if image_id in BRIGHT_DOT_BY_IMAGE:
            row["bright_dot_or_marker"] = BRIGHT_DOT_BY_IMAGE[image_id]
            seen.add(image_id)
        elif row.get("bright_dot_or_marker") in {"inside_mask", "outside_mask", "on_border", "uncertain"}:
            row["bright_dot_or_marker"] = ""

    missing = sorted(set(BRIGHT_DOT_BY_IMAGE) - seen)
    if missing:
        raise SystemExit(f"Missing image_id rows for bright-dot annotations: {missing}")

    write_csv(args.review_table, rows, fieldnames)
    print(f"Updated {len(seen)} bright-dot annotations in {args.review_table}")


if __name__ == "__main__":
    main()
