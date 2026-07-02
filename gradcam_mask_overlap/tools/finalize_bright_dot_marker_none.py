"""Fill remaining blank bright-dot / marker review cells with none."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = EXPERIMENT_ROOT / "metadata" / "manual_mask_review_template.csv"


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
    filled = 0
    for row in rows:
        if not row.get("bright_dot_or_marker"):
            row["bright_dot_or_marker"] = "none"
            filled += 1

    write_csv(args.review_table, rows, fieldnames)
    print(f"Filled {filled} blank bright_dot_or_marker cells with none")


if __name__ == "__main__":
    main()
