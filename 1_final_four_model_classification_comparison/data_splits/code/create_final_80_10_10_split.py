#!/usr/bin/env python3
"""Create final 80/10/10 patient-level splits from existing split CSVs."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_DIR = Path("results/splits")
OUT_DIR = SOURCE_DIR / "final_80_10_10"
SEED = 123


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_test_patients(
    test_patients: list[dict[str, str]],
    seed: int,
) -> tuple[set[str], set[str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_patients:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    val_ids: set[str] = set()
    final_test_ids: set[str] = set()

    for label, rows in sorted(by_label.items()):
        rows = rows[:]
        rng.shuffle(rows)
        val_count = len(rows) // 2
        val_rows = rows[:val_count]
        test_rows = rows[val_count:]
        val_ids.update(row["patient_id"] for row in val_rows)
        final_test_ids.update(row["patient_id"] for row in test_rows)

    return val_ids, final_test_ids


def filter_patients(rows: list[dict[str, str]], patient_ids: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row["patient_id"] in patient_ids]


def summarize(patient_rows: list[dict[str, str]], image_rows: list[dict[str, str]]) -> dict[str, object]:
    patient_counts = Counter(row["class_name"] for row in patient_rows)
    image_counts = Counter(row["class_name"] for row in image_rows)
    return {
        "patients": len(patient_rows),
        "images": len(image_rows),
        "patient_counts": dict(sorted(patient_counts.items())),
        "image_counts": dict(sorted(image_counts.items())),
    }


def assert_disjoint(*groups: set[str]) -> None:
    seen: set[str] = set()
    for group in groups:
        overlap = seen & group
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Patient leakage detected: {examples}")
        seen.update(group)


def main() -> None:
    train_patients = read_csv(SOURCE_DIR / "train_patients.csv")
    train_images = read_csv(SOURCE_DIR / "train_images.csv")
    test_patients = read_csv(SOURCE_DIR / "test_patients.csv")
    test_images = read_csv(SOURCE_DIR / "test_images.csv")

    val_ids, final_test_ids = split_test_patients(test_patients, SEED)
    train_ids = {row["patient_id"] for row in train_patients}
    assert_disjoint(train_ids, val_ids, final_test_ids)

    final_val_patients = filter_patients(test_patients, val_ids)
    final_test_patients = filter_patients(test_patients, final_test_ids)
    final_val_images = filter_patients(test_images, val_ids)
    final_test_images = filter_patients(test_images, final_test_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DIR / "train_patients.csv", OUT_DIR / "final_train_patients.csv")
    shutil.copy2(SOURCE_DIR / "train_images.csv", OUT_DIR / "final_train_images.csv")
    write_csv(
        OUT_DIR / "final_val_patients.csv",
        final_val_patients,
        ["patient_id", "label", "class_name"],
    )
    write_csv(
        OUT_DIR / "final_val_images.csv",
        final_val_images,
        ["path", "patient_id", "label", "class_name"],
    )
    write_csv(
        OUT_DIR / "final_test_patients.csv",
        final_test_patients,
        ["patient_id", "label", "class_name"],
    )
    write_csv(
        OUT_DIR / "final_test_images.csv",
        final_test_images,
        ["path", "patient_id", "label", "class_name"],
    )

    summary = {
        "source_dir": str(SOURCE_DIR),
        "seed": SEED,
        "split": "existing train kept as 80%; existing test split 50/50 into final val/test",
        "final_train": summarize(train_patients, train_images),
        "final_val": summarize(final_val_patients, final_val_images),
        "final_test": summarize(final_test_patients, final_test_images),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote final split files to {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
