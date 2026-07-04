"""Build a manual mask review template with image and mask references."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_ROOT.parent
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
MANUAL_LABEL_DIR = WORKSPACE_ROOT / "mask_qc_outputs" / "original_mask_binary_qc" / "manual_test_labels"

DEFAULT_MASTER = METADATA_DIR / "test_image_master_table.csv"
DEFAULT_MASKS = METADATA_DIR / "test_roi_mask_manifest.csv"
DEFAULT_OLD_MANUAL = MANUAL_LABEL_DIR / "manual_test_labels_image_level.csv"
DEFAULT_OUTPUT = METADATA_DIR / "manual_mask_review_template.csv"

FIELDNAMES = [
    "case_id",
    "image_id",
    "patient_id",
    "patient_num",
    "label",
    "class_name",
    "laterality",
    "view",
    "original_note",
    "previous_manual_annotation_type",
    "previous_manual_annotation_type_cn",
    "previous_usable_for_overlap_manual",
    "previous_mapping_scope",
    "previous_manual_experiment_group",
    "mask_area_fraction",
    "mask_qc_status",
    "mask_qc_note",
    "original_auto_suspicious",
    "original_auto_suspicious_reasons",
    "roi_image_path",
    "original_image_path",
    "roi_mask_path",
    "qc_overlay_path",
    "original_mask_path",
    "binary_filled_mask_path",
    "manual_mask_type",
    "mask_qc_type",
    "bright_dot_or_marker",
    "usable_for_overlap",
    "notes",
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


def patient_num(patient_id: str) -> str:
    return patient_id.split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-table", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--mask-manifest", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--old-manual-labels", type=Path, default=DEFAULT_OLD_MANUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    master_rows = read_csv(args.master_table)
    mask_rows = {row["image_id"]: row for row in read_csv(args.mask_manifest)}
    old_manual_rows = {row["image_id"]: row for row in read_csv(args.old_manual_labels)}

    output_rows: list[dict[str, object]] = []
    for row in sorted(master_rows, key=lambda item: (item["class_name"], item["patient_id"], item["image_id"])):
        image_id = row["image_id"]
        mask = mask_rows.get(image_id, {})
        old = old_manual_rows.get(image_id, {})
        output_rows.append(
            {
                "case_id": image_id,
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "patient_num": old.get("patient_num") or patient_num(row["patient_id"]),
                "label": row["true_label"],
                "class_name": row["class_name"],
                "laterality": old.get("laterality", ""),
                "view": old.get("view", ""),
                "original_note": old.get("manual_note", ""),
                "previous_manual_annotation_type": old.get("manual_annotation_type", row.get("manual_annotation_type", "")),
                "previous_manual_annotation_type_cn": old.get("manual_annotation_type_cn", ""),
                "previous_usable_for_overlap_manual": old.get("usable_for_overlap_manual", row.get("usable_for_overlap_manual", "")),
                "previous_mapping_scope": old.get("mapping_scope", ""),
                "previous_manual_experiment_group": old.get("manual_experiment_group", row.get("manual_experiment_group", "")),
                "mask_area_fraction": row.get("mask_area_fraction", ""),
                "mask_qc_status": row.get("mask_qc_status", ""),
                "mask_qc_note": row.get("mask_qc_note", ""),
                "original_auto_suspicious": old.get("original_auto_suspicious", ""),
                "original_auto_suspicious_reasons": old.get("original_auto_suspicious_reasons", ""),
                "roi_image_path": row.get("image_path", ""),
                "original_image_path": old.get("original_image_path") or mask.get("original_image_path", ""),
                "roi_mask_path": row.get("roi_mask_path", ""),
                "qc_overlay_path": mask.get("qc_overlay_path", ""),
                "original_mask_path": old.get("mask_path") or mask.get("original_mask_path", ""),
                "binary_filled_mask_path": old.get("binary_filled_mask_path", ""),
                "manual_mask_type": "",
                "mask_qc_type": "",
                "bright_dot_or_marker": "",
                "usable_for_overlap": "",
                "notes": "",
            }
        )

    write_csv(args.output, output_rows)
    print(f"Wrote {len(output_rows)} review rows to {args.output}")


if __name__ == "__main__":
    main()
