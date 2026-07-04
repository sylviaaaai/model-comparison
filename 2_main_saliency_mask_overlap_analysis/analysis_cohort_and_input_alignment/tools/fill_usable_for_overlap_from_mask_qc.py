"""Fill usable_for_overlap from manual mask type and mask-QC type.

Bright-dot / marker fields are intentionally not used in this rule. Marker
artifacts are handled as a separate sensitivity / limitation track.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_REVIEW = METADATA_DIR / "manual_mask_review_template.csv"
DEFAULT_REVIEW_CANDIDATES = METADATA_DIR / "usable_for_overlap_review_candidates.csv"
DEFAULT_SUMMARY = METADATA_DIR / "usable_for_overlap_rule_summary.json"

YES_MANUAL_TYPES = {"localized_mass_like", "localized_malignant_annotation"}
CAUTIOUS_MANUAL_TYPES = {
    "calcification_region",
    "architectural_distortion_like",
    "uncertain_annotation",
}
YES_QC_TYPES = {"tight_or_reasonable"}
CAUTIOUS_QC_TYPES = {"overinclusive_annotation", "border_touching", "diffuse_large"}
NO_QC_TYPES = {
    "foreground_like_or_contour",
    "cropped_or_truncated",
    "bad_or_mismatched",
}

MANUAL_OVERRIDES = {
    "0019_C_0019_1.LEFT_CC": {
        "manual_mask_type": "localized_malignant_annotation",
        "mask_qc_type": "tight_or_reasonable",
        "usable_for_overlap": "yes",
        "note_set": "manual correction 2026-07-01: LEFT view is localized_malignant_annotation, not diffuse_large",
        "rule_reason": "manual_override: localized malignant annotation with tight_or_reasonable mask",
    },
    "0019_C_0019_1.LEFT_MLO": {
        "manual_mask_type": "localized_malignant_annotation",
        "mask_qc_type": "tight_or_reasonable",
        "usable_for_overlap": "yes",
        "note_set": "manual correction 2026-07-01: LEFT view is localized_malignant_annotation, not diffuse_large",
        "rule_reason": "manual_override: localized malignant annotation with tight_or_reasonable mask",
    },
    "0019_C_0019_1.RIGHT_CC": {
        "usable_for_overlap": "cautious",
        "note_set": "manual correction 2026-07-01: RIGHT view is the diffuse_large problem side; usable_for_overlap set to cautious",
        "rule_reason": "manual_override: diffuse_large reviewed as cautious",
    },
    "0019_C_0019_1.RIGHT_MLO": {
        "usable_for_overlap": "cautious",
        "note_set": "manual correction 2026-07-01: RIGHT view is the diffuse_large problem side; usable_for_overlap set to cautious",
        "rule_reason": "manual_override: diffuse_large reviewed as cautious",
    },
}

REVIEW_FIELDNAMES = [
    "image_id",
    "patient_id",
    "patient_num",
    "label",
    "class_name",
    "laterality",
    "view",
    "mask_area_fraction",
    "manual_mask_type",
    "mask_qc_type",
    "bright_dot_or_marker",
    "usable_for_overlap",
    "rule_reason",
    "notes",
    "roi_image_path",
    "roi_mask_path",
    "qc_overlay_path",
]


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


def decide_usable(manual_mask_type: str, mask_qc_type: str) -> tuple[str, str]:
    if mask_qc_type in NO_QC_TYPES:
        return "no", f"mask_qc_type={mask_qc_type}"
    if mask_qc_type in CAUTIOUS_QC_TYPES:
        return "cautious", f"mask_qc_type={mask_qc_type}"
    if manual_mask_type in CAUTIOUS_MANUAL_TYPES:
        return "cautious", f"manual_mask_type={manual_mask_type}"
    if manual_mask_type in YES_MANUAL_TYPES and mask_qc_type in YES_QC_TYPES:
        return "yes", "localized lesion annotation with tight_or_reasonable mask"
    return "cautious", f"fallback manual_mask_type={manual_mask_type};mask_qc_type={mask_qc_type}"


def append_note(existing: str, addition: str) -> str:
    existing = existing.strip()
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--review-candidates", type=Path, default=DEFAULT_REVIEW_CANDIDATES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_csv(args.review)
    rule_reasons: dict[str, str] = {}
    for row in rows:
        usable, reason = decide_usable(row["manual_mask_type"], row["mask_qc_type"])
        row["usable_for_overlap"] = usable
        rule_reasons[row["image_id"]] = reason
        override = MANUAL_OVERRIDES.get(row["image_id"])
        if override:
            for field in ["manual_mask_type", "mask_qc_type", "usable_for_overlap"]:
                if field in override:
                    row[field] = override[field]
            if "note_set" in override:
                row["notes"] = override["note_set"]
            if "note_append" in override:
                row["notes"] = append_note(row.get("notes", ""), override["note_append"])
            rule_reasons[row["image_id"]] = override["rule_reason"]

    fieldnames = list(rows[0].keys())
    write_csv(args.review, rows, fieldnames)

    review_candidates = []
    for row in rows:
        if row["usable_for_overlap"] in {"cautious", "no"}:
            review_row = {field: row.get(field, "") for field in REVIEW_FIELDNAMES}
            review_row["rule_reason"] = rule_reasons[row["image_id"]]
            review_candidates.append(review_row)
    write_csv(args.review_candidates, review_candidates, REVIEW_FIELDNAMES)

    summary = {
        "n_rows": len(rows),
        "usable_for_overlap_counts": dict(Counter(row["usable_for_overlap"] for row in rows)),
        "manual_mask_type_counts": dict(Counter(row["manual_mask_type"] for row in rows)),
        "mask_qc_type_counts": dict(Counter(row["mask_qc_type"] for row in rows)),
        "review_candidates_rows": len(review_candidates),
        "rule": {
            "yes": {
                "manual_mask_type": sorted(YES_MANUAL_TYPES),
                "mask_qc_type": sorted(YES_QC_TYPES),
                "description": "localized mass-like or localized malignant annotation with tight/reasonable mask",
            },
            "cautious": {
                "manual_mask_type": sorted(CAUTIOUS_MANUAL_TYPES),
                "mask_qc_type": sorted(CAUTIOUS_QC_TYPES),
                "description": "non-mass-like/uncertain annotation or mild mask-QC concern",
            },
            "no": {
                "mask_qc_type": sorted(NO_QC_TYPES),
                "description": "diffuse, foreground-like, cropped/truncated, or mismatched/bad mask",
            },
            "bright_dot_policy": "bright_dot_or_marker is not used to decide usable_for_overlap",
        },
        "manual_overrides": MANUAL_OVERRIDES,
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Updated {len(rows)} rows in {args.review}")
    print(f"Wrote {len(review_candidates)} cautious/no review rows to {args.review_candidates}")
    print(f"Wrote summary to {args.summary}")
    print("usable_for_overlap counts:", summary["usable_for_overlap_counts"])


if __name__ == "__main__":
    main()
