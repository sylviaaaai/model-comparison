"""Render four-model QC panels for a mask-QC subset from the master table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from render_overlap_smoke_qc import PRIMARY_METHODS, primary_rows, read_csv, render_case, write_csv


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_MASTER = METADATA_DIR / "test_image_master_table.csv"
DEFAULT_ANALYSIS = METADATA_DIR / "analysis_ready_overlap_manifest.csv"
DEFAULT_IMAGES = METADATA_DIR / "test_images_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "qc_figures" / "mask_qc_subset_top20"


def read_master(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_rows(
    rows: list[dict[str, str]],
    mask_qc_type: str,
    max_cases: int,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["mask_qc_type"] == mask_qc_type]
    selected = sorted(
        selected,
        key=lambda row: (
            row["patient_id"],
            row["image_id"],
        ),
    )
    if max_cases > 0:
        selected = selected[:max_cases]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mask-qc-type", default="diffuse_large")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    args = parser.parse_args()

    master_rows = read_master(args.master)
    selected = select_rows(master_rows, args.mask_qc_type, args.max_cases)
    analysis_rows = primary_rows(read_csv(args.analysis_manifest))
    image_rows = {row["image_id"]: row for row in read_csv(args.image_manifest)}
    by_image_model = {(row["image_id"], row["model_name"]): row for row in analysis_rows}

    index_rows: list[dict[str, object]] = []
    for row in selected:
        image_id = row["image_id"]
        rows_by_model = {
            model_name: by_image_model[(image_id, model_name)]
            for model_name in PRIMARY_METHODS
        }
        group = (
            f"mask_qc={row['mask_qc_type']} | "
            f"usable={row['usable_for_overlap']} | "
            f"mask_area={row['mask_area_fraction']}"
        )
        output_path = args.output_dir / args.mask_qc_type / f"{image_id}_top20_four_model_qc.png"
        render_case(
            image_id=image_id,
            group=group,
            rows_by_model=rows_by_model,
            roi_image_path=image_rows[image_id]["roi_image_path"],
            output_path=output_path,
            top_fraction=args.top_fraction,
        )
        index_rows.append(
            {
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "true_label": row["true_label"],
                "manual_mask_type": row["manual_mask_type"],
                "mask_qc_type": row["mask_qc_type"],
                "usable_for_overlap": row["usable_for_overlap"],
                "mask_area_fraction": row["mask_area_fraction"],
                "manual_review_notes": row["manual_review_notes"],
                "figure_path": str(output_path),
            }
        )

    write_csv(
        args.output_dir / f"{args.mask_qc_type}_figure_index.csv",
        index_rows,
        [
            "image_id",
            "patient_id",
            "true_label",
            "manual_mask_type",
            "mask_qc_type",
            "usable_for_overlap",
            "mask_area_fraction",
            "manual_review_notes",
            "figure_path",
        ],
    )
    print(f"Rendered {len(index_rows)} {args.mask_qc_type} QC panels to {args.output_dir}")


if __name__ == "__main__":
    main()
