"""Seed the new manual mask review table from the latest reviewer notes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "metadata" / "manual_mask_review_template.csv"
DEFAULT_OUTPUT = DEFAULT_INPUT

TARGET_COLUMNS = ["manual_mask_type", "mask_qc_type", "notes"]


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


def norm_patient(value: str) -> str:
    value = str(value).strip()
    return str(int(value)) if value.isdigit() else value.lstrip("0")


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        norm_patient(row["patient_num"]),
        row.get("laterality", "").upper(),
        row.get("view", "").upper(),
    )


def manual_type_for(shape: str, label: str) -> str:
    if shape in {"mass", "cyst"}:
        return "localized_malignant_annotation" if label == "malignant" else "localized_mass_like"
    if shape == "calcification":
        return "calcification_region"
    if shape == "architectural":
        return "architectural_distortion_like"
    return "uncertain_annotation"


def default_qc_for(shape: str, large_lesion: bool = False, foreground_like: bool = False) -> str:
    if shape == "cyst":
        return "overinclusive_annotation"
    if large_lesion:
        return "diffuse_large"
    if foreground_like:
        return "foreground_like_or_contour"
    return "tight_or_reasonable"


def note(*parts: str) -> str:
    return "; ".join(part for part in parts if part)


# Patient-level or side-level morphology labels from the latest reviewer notes.
# scope format: patient_num -> list[(laterality_or_none, shape, base_qc, base_note)].
MORPHOLOGY: dict[str, list[tuple[str | None, str, str | None, str]]] = {
    # Benign.
    "217": [(None, "architectural", None, "")],
    "234": [(None, "architectural", None, "")],
    "239": [(None, "calcification", None, "")],
    "248": [("LEFT", "mass", None, ""), ("RIGHT", "architectural", None, "left/right breasts differ")],
    "249": [(None, "mass", None, "")],
    "281": [(None, "calcification", None, "")],
    "289": [(None, "architectural", None, "")],
    "303": [(None, "mass", None, "")],
    "311": [(None, "mass", None, "")],
    "318": [(None, "mass", None, "")],
    "326": [(None, "mass", None, "")],
    "330": [(None, "architectural", None, "")],
    "355": [(None, "calcification", None, "")],
    "363": [(None, "mass", None, "")],
    "373": [(None, "mass", None, "")],
    "386": [("LEFT", "mass", None, ""), ("RIGHT", "mass", None, "")],
    "388": [(None, "mass", None, "")],
    "415": [(None, "mass", None, "")],
    "417": [(None, "whole", "foreground_like_or_contour", "whole-breast contour-like")],
    "508": [("LEFT", "calcification", None, ""), ("RIGHT", "mass", None, "")],
    "1264": [(None, "calcification", None, "")],
    "1271": [(None, "mass", None, "")],
    "1298": [(None, "mass", None, "")],
    "1300": [(None, "calcification", None, "")],
    "1321": [(None, "mass", None, "")],
    "1648": [(None, "calcification", None, "uncertain calcification-like annotation")],
    "1685": [(None, "calcification", None, "uncertain calcification-like annotation")],
    "1708": [(None, "mass", None, "")],
    "1745": [(None, "mass", None, "")],
    "1756": [(None, "mass", None, "")],
    "1771": [(None, "mass", None, "")],
    "1776": [(None, "mass", None, "")],
    "1778": [(None, "mass", None, "")],
    "1798": [(None, "mass", None, "unclear visually")],
    "1799": [(None, "mass", None, "unclear visually")],
    "1801": [(None, "mass", None, "")],
    "1814": [(None, "mass", None, "")],
    "1831": [(None, "calcification", None, "")],
    "1832": [(None, "calcification", None, "")],
    "1836": [(None, "mass", None, "")],
    "1843": [(None, "mass", None, "")],
    "1854": [(None, "calcification", None, "")],
    "1858": [(None, "mass", None, "")],
    "1912": [(None, "mass", None, "")],
    "1913": [(None, "calcification", None, "")],
    "1920": [(None, "mass", None, "")],
    "1925": [(None, "calcification", None, "")],
    "1940": [(None, "mass", None, "")],
    "1945": [(None, "calcification", None, "")],
    "3095": [(None, "calcification", None, "")],
    "3133": [(None, "architectural", None, "")],
    "3151": [(None, "mass", None, "")],
    "3159": [(None, "cyst", "overinclusive_annotation", "small cyst-like annotation; mask likely overinclusive")],
    "3421": [(None, "calcification", None, "")],
    "3434": [(None, "calcification", None, "")],
    "3436": [(None, "calcification", None, "")],
    "3455": [
        ("LEFT", "cyst", "overinclusive_annotation", "small cyst-like annotation; mask likely overinclusive"),
        ("RIGHT", "calcification", None, ""),
    ],
    "4006": [(None, "mass", None, "")],
    "4014": [(None, "mass", None, "")],
    "4038": [(None, "mass", None, "")],
    "4044": [(None, "calcification", None, "")],
    "4055": [(None, "architectural", None, "")],
    "4066": [(None, "mass", None, "")],
    "4069": [(None, "mass", None, "")],
    "4080": [(None, "mass", None, "")],
    "4194": [(None, "calcification", None, "")],
    "4196": [(None, "calcification", None, "")],
    # Malignant cases explicitly reviewed by the user.
    "4": [(None, "mass", None, "")],
    "5": [(None, "whole", "diffuse_large", "large lesion/tissue region in both CC and MLO")],
    "18": [(None, "mass", None, "")],
    "19": [(None, "whole", "diffuse_large", "large lesion/tissue region in both CC and MLO")],
    "45": [(None, "calcification", None, "")],
    "55": [(None, "whole", "diffuse_large", "large lesion/tissue region in both CC and MLO")],
    "159": [(None, "whole", "diffuse_large", "large lesion/tissue region in both CC and MLO")],
    "189": [(None, "mass", None, "")],
    "481": [(None, "mass", None, "")],
    "1538": [(None, "architectural", None, "")],
    "1570": [(None, "mass", None, "")],
    "1573": [(None, "cyst", "overinclusive_annotation", "small cyst-like annotation; mask likely overinclusive")],
    "1772": [(None, "architectural", None, "")],
    "1851": [(None, "whole", "diffuse_large", "large lesion/tissue region")],
    "1897": [(None, "calcification", None, "")],
    "3002": [(None, "whole", "diffuse_large", "large lesion/tissue region")],
    "3019": [(None, "calcification", None, "")],
    "3024": [(None, "whole", "diffuse_large", "large lesion/tissue region")],
    "3109": [(None, "mass", None, "")],
    "4087": [(None, "mass", None, "")],
    "4113": [(None, "mass", None, "")],
}


# Row-specific QC problems from reviewer notes.
SPECIFIC_QC: dict[tuple[str, str, str], tuple[str, str]] = {
    ("1745", "RIGHT", "CC"): ("cropped_or_truncated", "RIGHT CC left border touching; mass truncated"),
    ("1912", "RIGHT", "MLO"): ("cropped_or_truncated", "RIGHT MLO border touching; mass truncated"),
    ("4006", "LEFT", "CC"): ("border_touching", "LEFT CC border touching; mass complete"),
    ("4006", "LEFT", "MLO"): ("cropped_or_truncated", "LEFT MLO border touching; mass truncated"),
    ("4194", "LEFT", "CC"): ("border_touching", "left-side border touching"),
    ("4194", "LEFT", "MLO"): ("border_touching", "left-side border touching"),
    ("4", "RIGHT", "CC"): ("cropped_or_truncated", "RIGHT CC right border touching; mass truncated"),
    ("18", "LEFT", "CC"): ("border_touching", "LEFT CC border touching; mass tissue complete"),
    ("189", "RIGHT", "CC"): ("cropped_or_truncated", "border touching; mass truncated"),
    ("189", "RIGHT", "MLO"): ("cropped_or_truncated", "border touching; mass truncated"),
    ("481", "LEFT", "CC"): ("border_touching", "LEFT CC left border touching"),
    ("1570", "RIGHT", "CC"): ("border_touching", "RIGHT CC left border touching"),
    ("1851", "LEFT", "MLO"): ("diffuse_large", "LEFT MLO right border touching"),
    ("3109", "RIGHT", "CC"): ("cropped_or_truncated", "right border touching; mass truncated"),
    ("3109", "RIGHT", "MLO"): ("cropped_or_truncated", "right border touching; mass truncated"),
    ("4087", "RIGHT", "MLO"): ("border_touching", "RIGHT MLO right border touching; mass complete"),
    ("4113", "LEFT", "CC"): ("cropped_or_truncated", "LEFT CC left border touching; mass truncated"),
}


def morphology_for(row: dict[str, str]) -> tuple[str, str | None, str]:
    pid = norm_patient(row["patient_num"])
    laterality = row.get("laterality", "").upper()
    entries = MORPHOLOGY.get(pid, [])
    for scope_laterality, shape, base_qc, base_note in entries:
        if scope_laterality is None or scope_laterality == laterality:
            return shape, base_qc, base_note
    if row.get("label") == "malignant":
        return "mass", None, ""
    return "uncertain", None, "missing reviewer morphology rule"


def seed_row(row: dict[str, str]) -> None:
    shape, base_qc, base_note = morphology_for(row)
    row["manual_mask_type"] = manual_type_for(shape, row.get("label", ""))

    large_lesion = base_qc == "diffuse_large"
    foreground_like = base_qc == "foreground_like_or_contour"
    row["mask_qc_type"] = base_qc or default_qc_for(shape, large_lesion, foreground_like)

    specific = SPECIFIC_QC.get(key(row))
    row_note = base_note
    if specific:
        row["mask_qc_type"], specific_note = specific
        row_note = note(row_note, specific_note)
    row["notes"] = row_note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows, fieldnames = read_csv(args.input)
    for row in rows:
        seed_row(row)
        row["bright_dot_or_marker"] = row.get("bright_dot_or_marker", "")
        row["usable_for_overlap"] = row.get("usable_for_overlap", "")

    write_csv(args.output, rows, fieldnames)
    print(f"Seeded {len(rows)} rows in {args.output}")


if __name__ == "__main__":
    main()
