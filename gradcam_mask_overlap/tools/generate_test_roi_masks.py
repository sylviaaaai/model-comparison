"""Generate ROI-space lesion masks for the Grad-CAM overlap experiment.

The script reproduces the original ROI crop rule used for model inputs:

1. Threshold the original DDSM `_Mask.jpg`.
2. Use the nonzero mask contour to compute the bounding box.
3. Make a square crop around the box center with padding=0.1.
4. Crop the filled lesion mask with the same coordinates.
5. Resize the mask to 224 x 224 with nearest-neighbor interpolation.

It updates `metadata/test_roi_mask_manifest.csv` in place and writes QC overlay
PNGs next to the generated ROI masks.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
MASK_MANIFEST = METADATA_DIR / "test_roi_mask_manifest.csv"
RUN_LOG = METADATA_DIR / "runs.jsonl"
ROI_SIZE = 224
EXTRA_FIELDNAMES = [
    "original_image_width",
    "original_image_height",
    "original_mask_width",
    "original_mask_height",
    "roi_width",
    "roi_height",
    "mask_empty",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_gray(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def threshold_mask(mask_image: Image.Image, threshold: int) -> np.ndarray:
    arr = np.array(mask_image)
    return arr > threshold


def bounding_rect(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)
    return x, y, w, h


def square_crop_from_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    max_side = max(w, h)
    x_center = x + w // 2
    y_center = y + h // 2
    half_side = int(max_side / 2 * (1 + padding))

    x1 = max(0, x_center - half_side)
    y1 = max(0, y_center - half_side)
    x2 = min(image_width, x_center + half_side)
    y2 = min(image_height, y_center + half_side)
    return int(x1), int(y1), int(x2), int(y2)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = image.resize((size, size), Image.Resampling.NEAREST)
    return np.array(resized) > 0


def crop_mask_with_padding(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    crop_height = y2 - y1
    crop_width = x2 - x1
    cropped = np.zeros((crop_height, crop_width), dtype=bool)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(mask.shape[1], x2)
    src_y2 = min(mask.shape[0], y2)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return cropped

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    cropped[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return cropped


def mask_contour(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_qc_overlay(roi_image_path: str, roi_mask: np.ndarray, path: Path) -> None:
    roi = load_gray(roi_image_path).resize((ROI_SIZE, ROI_SIZE), Image.Resampling.BILINEAR)
    base = np.array(roi, dtype=np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    fill = roi_mask
    contour = mask_contour(roi_mask)
    rgb[fill, 0] = 0.65 * rgb[fill, 0] + 0.35 * 255
    rgb[fill, 1] = 0.65 * rgb[fill, 1]
    rgb[fill, 2] = 0.65 * rgb[fill, 2]
    rgb[contour] = np.array([0, 255, 255], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def process_row(row: dict[str, str]) -> dict[str, object]:
    updated: dict[str, object] = dict(row)

    original_image_path = Path(row["original_image_path"])
    original_mask_path = Path(row["original_mask_path"])
    roi_image_path = Path(row["roi_image_path"])
    roi_mask_path = Path(row["roi_mask_path"])
    qc_overlay_path = Path(row["qc_overlay_path"])
    threshold = int(float(row.get("mask_threshold") or 127))
    padding = float(row.get("padding") or 0.1)

    if not original_image_path.exists():
        updated.update({"qc_status": "failed", "qc_note": "missing_original_image"})
        return updated
    if not original_mask_path.exists():
        updated.update({"has_mask": False, "qc_status": "failed", "qc_note": "missing_original_mask"})
        return updated
    if not roi_image_path.exists():
        updated.update({"qc_status": "failed", "qc_note": "missing_roi_image"})
        return updated

    original_image = load_gray(str(original_image_path))
    original_mask_image = load_gray(str(original_mask_path))

    image_width, image_height = original_image.size
    mask_width, mask_height = original_mask_image.size
    size_note = ""
    if (image_width, image_height) != (mask_width, mask_height):
        size_note = f"size_mismatch_tolerated:image={image_width}x{image_height};mask={mask_width}x{mask_height}"

    contour_binary = threshold_mask(original_mask_image, threshold)
    bbox = bounding_rect(contour_binary)
    if bbox is None:
        updated.update({"mask_empty": True, "qc_status": "failed", "qc_note": "empty_thresholded_mask"})
        return updated

    filled_binary = ndimage.binary_fill_holes(contour_binary)
    fill_added_pixels = int(filled_binary.sum() - contour_binary.sum())
    fill_note = "filled_contour"
    if fill_added_pixels <= 0:
        fill_note = "contour_not_filled_check_mask_closure"

    x1, y1, x2, y2 = square_crop_from_bbox(bbox, image_width, image_height, padding)
    if x2 <= x1 or y2 <= y1:
        updated.update({"qc_status": "failed", "qc_note": "invalid_crop"})
        return updated

    cropped_mask = crop_mask_with_padding(filled_binary, x1, y1, x2, y2)
    roi_mask = resize_mask(cropped_mask, ROI_SIZE)
    save_mask(roi_mask, roi_mask_path)
    roi_image_width, roi_image_height = load_gray(str(roi_image_path)).size
    save_qc_overlay(str(roi_image_path), roi_mask, qc_overlay_path)

    mask_area_original = int(filled_binary.sum())
    mask_area_roi = int(roi_mask.sum())
    mask_area_roi_fraction = mask_area_roi / float(ROI_SIZE * ROI_SIZE)
    x, y, w, h = bbox

    qc_note = fill_note if not size_note else f"{fill_note};{size_note}"

    updated.update(
        {
            "has_mask": True,
            "mask_empty": False,
            "original_image_width": image_width,
            "original_image_height": image_height,
            "original_mask_width": mask_width,
            "original_mask_height": mask_height,
            "roi_width": roi_image_width,
            "roi_height": roi_image_height,
            "lesion_bbox_x": x,
            "lesion_bbox_y": y,
            "lesion_bbox_w": w,
            "lesion_bbox_h": h,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "crop_width": x2 - x1,
            "crop_height": y2 - y1,
            "mask_area_original_px": mask_area_original,
            "mask_area_roi_px": mask_area_roi,
            "mask_area_roi_fraction": f"{mask_area_roi_fraction:.8f}",
            "mask_fill_method": f"binary_fill_holes_after_threshold_{threshold}",
            "qc_status": "generated_pending_review",
            "qc_note": qc_note,
        }
    )
    return updated


def append_run_log(summary: dict[str, object]) -> None:
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MASK_MANIFEST)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows for a smoke test.")
    args = parser.parse_args()

    rows, fieldnames = read_csv(args.manifest)
    for fieldname in EXTRA_FIELDNAMES:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)
    if args.limit:
        process_rows = rows[: args.limit]
        untouched_rows = rows[args.limit :]
    else:
        process_rows = rows
        untouched_rows = []

    updated_rows = [process_row(row) for row in process_rows] + untouched_rows
    write_csv(args.manifest, updated_rows, fieldnames)

    status_counts: dict[str, int] = {}
    for row in updated_rows:
        status = str(row.get("qc_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "run_id": "test_roi_mask_generation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_type": "test_roi_mask_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "processed_rows": len(process_rows),
        "total_rows": len(rows),
        "roi_size": ROI_SIZE,
        "status_counts": status_counts,
    }
    append_run_log(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
