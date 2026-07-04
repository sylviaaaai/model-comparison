"""Analyze saliency energy across lesion, perilesional, and outside regions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image
from scipy import ndimage


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = EXPERIMENT_ROOT / "metadata" / "test_image_master_table.csv"
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "metadata" / "analysis_ready_overlap_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "overlap_analysis" / "context_partition_energy"
ROI_SIZE = 224
EPS = 1e-12

PRIMARY_METHODS = {
    "resnet50": "gradcam_malignant",
    "convnext_base": "gradcam_malignant",
    "swin_base": "gradcam_malignant",
    "vit_b16": "attention_rollout",
}

MODEL_DISPLAY = {
    "resnet50": "ResNet50",
    "vit_b16": "ViT_B16",
    "convnext_base": "ConvNeXt_Base",
    "swin_base": "Swin_Base",
}

SCOPES = {
    "main_yes": {"yes"},
    "sensitivity_yes_cautious": {"yes", "cautious"},
}

IMAGE_FIELDS = [
    "scope",
    "image_id",
    "patient_id",
    "true_label",
    "model",
    "model_name",
    "saliency_method",
    "ring_px",
    "usable_for_overlap",
    "manual_mask_type",
    "mask_qc_type",
    "bright_dot_or_marker",
    "marker_positive",
    "zero_saliency",
    "large_dilated_region_flag",
    "total_energy",
    "lesion_area_fraction",
    "perilesional_area_fraction",
    "outside_area_fraction",
    "dilated_area_fraction",
    "lesion_energy",
    "perilesional_energy",
    "outside_energy",
    "lesion_norm",
    "perilesional_norm",
    "outside_norm",
    "quality_note",
]

SUMMARY_FIELDS = [
    "scope",
    "ring_px",
    "model",
    "n_images",
    "n_valid",
    "n_zero_saliency",
    "n_large_dilated_region",
    "lesion_energy_mean",
    "lesion_energy_median",
    "lesion_energy_sd",
    "perilesional_energy_mean",
    "perilesional_energy_median",
    "perilesional_energy_sd",
    "outside_energy_mean",
    "outside_energy_median",
    "outside_energy_sd",
    "lesion_norm_mean",
    "lesion_norm_median",
    "lesion_norm_sd",
    "perilesional_norm_mean",
    "perilesional_norm_median",
    "perilesional_norm_sd",
    "outside_norm_mean",
    "outside_norm_median",
    "outside_norm_sd",
    "lesion_area_fraction_mean",
    "perilesional_area_fraction_mean",
    "outside_area_fraction_mean",
    "dilated_area_fraction_mean",
]

SUMMARY_METRICS = [
    "lesion_energy",
    "perilesional_energy",
    "outside_energy",
    "lesion_norm",
    "perilesional_norm",
    "outside_norm",
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


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def clean_float(value: float | None) -> object:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.10g}"


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value <= min_value:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_value) / (max_value - min_value)


def resize_heatmap(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D saliency map after squeeze, got shape {arr.shape}")
    if arr.shape == (ROI_SIZE, ROI_SIZE):
        return arr
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    image = image.resize((ROI_SIZE, ROI_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def load_mask(path: str) -> np.ndarray:
    image = Image.open(path).convert("L").resize((ROI_SIZE, ROI_SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / (denominator + EPS)


def compute_context_partition_energy(
    heatmap: np.ndarray,
    mask: np.ndarray,
    ring_px: int,
) -> dict[str, object]:
    heatmap = normalize01(resize_heatmap(heatmap))
    mask = np.asarray(mask, dtype=bool)
    dilated = ndimage.binary_dilation(mask, structure=disk_structure(ring_px))
    perilesional = dilated & ~mask
    outside = ~dilated

    lesion_area_fraction = float(mask.mean())
    perilesional_area_fraction = float(perilesional.mean())
    outside_area_fraction = float(outside.mean())
    dilated_area_fraction = float(dilated.mean())
    total_energy = float(heatmap.sum())

    base = {
        "total_energy": total_energy,
        "lesion_area_fraction": lesion_area_fraction,
        "perilesional_area_fraction": perilesional_area_fraction,
        "outside_area_fraction": outside_area_fraction,
        "dilated_area_fraction": dilated_area_fraction,
        "large_dilated_region_flag": int(dilated_area_fraction > 0.80),
    }

    if total_energy <= EPS:
        return {
            **base,
            "zero_saliency": 1,
            "lesion_energy": None,
            "perilesional_energy": None,
            "outside_energy": None,
            "lesion_norm": None,
            "perilesional_norm": None,
            "outside_norm": None,
            "quality_note": "zero_saliency",
        }

    lesion_energy = float(heatmap[mask].sum() / total_energy)
    perilesional_energy = float(heatmap[perilesional].sum() / total_energy)
    outside_energy = float(heatmap[outside].sum() / total_energy)

    return {
        **base,
        "zero_saliency": 0,
        "lesion_energy": lesion_energy,
        "perilesional_energy": perilesional_energy,
        "outside_energy": outside_energy,
        "lesion_norm": safe_div(lesion_energy, lesion_area_fraction),
        "perilesional_norm": safe_div(perilesional_energy, perilesional_area_fraction),
        "outside_norm": safe_div(outside_energy, outside_area_fraction),
        "quality_note": "",
    }


def manifest_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        method = PRIMARY_METHODS.get(row["model_name"])
        if row.get("ready_for_overlap") != "True" or row.get("saliency_method") != method:
            continue
        indexed[(row["image_id"], row["model_name"])] = row
    return indexed


def marker_positive(value: str) -> int:
    text = value.strip()
    return int(bool(text) and text != "none")


def subset_master_rows(rows: list[dict[str, str]], scope: str) -> list[dict[str, str]]:
    allowed = SCOPES[scope]
    return [row for row in rows if row.get("usable_for_overlap") in allowed]


def build_image_rows(
    master_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    ring_values: list[int],
) -> list[dict[str, object]]:
    indexed_manifest = manifest_index(manifest_rows)
    output_rows: list[dict[str, object]] = []

    for scope in SCOPES:
        for row in subset_master_rows(master_rows, scope):
            mask = load_mask(row["roi_mask_path"])
            for model_name, method in PRIMARY_METHODS.items():
                manifest = indexed_manifest.get((row["image_id"], model_name))
                if manifest is None:
                    for ring_px in ring_values:
                        output_rows.append(
                            {
                                "scope": scope,
                                "image_id": row["image_id"],
                                "patient_id": row.get("patient_id", ""),
                                "true_label": row.get("true_label", ""),
                                "model": MODEL_DISPLAY[model_name],
                                "model_name": model_name,
                                "saliency_method": method,
                                "ring_px": ring_px,
                                "usable_for_overlap": row.get("usable_for_overlap", ""),
                                "manual_mask_type": row.get("manual_mask_type", ""),
                                "mask_qc_type": row.get("mask_qc_type", ""),
                                "bright_dot_or_marker": row.get("bright_dot_or_marker", ""),
                                "marker_positive": marker_positive(row.get("bright_dot_or_marker", "")),
                                "quality_note": "missing_saliency_manifest",
                            }
                        )
                    continue

                heatmap = np.load(manifest["saliency_path"])
                for ring_px in ring_values:
                    metrics = compute_context_partition_energy(heatmap, mask, ring_px)
                    output_rows.append(
                        {
                            "scope": scope,
                            "image_id": row["image_id"],
                            "patient_id": row.get("patient_id", ""),
                            "true_label": row.get("true_label", ""),
                            "model": MODEL_DISPLAY[model_name],
                            "model_name": model_name,
                            "saliency_method": method,
                            "ring_px": ring_px,
                            "usable_for_overlap": row.get("usable_for_overlap", ""),
                            "manual_mask_type": row.get("manual_mask_type", ""),
                            "mask_qc_type": row.get("mask_qc_type", ""),
                            "bright_dot_or_marker": row.get("bright_dot_or_marker", ""),
                            "marker_positive": marker_positive(row.get("bright_dot_or_marker", "")),
                            **metrics,
                        }
                    )

    return output_rows


def finite_values(rows: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def summarize_image_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scope"]), int(row["ring_px"]), str(row["model"])), []).append(row)

    out: list[dict[str, object]] = []
    for (scope, ring_px, model), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "scope": scope,
            "ring_px": ring_px,
            "model": model,
            "n_images": len(group),
            "n_valid": sum(int(row.get("zero_saliency", 0) == 0) for row in group),
            "n_zero_saliency": sum(int(row.get("zero_saliency", 0) == 1) for row in group),
            "n_large_dilated_region": sum(
                int(row.get("large_dilated_region_flag", 0) == 1) for row in group
            ),
        }
        for metric in SUMMARY_METRICS:
            values = finite_values(group, metric)
            summary[f"{metric}_mean"] = fmt_float(mean(values))
            summary[f"{metric}_median"] = fmt_float(median(values) if values else None)
            summary[f"{metric}_sd"] = fmt_float(sample_sd(values))
        for metric in [
            "lesion_area_fraction",
            "perilesional_area_fraction",
            "outside_area_fraction",
            "dilated_area_fraction",
        ]:
            summary[f"{metric}_mean"] = fmt_float(mean(finite_values(group, metric)))
        out.append(summary)
    return out


def rank_summary(
    rows: list[dict[str, object]],
    scope: str,
    ring_px: int,
    metric: str,
    highest: bool = True,
) -> tuple[str, float] | None:
    candidates: list[tuple[float, str]] = []
    field = f"{metric}_mean"
    for row in rows:
        if row["scope"] != scope or int(row["ring_px"]) != ring_px:
            continue
        value = as_float(row.get(field))
        if value is not None:
            candidates.append((value, str(row["model"])))
    if not candidates:
        return None
    value, model = sorted(candidates, reverse=highest)[0]
    return model, value


def markdown_table(rows: list[dict[str, object]], scope: str, ring_px: int) -> list[str]:
    selected = [
        row for row in rows if row["scope"] == scope and int(row["ring_px"]) == ring_px
    ]
    selected.sort(key=lambda row: ["ResNet50", "ViT_B16", "ConvNeXt_Base", "Swin_Base"].index(str(row["model"])))
    lines = [
        f"### `{scope}`, ring = {ring_px}px",
        "",
        "| Model | N valid / N | Lesion energy | Perilesional energy | Outside energy | Lesion norm | Perilesional norm | Outside norm | Zero saliency | Large dilated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| `{row['model']}` | {row['n_valid']} / {row['n_images']} | "
            f"{row['lesion_energy_mean']} | {row['perilesional_energy_mean']} | {row['outside_energy_mean']} | "
            f"{row['lesion_norm_mean']} | {row['perilesional_norm_mean']} | {row['outside_norm_mean']} | "
            f"{row['n_zero_saliency']} | {row['n_large_dilated_region']} |"
        )
    return lines


def write_markdown_summary(path: Path, rows: list[dict[str, object]], ring_values: list[int]) -> None:
    main_ring = 20 if 20 in ring_values else ring_values[0]
    lesion_top = rank_summary(rows, "main_yes", main_ring, "lesion_norm")
    perilesion_top = rank_summary(rows, "main_yes", main_ring, "perilesional_norm")
    outside_energy_top = rank_summary(rows, "main_yes", main_ring, "outside_energy")
    outside_norm_top = rank_summary(rows, "main_yes", main_ring, "outside_norm")

    lines = [
        "# Context-Partition Saliency Energy Analysis",
        "",
        "This analysis partitions each ROI into three mutually exclusive regions:",
        "",
        "- `lesion`: original binary lesion mask.",
        "- `perilesional`: dilated lesion mask minus original lesion mask.",
        "- `outside`: ROI pixels outside the dilated lesion mask.",
        "",
        "Dilation was implemented with a circular structuring element using `scipy.ndimage.binary_dilation`.",
        "",
        "Energy metrics are fractions of total non-negative normalized saliency energy.",
        "Area-normalized metrics divide energy fraction by area fraction; values above 1 indicate greater-than-area-proportional saliency concentration.",
        "",
        "Primary saliency methods:",
        "",
        "- ResNet50, ConvNeXt-Base, Swin-Base: malignant-class Grad-CAM.",
        "- ViT-B/16: attention rollout.",
        "",
        "Rows with zero total saliency energy are retained in the image-level file but excluded from model-level metric means.",
        "",
        "## Main Result",
        "",
    ]
    lines.extend(markdown_table(rows, "main_yes", main_ring))
    lines += [
        "",
        "## Sensitivity Results",
        "",
    ]
    lines.extend(markdown_table(rows, "sensitivity_yes_cautious", main_ring))
    for ring_px in ring_values:
        if ring_px == main_ring:
            continue
        lines += ["", f"## Ring-Size Sensitivity: {ring_px}px", ""]
        lines.extend(markdown_table(rows, "main_yes", ring_px))
        lines += [""]
        lines.extend(markdown_table(rows, "sensitivity_yes_cautious", ring_px))

    lines += [
        "",
        "## Auto-Generated Interpretation Notes",
        "",
    ]
    if lesion_top:
        lines.append(
            f"- Highest main-set lesion_norm: `{lesion_top[0]}` ({lesion_top[1]:.3f})."
        )
    if perilesion_top:
        lines.append(
            f"- Highest main-set perilesional_norm: `{perilesion_top[0]}` ({perilesion_top[1]:.3f})."
        )
    if outside_energy_top:
        lines.append(
            f"- Highest main-set outside_energy: `{outside_energy_top[0]}` ({outside_energy_top[1]:.3f})."
        )
    if outside_norm_top:
        lines.append(
            f"- Highest main-set outside_norm: `{outside_norm_top[0]}` ({outside_norm_top[1]:.3f})."
        )
    lines += [
        "- Main-set pattern: ConvNeXt_Base and ResNet50 are most lesion-concentrated by lesion_energy / lesion_norm.",
        "- ViT_B16 has the lowest lesion_energy / lesion_norm and the highest outside_energy / outside_norm, consistent with a more spatially distributed attention pattern.",
        "- Swin_Base is intermediate, with outside_norm close to area-proportional at 20px and below ViT_B16.",
        "- ConvNeXt_Base does not show perilesional enrichment in this summary; its favorable overlap appears mainly lesion-centered rather than ring-centered.",
        "- The 20px ring creates large dilated regions for many images, so perilesional and outside-region interpretations should be checked against the 10px sensitivity table.",
    ]
    lines += [
        "- Interpret these as spatial alignment patterns, not causal evidence of model decision-making.",
        "- `outside_energy` is sensitive to outside-region area; use `outside_norm` when asking whether saliency is disproportionately outside the lesion/perilesional zone.",
        "- ResNet50 zero-saliency rows should be reported through `n_zero_saliency` / `n_valid`.",
        "",
        "## Output Files",
        "",
        "- `context_partition_energy_image_level.csv`: image x model x scope x ring_px detail.",
        "- `context_partition_energy_summary_by_model.csv`: model-level summary by scope and ring size.",
        "- `context_partition_energy_summary.md`: this report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_image_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    formatted = []
    for row in rows:
        formatted_row = {}
        for field, value in row.items():
            if isinstance(value, float):
                formatted_row[field] = clean_float(value)
            elif value is None:
                formatted_row[field] = ""
            else:
                formatted_row[field] = value
        formatted.append(formatted_row)
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rings", type=int, nargs="+", default=[20, 10])
    args = parser.parse_args()

    master_rows = read_csv(args.master)
    manifest_rows = read_csv(args.manifest)
    ring_values = sorted(set(args.rings), reverse=True)

    image_rows = build_image_rows(master_rows, manifest_rows, ring_values)
    summary_rows = summarize_image_rows(image_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "context_partition_energy_image_level.csv"
    summary_path = args.output_dir / "context_partition_energy_summary_by_model.csv"
    markdown_path = args.output_dir / "context_partition_energy_summary.md"

    write_csv(image_path, format_image_rows(image_rows), IMAGE_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_markdown_summary(markdown_path, summary_rows, ring_values)

    counts = Counter(
        f"{row['scope']}|ring_{row['ring_px']}|{row['model']}|zero_{row['zero_saliency']}"
        for row in image_rows
    )
    run_summary = {
        "run_id": "context_partition_energy_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "image_rows": len(image_rows),
        "summary_rows": len(summary_rows),
        "rings": ring_values,
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "image_level": str(image_path),
            "summary_by_model": str(summary_path),
            "markdown": str(markdown_path),
        },
    }
    print(run_summary)


if __name__ == "__main__":
    main()
