"""Export raw ViT-B/16 attention rollout maps for all final test images.

Outputs 224x224 normalized attention maps as `.npy` plus uint8 PNG previews and
updates `gradcam_mask_overlap/metadata/saliency_manifest.csv` for ViT.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import numpy as np
import torch
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = EXPERIMENT_ROOT / "metadata"
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "VIT/results/vit_b16_torch_final_all_blocks_dropout04/final/all_blocks/final_model.pt"
DEFAULT_ATTENTION_SCRIPT = WORKSPACE_ROOT / "VIT/tools/attention_rollout_vit_b16_torch.py"
DEFAULT_PREDICTIONS = METADATA_DIR / "model_predictions.csv"
DEFAULT_SALIENCY_MANIFEST = METADATA_DIR / "saliency_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "data/saliency_maps/vit_b16/attention_rollout_predclass"
RUN_LOG = METADATA_DIR / "runs.jsonl"
IMG_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--attention-script", type=Path, default=DEFAULT_ATTENTION_SCRIPT)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--saliency-manifest", type=Path, default=DEFAULT_SALIENCY_MANIFEST)
    parser.add_argument("--run-log", type=Path, default=RUN_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--discard-ratio", type=float, default=0.0)
    parser.add_argument("--method", choices=["rollout", "last_layer_cls"], default="rollout")
    parser.add_argument("--percentile", type=float, default=99.0)
    return parser.parse_args()


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


def normalize_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if len(raw_path) >= 3 and raw_path[1:3] == ":\\":
        win_path = PureWindowsPath(raw_path)
        drive = win_path.drive.rstrip(":").lower()
        mapped = Path("/mnt") / drive / Path(*win_path.parts[1:])
        if mapped.exists():
            return mapped
    mapped = Path(raw_path.replace("\\", "/"))
    if mapped.exists():
        return mapped
    return path


def image_id_from_path(path: str) -> str:
    return normalize_path(path).stem


def load_attention_module(path: Path):
    spec = importlib.util.spec_from_file_location("vit_attention_rollout_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ViT attention script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_heatmap(heatmap: np.ndarray, output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    npy_path = Path(str(output_base) + ".npy")
    png_path = Path(str(output_base) + ".png")
    np.save(npy_path, heatmap.astype(np.float32))
    Image.fromarray((heatmap * 255).round().astype(np.uint8), mode="L").save(png_path)
    return npy_path, png_path


def append_run_log(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def saliency_id_for_prediction(prediction_id: str) -> str:
    return f"{prediction_id}__attention_rollout__predclass"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    predictions, _ = read_csv(args.predictions_csv)
    vit_predictions = [row for row in predictions if row["model_name"] == "vit_b16"]
    if args.limit:
        vit_predictions = vit_predictions[: args.limit]

    image_paths_by_id: dict[str, Path] = {}
    source_prediction_csvs = sorted({Path(row["source_predictions_csv"]) for row in vit_predictions})
    for source_csv in source_prediction_csvs:
        source_rows, _ = read_csv(source_csv)
        for source_row in source_rows:
            image_paths_by_id[image_id_from_path(source_row["path"])] = normalize_path(source_row["path"])

    attention_module = load_attention_module(args.attention_script)
    model = attention_module.build_model(args.model_path, device)
    transform = attention_module.make_transform()
    print(f"Device: {device}")
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print(f"ViT attention script: {args.attention_script}")
    print(f"method={args.method} discard_ratio={args.discard_ratio} percentile={args.percentile}")

    manifest_rows, manifest_fields = read_csv(args.saliency_manifest)
    for extra_field in ["recomputed_cancer_probability", "source_cancer_probability"]:
        if extra_field not in manifest_fields:
            manifest_fields.append(extra_field)
    manifest_by_prediction = {
        row["prediction_id"]: row
        for row in manifest_rows
        if row.get("model_name") == "vit_b16" and row.get("saliency_method") == "attention_rollout"
    }
    existing_saliency_ids = {row["saliency_id"] for row in manifest_rows}

    exported = 0
    skipped_existing = 0
    failed = 0
    new_manifest_rows: list[dict[str, object]] = []

    for index, row in enumerate(vit_predictions, start=1):
        image_id = row["image_id"]
        prediction_id = row["prediction_id"]
        image_path = image_paths_by_id.get(image_id)
        if image_path is None or not image_path.exists():
            print(f"[{index}/{len(vit_predictions)}] missing image for {image_id}")
            failed += 1
            continue

        source_probability = float(row["cancer_probability"])
        output_base = args.output_dir / image_id[:4] / image_id
        npy_path = Path(str(output_base) + ".npy")
        png_path = Path(str(output_base) + ".png")
        probability: float | None = None
        if args.skip_existing and npy_path.exists() and png_path.exists():
            skipped_existing += 1
        else:
            _original, image_tensor = attention_module.load_image(str(image_path), transform, device)
            probability, heatmap = attention_module.collect_attention_rollout(
                model,
                image_tensor,
                discard_ratio=args.discard_ratio,
                method=args.method,
                percentile=args.percentile,
            )
            npy_path, png_path = save_heatmap(heatmap, output_base)
            exported += 1
            if exported % 25 == 0 or exported == 1:
                print(f"Exported {exported}/{len(vit_predictions)}: {image_id}")

        saliency_id = saliency_id_for_prediction(prediction_id)
        manifest_row = manifest_by_prediction.get(prediction_id)
        if manifest_row is None:
            manifest_row = {
                "saliency_id": saliency_id,
                "image_id": image_id,
                "prediction_id": prediction_id,
                "model_name": "vit_b16",
                "architecture": row.get("architecture", "Transformer"),
                "saliency_method": "attention_rollout",
                "target_class_for_cam": "predicted_label",
                "target_class_value": row["predicted_label"],
                "source_summary_csv": "",
            }
            if saliency_id not in existing_saliency_ids:
                new_manifest_rows.append(manifest_row)
                existing_saliency_ids.add(saliency_id)
        manifest_row.update(
            {
                "saliency_id": saliency_id,
                "image_id": image_id,
                "prediction_id": prediction_id,
                "model_name": "vit_b16",
                "architecture": row.get("architecture", "Transformer"),
                "saliency_method": "attention_rollout",
                "target_class_for_cam": "predicted_label",
                "target_class_value": row["predicted_label"],
                "saliency_path": str(npy_path),
                "saliency_width": IMG_SIZE,
                "saliency_height": IMG_SIZE,
                "saliency_normalization": "minmax_0_1_attention_rollout_resized_224",
                "status": "raw_map_available",
                "source_cancer_probability": f"{source_probability:.8f}",
            }
        )
        if probability is not None:
            manifest_row["recomputed_cancer_probability"] = f"{probability:.8f}"

    if new_manifest_rows:
        manifest_rows.extend(new_manifest_rows)
    write_csv(args.saliency_manifest, manifest_rows, manifest_fields)

    summary = {
        "run_id": "vit_raw_attention_rollout_export_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_type": "vit_raw_attention_rollout_export",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path),
        "attention_script": str(args.attention_script),
        "output_dir": str(args.output_dir),
        "method": args.method,
        "discard_ratio": args.discard_ratio,
        "percentile": args.percentile,
        "processed_predictions": len(vit_predictions),
        "exported": exported,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }
    append_run_log(args.run_log, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
