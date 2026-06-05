#!/usr/bin/env python3
"""Generate attention rollout visualizations for the final PyTorch ViT-B/16 model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("This script requires timm. Install with: pip install timm") from exc


IMG_SIZE = 224
MODEL_NAME = "vit_base_patch16_224"
DEFAULT_EXPERIMENT_DIR = Path("VIT/results/vit_b16_torch_final_all_blocks_dropout04")
DEFAULT_MODEL_PATH = DEFAULT_EXPERIMENT_DIR / "final/all_blocks/final_model.pt"
DEFAULT_SPLIT_CSV = Path("results/splits/final_80_10_10/final_test_images.csv")
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "attention_rollout"
CLASS_NAMES = {0: "Benign", 1: "Cancer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ViT-B/16 attention rollout panels for final test images."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-group", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--method",
        choices=["rollout", "last_layer_cls"],
        default="rollout",
        help="Attention visualization method.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for the attention heatmap.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Upper percentile used for heatmap normalization. Use 100 to disable clipping.",
    )
    parser.add_argument(
        "--discard-ratio",
        type=float,
        default=0.9,
        help="Optional fraction of weakest patch attentions to discard per layer.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize_image_path(path: str) -> str:
    normalized = path.strip()
    if os.name != "nt" and len(normalized) >= 3 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    if os.name != "nt":
        return normalized.replace("\\", "/")
    return normalized


def read_split(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["path"] = normalize_image_path(row["path"])
            row["label"] = int(row["label"])
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    return records


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    dropout_rate = float(config.get("dropout_rate", 0.4))
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1, drop_rate=dropout_rate)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_attention_rollout(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    discard_ratio: float = 0.0,
    method: str = "rollout",
    percentile: float = 99.0,
) -> tuple[float, np.ndarray]:
    attention_maps: list[torch.Tensor] = []
    handles = []

    def make_hook(attn_module):
        def hook(_module, inputs, _output):
            x = inputs[0]
            batch_size, tokens, channels = x.shape
            qkv = attn_module.qkv(x)
            qkv = qkv.reshape(
                batch_size,
                tokens,
                3,
                attn_module.num_heads,
                channels // attn_module.num_heads,
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key = qkv[0], qkv[1]
            attention = (query @ key.transpose(-2, -1)) * attn_module.scale
            attention = attention.softmax(dim=-1)
            attention_maps.append(attention.detach())

        return hook

    for block in model.blocks:
        handles.append(block.attn.register_forward_hook(make_hook(block.attn)))

    try:
        with torch.no_grad():
            logits = model(image_tensor).reshape(-1)
            probability = float(torch.sigmoid(logits)[0].detach().cpu())
    finally:
        for handle in handles:
            handle.remove()

    if not attention_maps:
        raise ValueError("No attention maps were captured from the ViT blocks.")

    if method == "last_layer_cls":
        mask = attention_maps[-1].mean(dim=1)[0, 0, 1:]
    else:
        num_tokens = attention_maps[0].shape[-1]
        rollout = torch.eye(num_tokens, device=image_tensor.device).unsqueeze(0)
        for attention in attention_maps:
            attention = attention.mean(dim=1)
            if discard_ratio > 0:
                flat = attention[:, 1:, 1:].reshape(attention.shape[0], -1)
                discard_count = int(flat.shape[-1] * discard_ratio)
                if discard_count > 0:
                    indices = flat.topk(discard_count, dim=-1, largest=False).indices
                    flat.scatter_(1, indices, 0)
            attention = attention + torch.eye(num_tokens, device=attention.device).unsqueeze(0)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            rollout = attention @ rollout
        mask = rollout[0, 0, 1:]

    grid_size = int(np.sqrt(mask.numel()))
    mask = mask.reshape(1, 1, grid_size, grid_size)
    mask = F.interpolate(mask, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    mask = mask.squeeze().detach().cpu().numpy()
    mask = normalize_heatmap(mask, percentile)
    return probability, mask


def normalize_heatmap(mask: np.ndarray, percentile: float) -> np.ndarray:
    mask = mask.astype(np.float32)
    low = float(np.percentile(mask, 1.0))
    high = float(np.percentile(mask, percentile)) if percentile < 100 else float(mask.max())
    if high <= low:
        low = float(mask.min())
        high = float(mask.max())
    mask = np.clip(mask, low, high)
    mask = mask - mask.min()
    return mask / (mask.max() + 1e-8)


def load_image(path: str, transform, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    original = np.asarray(image).astype(np.uint8)
    tensor = transform(image).unsqueeze(0).to(device)
    return original, tensor


def group_name(label: int, predicted: int) -> str:
    if label == 1 and predicted == 1:
        return "correct_cancer"
    if label == 0 and predicted == 0:
        return "correct_benign"
    if label == 0 and predicted == 1:
        return "false_positive"
    return "false_negative"


def save_attention_panel(
    original: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    cmap_name: str,
    alpha: float = 0.40,
) -> None:
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(heatmap)[..., :3]
    overlay = np.clip((1 - alpha) * (original / 255.0) + alpha * colored, 0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=160)
    panels = [
        ("Original", original / 255.0),
        ("Attention", heatmap),
        ("Overlay", overlay),
    ]
    for ax, (name, image) in zip(axes, panels):
        ax.imshow(image, cmap=cmap_name if name == "Attention" else None)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    model = build_model(args.model_path, device)
    transform = make_transform()
    records = read_split(args.split_csv, args.limit)
    counts = {
        "correct_cancer": 0,
        "correct_benign": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    summary_rows = []

    for index, record in enumerate(records, start=1):
        image_path = str(record["path"])
        if not Path(image_path).exists():
            print(f"Skipping missing image: {image_path}")
            continue

        original, image_tensor = load_image(image_path, transform, device)
        probability, heatmap = collect_attention_rollout(
            model,
            image_tensor,
            discard_ratio=args.discard_ratio,
            method=args.method,
            percentile=args.percentile,
        )
        label = int(record["label"])
        predicted = int(probability >= args.threshold)
        group = group_name(label, predicted)

        summary_rows.append(
            {
                "index": index,
                "path": image_path,
                "patient_id": record.get("patient_id", ""),
                "label": label,
                "label_name": CLASS_NAMES[label],
                "predicted": predicted,
                "predicted_name": CLASS_NAMES[predicted],
                "cancer_probability": probability,
                "group": group,
            }
        )

        if counts[group] >= args.max_per_group:
            continue
        counts[group] += 1
        stem = Path(image_path).stem.replace(" ", "_")
        output_path = args.output_dir / group / f"{counts[group]:02d}_{stem}.png"
        title = (
            f"True: {CLASS_NAMES[label]} | Pred: {CLASS_NAMES[predicted]} | "
            f"P(Cancer)={probability:.3f}"
        )
        save_attention_panel(original, heatmap, output_path, title, cmap_name=args.cmap)
        print(f"Saved {group}: {output_path}")

        if all(value >= args.max_per_group for value in counts.values()):
            print("Reached max-per-group for all groups.")
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "path",
            "patient_id",
            "label",
            "label_name",
            "predicted",
            "predicted_name",
            "cancer_probability",
            "group",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    counts_path = args.output_dir / "summary_counts.json"
    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("Saved summary:", summary_path)
    print("Saved counts:", counts_path)


if __name__ == "__main__":
    main()
