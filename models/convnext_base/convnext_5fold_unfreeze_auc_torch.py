#!/usr/bin/env python3
"""Train ConvNeXt 5-fold unfreezing experiments with PyTorch/timm.

This is the RTX 5090-friendly ConvNeXt counterpart to the ResNet50 and
ViT-B/16 experiments. It keeps the same patient-level split files and result
layout so final comparisons remain directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("This script requires timm. Install with: pip install timm") from exc


IMG_SIZE = 224
BATCH_SIZE = 16
BASE_SEED = 123
FOLDS = [1, 2, 3, 4, 5]

SPLIT_DIR = Path("results") / "splits"
FOLDS_DIR = SPLIT_DIR / "folds"
FINAL_SPLIT_DIR = SPLIT_DIR / "final_80_10_10"
DEFAULT_OUTPUT_DIR = Path("ConvNeXt") / "results" / "convnext_base_torch_unfreeze_5fold"

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
PRETRAINED = True
DROPOUT_RATE = 0.4

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 50
STAGE1_LR = 1e-4
STAGE2_LR = 1e-5
STAGE2_WEIGHT_DECAY = 1e-4
STAGE1_PATIENCE = 3
STAGE2_PATIENCE = 6
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-7

NUM_WORKERS = 4
THRESHOLD = 0.5

UNFREEZE_LEVELS = [
    {
        "name": "head_only",
        "label": "Head only",
        "stages": [],
        "freeze_norm": True,
        "description": "ConvNeXt frozen; train classifier head only.",
    },
    {
        "name": "stage4",
        "label": "Stage 4",
        "stages": [3],
        "freeze_norm": True,
        "description": "Unfreeze ConvNeXt stage 4, matching a ResNet conv5-style setting.",
    },
    {
        "name": "stage3_4",
        "label": "Stage 3-4",
        "stages": [2, 3],
        "freeze_norm": True,
        "description": "Unfreeze ConvNeXt stages 3 and 4, matching a ResNet conv4-5-style setting.",
    },
    {
        "name": "stage2_4",
        "label": "Stage 2-4",
        "stages": [1, 2, 3],
        "freeze_norm": True,
        "description": "Unfreeze ConvNeXt stages 2 through 4, matching a ResNet conv3-5-style setting.",
    },
    {
        "name": "all_non_norm",
        "label": "All non-Norm",
        "stages": "all",
        "freeze_norm": True,
        "description": "Unfreeze all non-normalization ConvNeXt backbone parameters.",
    },
]

PATH_REPLACEMENTS: list[tuple[str, str]] = []


@dataclass
class Config:
    output_dir: Path
    folds: list[int]
    settings: list[str] | None
    batch_size: int
    base_seed: int
    initial_epochs: int
    fine_tune_epochs: int
    dropout_rate: float
    stage2_lr: float
    stage2_weight_decay: float
    model_name: str
    max_train_samples: int | None
    max_val_samples: int | None
    num_workers: int
    no_pretrained: bool
    amp: bool
    final_train: bool
    final_split_dir: Path
    data_check_only: bool
    no_skip_existing: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train PyTorch/timm ConvNeXt 5-fold experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--settings", nargs="+", choices=[s["name"] for s in UNFREEZE_LEVELS])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--dropout-rate", type=float, default=DROPOUT_RATE)
    parser.add_argument("--stage2-lr", type=float, default=STAGE2_LR)
    parser.add_argument("--stage2-weight-decay", type=float, default=STAGE2_WEIGHT_DECAY)
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=(
            "timm ConvNeXt model name. Recommended main model: "
            "convnext_base.fb_in22k_ft_in1k; resource check: convnext_small.fb_in22k_ft_in1k."
        ),
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--final-train", action="store_true")
    parser.add_argument("--final-split-dir", type=Path, default=FINAL_SPLIT_DIR)
    parser.add_argument("--data-check-only", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    return Config(**vars(args))


def selected_unfreeze_levels(config: Config) -> list[dict[str, object]]:
    if config.settings is None:
        return UNFREEZE_LEVELS
    selected = set(config.settings)
    return [setting for setting in UNFREEZE_LEVELS if setting["name"] in selected]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_image_path(path: str) -> str:
    normalized = path.strip()
    for source, target in PATH_REPLACEMENTS:
        if normalized.startswith(source):
            normalized = target + normalized[len(source):]

    if os.name != "nt" and len(normalized) >= 3 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    if os.name != "nt":
        return normalized.replace("\\", "/")
    return normalized


def read_image_csv(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "path": normalize_image_path(row["path"]),
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "patient_id": row["patient_id"],
                }
            )
    if not rows:
        raise ValueError(f"No image records found in {csv_path}")
    return rows


def limit_records(records: list[dict[str, object]], limit: int | None) -> list[dict[str, object]]:
    if limit is None or limit >= len(records):
        return records
    by_label: dict[int, list[dict[str, object]]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)
    labels = sorted(by_label)
    selected = []
    positions = {label: 0 for label in labels}
    while len(selected) < limit:
        added = False
        for label in labels:
            pos = positions[label]
            if pos < len(by_label[label]) and len(selected) < limit:
                selected.append(by_label[label][pos])
                positions[label] += 1
                added = True
        if not added:
            break
    return selected


def verify_records_exist(records: list[dict[str, object]], label: str) -> None:
    missing = [str(record["path"]) for record in records if not Path(str(record["path"])).exists()]
    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(f"{label}: {len(missing)} image paths do not exist.\n{sample}")
    print(f"{label}: verified {len(records)} image paths")


class RoiDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(str(record["path"])).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return image, label


def make_transforms(training: bool):
    if training:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2)),
                transforms.ColorJitter(contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def make_loader(records: list[dict[str, object]], training: bool, config: Config, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = RoiDataset(records, make_transforms(training))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if training else None,
        persistent_workers=config.num_workers > 0,
    )


def build_model(config: Config) -> nn.Module:
    model = timm.create_model(
        config.model_name,
        pretrained=not config.no_pretrained,
        num_classes=1,
        drop_rate=config.dropout_rate,
    )
    return model


def set_head_trainable(model: nn.Module, trainable: bool = True) -> None:
    for name, param in model.named_parameters():
        if name.startswith("head."):
            param.requires_grad = trainable


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def is_norm_parameter_name(name: str) -> bool:
    parts = name.split(".")
    return any(part in {"norm", "norm_pre"} for part in parts)


def set_module_trainable(module: nn.Module, trainable: bool, freeze_norm: bool) -> None:
    for name, param in module.named_parameters():
        if freeze_norm and is_norm_parameter_name(name):
            param.requires_grad = False
        else:
            param.requires_grad = trainable


def apply_unfreeze_setting(model: nn.Module, setting: dict[str, object]) -> tuple[int, list[str]]:
    freeze_all(model)
    set_head_trainable(model, True)
    stages = getattr(model, "stages", None)
    if stages is None:
        raise ValueError("Expected timm ConvNeXt model to expose model.stages.")

    setting_stages = setting["stages"]
    freeze_norm = bool(setting.get("freeze_norm", True))
    if setting_stages == "all":
        selected_indices = list(range(len(stages)))
        set_module_trainable(getattr(model, "stem"), True, freeze_norm)
    else:
        selected_indices = [int(index) for index in setting_stages]

    for idx in selected_indices:
        set_module_trainable(stages[idx], True, freeze_norm)

    trainable_tensors = sum(1 for param in model.parameters() if param.requires_grad)
    unfrozen = [f"stages.{idx}" for idx in selected_indices]
    if setting_stages == "all":
        unfrozen.insert(0, "stem")
    return trainable_tensors, unfrozen


def trainable_parameters(model: nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable, total - trainable


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool) -> dict[str, float]:
    model.train()
    losses, y_true, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.tolist())
    return summarize_epoch(losses, y_true, y_prob, prefix="")


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    losses, y_true, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).reshape(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        losses.append(float(loss.cpu()))
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.tolist())
    return summarize_epoch(losses, y_true, y_prob, prefix="val_")


def summarize_epoch(losses: list[float], y_true: list[int], y_prob: list[float], prefix: str) -> dict[str, float]:
    y_true_array = np.array(y_true, dtype=np.int32)
    y_prob_array = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob_array >= THRESHOLD).astype(np.int32)
    accuracy = float(np.mean(y_pred == y_true_array)) if len(y_true_array) else 0.0
    auc = float(roc_auc_score(y_true_array, y_prob_array)) if len(np.unique(y_true_array)) > 1 else 0.0
    return {
        f"{prefix}loss": float(np.mean(losses)) if losses else 0.0,
        f"{prefix}accuracy": accuracy,
        f"{prefix}auc": auc,
    }


def fit_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs: int,
    patience: int,
    run_dir: Path,
    start_epoch: int,
    use_amp: bool,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], float]:
    rows = []
    best_auc = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for offset in range(epochs):
        epoch = start_epoch + offset + 1
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_auc"])
        row = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.4f} auc={row['auc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_auc={row['val_auc']:.4f} lr={row['lr']:.2e}"
        )

        if val_metrics["val_auc"] > best_auc:
            best_auc = val_metrics["val_auc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best val_auc={best_auc:.4f}")
                break

    run_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "training_log.csv", index=False)
    return rows, best_state, best_auc


@torch.no_grad()
def collect_predictions_timed(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    y_true, y_prob = [], []
    start = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).reshape(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true.extend(labels.numpy().astype(int).tolist())
        y_prob.extend(probs.tolist())
    elapsed = time.perf_counter() - start
    return np.array(y_true, dtype=np.int32), np.array(y_prob, dtype=np.float32), elapsed


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_pred = (y_prob >= THRESHOLD).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
    )


def run_one_fold(setting: dict[str, object], fold: int, config: Config, device: torch.device) -> dict[str, object]:
    run_seed = config.base_seed + fold
    set_global_seed(run_seed)
    train_records = limit_records(read_image_csv(FOLDS_DIR / f"fold_{fold}_train_images.csv"), config.max_train_samples)
    val_records = limit_records(read_image_csv(FOLDS_DIR / f"fold_{fold}_val_images.csv"), config.max_val_samples)
    verify_records_exist(train_records[: min(len(train_records), 20)], f"fold {fold} train sample")
    verify_records_exist(val_records[: min(len(val_records), 20)], f"fold {fold} val sample")

    train_loader = make_loader(train_records, True, config, run_seed)
    val_loader = make_loader(val_records, False, config, run_seed)
    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()

    freeze_all(model)
    set_head_trainable(model, True)
    optimizer1 = torch.optim.Adam(trainable_parameters(model), lr=STAGE1_LR)
    run_dir = config.output_dir / str(setting["name"]) / f"fold_{fold}"
    rows1, state1, best1 = fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer1,
        make_scheduler(optimizer1),
        device,
        config.initial_epochs,
        STAGE1_PATIENCE,
        run_dir / "stage1",
        0,
        config.amp and device.type == "cuda",
    )
    model.load_state_dict(state1)

    trainable_tensors, unfrozen_blocks = apply_unfreeze_setting(model, setting)
    optimizer2 = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config.stage2_lr,
        weight_decay=config.stage2_weight_decay,
    )
    rows2, state2, best2 = fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer2,
        make_scheduler(optimizer2),
        device,
        config.fine_tune_epochs,
        STAGE2_PATIENCE,
        run_dir,
        len(rows1),
        config.amp and device.type == "cuda",
    )
    model.load_state_dict(state2)
    val_metrics = evaluate(model, val_loader, criterion, device)

    return {
        "unfreeze_name": setting["name"],
        "unfreeze_label": setting["label"],
        "fold": fold,
        "val_auc": val_metrics["val_auc"],
        "best_history_val_auc": max(best1, best2),
        "last_history_val_auc": rows2[-1]["val_auc"] if rows2 else (rows1[-1]["val_auc"] if rows1 else 0.0),
        "trainable_base_layers": trainable_tensors,
        "unfrozen_blocks": ";".join(unfrozen_blocks),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "positive_class": "Cancer",
        "run_seed": run_seed,
    }


def run_final_training(setting: dict[str, object], config: Config, device: torch.device) -> dict[str, object]:
    set_global_seed(config.base_seed)
    train_records = limit_records(read_image_csv(config.final_split_dir / "final_train_images.csv"), config.max_train_samples)
    val_records = limit_records(read_image_csv(config.final_split_dir / "final_val_images.csv"), config.max_val_samples)
    test_records = read_image_csv(config.final_split_dir / "final_test_images.csv")
    verify_records_exist(train_records[: min(len(train_records), 20)], "final train sample")
    verify_records_exist(val_records[: min(len(val_records), 20)], "final val sample")
    verify_records_exist(test_records[: min(len(test_records), 20)], "final test sample")

    train_loader = make_loader(train_records, True, config, config.base_seed)
    val_loader = make_loader(val_records, False, config, config.base_seed)
    test_loader = make_loader(test_records, False, config, config.base_seed)
    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    train_start = time.perf_counter()

    freeze_all(model)
    set_head_trainable(model, True)
    optimizer1 = torch.optim.Adam(trainable_parameters(model), lr=STAGE1_LR)
    run_dir = config.output_dir / "final" / str(setting["name"])
    rows1, state1, best1 = fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer1,
        make_scheduler(optimizer1),
        device,
        config.initial_epochs,
        STAGE1_PATIENCE,
        run_dir / "stage1",
        0,
        config.amp and device.type == "cuda",
    )
    model.load_state_dict(state1)

    trainable_tensors, unfrozen_blocks = apply_unfreeze_setting(model, setting)
    optimizer2 = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config.stage2_lr,
        weight_decay=config.stage2_weight_decay,
    )
    rows2, state2, best2 = fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer2,
        make_scheduler(optimizer2),
        device,
        config.fine_tune_epochs,
        STAGE2_PATIENCE,
        run_dir,
        len(rows1),
        config.amp and device.type == "cuda",
    )
    model.load_state_dict(state2)
    training_time_s = time.perf_counter() - train_start
    epochs_ran = len(rows1) + len(rows2)

    val_metrics = evaluate(model, val_loader, criterion, device)
    y_true, y_prob, test_inference_time_s = collect_predictions_timed(model, test_loader, device)
    metrics = binary_metrics(y_true, y_prob)
    final_model_path = run_dir / "final_model.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "setting": setting, "config": vars(config)}, final_model_path)
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    test_steps = int(np.ceil(len(test_records) / config.batch_size))

    return {
        "unfreeze_name": setting["name"],
        "unfreeze_label": setting["label"],
        "val_auc": val_metrics["val_auc"],
        "test_auc": metrics["auc"],
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_tp": metrics["tp"],
        "test_tn": metrics["tn"],
        "test_fp": metrics["fp"],
        "test_fn": metrics["fn"],
        "best_history_val_auc": max(best1, best2),
        "last_history_val_auc": rows2[-1]["val_auc"] if rows2 else (rows1[-1]["val_auc"] if rows1 else 0.0),
        "trainable_base_layers": trainable_tensors,
        "unfrozen_blocks": ";".join(unfrozen_blocks),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_size_mb": final_model_path.stat().st_size / (1024 * 1024),
        "training_time_s": training_time_s,
        "avg_epoch_time_s": training_time_s / epochs_ran if epochs_ran else 0.0,
        "test_inference_time_s": test_inference_time_s,
        "test_ms_per_step": test_inference_time_s * 1000.0 / test_steps if test_steps else 0.0,
        "epochs_ran": epochs_ran,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "positive_class": "Cancer",
        "run_seed": config.base_seed,
        "batch_size": config.batch_size,
        "dropout_rate": config.dropout_rate,
        "stage2_lr": config.stage2_lr,
        "stage2_weight_decay": config.stage2_weight_decay,
        "model_name": config.model_name,
        "pretrained": not config.no_pretrained,
        "final_split_dir": str(config.final_split_dir),
        "final_model_path": str(final_model_path),
    }


def append_csv(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fmt_metric(value: object) -> str:
    return f"{float(value):.4f}"


def fmt_seconds(value: object) -> str:
    return f"{float(value):.2f}"


def model_display_name(model_name: str) -> str:
    prefix = model_name.split(".")[0]
    mapping = {
        "convnext_tiny": "ConvNeXt-Tiny",
        "convnext_small": "ConvNeXt-Small",
        "convnext_base": "ConvNeXt-Base",
        "convnext_large": "ConvNeXt-Large",
    }
    return mapping.get(prefix, model_name)


def write_simple_markdown_table(path: Path, headers: list[str], values: list[str]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_tables(output_dir: Path, row: dict[str, object]) -> None:
    model_display = model_display_name(str(row["model_name"]))
    model_name = f"{model_display}-{row['unfreeze_label']} (dropout={row['dropout_rate']})"

    performance_headers = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1"]
    performance_row = {
        "Model": model_name,
        "Accuracy": fmt_metric(row["test_accuracy"]),
        "AUC": fmt_metric(row["test_auc"]),
        "Precision": fmt_metric(row["test_precision"]),
        "Recall": fmt_metric(row["test_recall"]),
        "F1": fmt_metric(row["test_f1"]),
    }
    performance_csv = output_dir / "final_performance_table.csv"
    performance_md = output_dir / "final_performance_table.md"
    with performance_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=performance_headers)
        writer.writeheader()
        writer.writerow(performance_row)
    write_simple_markdown_table(
        performance_md,
        performance_headers,
        [performance_row[header] for header in performance_headers],
    )

    resource_headers = [
        "Model",
        "Total Params",
        "Trainable Params",
        "Non-Trainable Params",
        "Model Size (MB)",
        "Training Time (s)",
        "Avg Epoch Time (s)",
        "Test Inference Time (s)",
        "Test ms/step",
    ]
    resource_row = {
        "Model": model_name,
        "Total Params": str(row["total_params"]),
        "Trainable Params": str(row["trainable_params"]),
        "Non-Trainable Params": str(row["non_trainable_params"]),
        "Model Size (MB)": f"{float(row['model_size_mb']):.2f}",
        "Training Time (s)": fmt_seconds(row["training_time_s"]),
        "Avg Epoch Time (s)": fmt_seconds(row["avg_epoch_time_s"]),
        "Test Inference Time (s)": fmt_seconds(row["test_inference_time_s"]),
        "Test ms/step": f"{float(row['test_ms_per_step']):.2f}",
    }
    resource_csv = output_dir / "final_resource_table.csv"
    resource_md = output_dir / "final_resource_table.md"
    with resource_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resource_headers)
        writer.writeheader()
        writer.writerow(resource_row)
    write_simple_markdown_table(
        resource_md,
        resource_headers,
        [resource_row[header] for header in resource_headers],
    )


def load_fold_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "unfreeze_name": row["unfreeze_name"],
                    "unfreeze_label": row["unfreeze_label"],
                    "fold": int(row["fold"]),
                    "val_auc": float(row["val_auc"]),
                    "trainable_base_layers": int(row["trainable_base_layers"]),
                }
            )
    return rows


def save_summary(config: Config) -> None:
    rows = load_fold_rows(config.output_dir / "fold_auc_results.csv")
    if not rows:
        return
    summaries = []
    for setting in UNFREEZE_LEVELS:
        setting_rows = [row for row in rows if row["unfreeze_name"] == setting["name"]]
        if not setting_rows:
            continue
        aucs = np.array([row["val_auc"] for row in setting_rows], dtype=float)
        summaries.append(
            {
                "unfreeze_name": setting["name"],
                "unfreeze_label": setting["label"],
                "folds_completed": len(setting_rows),
                "mean_val_auc": float(np.mean(aucs)),
                "std_val_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "min_val_auc": float(np.min(aucs)),
                "max_val_auc": float(np.max(aucs)),
                "trainable_base_layers": setting_rows[0]["trainable_base_layers"],
                "description": setting["description"],
            }
        )
    pd.DataFrame(summaries).to_csv(config.output_dir / "unfreeze_auc_summary.csv", index=False)
    (config.output_dir / "unfreeze_auc_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    plot_results(config, rows, summaries)


def plot_results(
    config: Config,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> None:
    if not rows or not summaries:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_by_name = {setting["name"]: setting["label"] for setting in UNFREEZE_LEVELS}
    ordered_names = [str(item["unfreeze_name"]) for item in summaries]
    x_positions = np.arange(len(ordered_names))

    plt.figure(figsize=(11, 6))
    for x, name in zip(x_positions, ordered_names):
        setting_rows = sorted(
            [row for row in rows if row["unfreeze_name"] == name],
            key=lambda row: int(row["fold"]),
        )
        aucs = np.array([float(row["val_auc"]) for row in setting_rows], dtype=float)
        folds = [int(row["fold"]) for row in setting_rows]
        jitter = np.array([0.0]) if len(aucs) == 1 else np.linspace(-0.14, 0.14, len(aucs))
        plt.scatter(
            np.full(len(aucs), x) + jitter,
            aucs,
            s=54,
            alpha=0.85,
            color="#3f6b4f",
            edgecolor="white",
            linewidth=0.8,
            label="Fold AUC" if x == 0 else None,
        )
        for x_jitter, auc, fold in zip(jitter, aucs, folds):
            plt.text(
                x + x_jitter,
                auc + 0.003,
                str(fold),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#243047",
            )

    mean_values = [float(item["mean_val_auc"]) for item in summaries]
    std_values = [float(item["std_val_auc"]) for item in summaries]
    plt.plot(
        x_positions,
        mean_values,
        color="#b53d2d",
        marker="D",
        markersize=7,
        linewidth=2,
        label="5-fold mean AUC",
    )
    plt.errorbar(
        x_positions,
        mean_values,
        yerr=std_values,
        fmt="none",
        ecolor="#b53d2d",
        elinewidth=1.4,
        capsize=5,
        alpha=0.75,
        label="Mean +/- SD",
    )
    for x, mean_auc in zip(x_positions, mean_values):
        plt.text(
            x,
            mean_auc - 0.012,
            f"{mean_auc:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            color="#7a261d",
            fontweight="bold",
        )

    plt.xticks(x_positions, [label_by_name.get(name, name) for name in ordered_names])
    plt.ylabel("Validation AUC")
    plt.xlabel("ConvNeXt unfreezing setting")
    plt.title("ConvNeXt unfreezing level selection by 5-fold validation AUC")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(config.output_dir / "unfreeze_auc_5fold.png", dpi=300)
    plt.close()


def run_data_check(config: Config) -> None:
    if config.final_train:
        for split in ["final_train", "final_val", "final_test"]:
            verify_records_exist(read_image_csv(config.final_split_dir / f"{split}_images.csv"), split)
        return
    for fold in config.folds:
        verify_records_exist(read_image_csv(FOLDS_DIR / f"fold_{fold}_train_images.csv"), f"fold {fold} train")
        verify_records_exist(read_image_csv(FOLDS_DIR / f"fold_{fold}_val_images.csv"), f"fold {fold} val")


def main() -> None:
    config = parse_args()
    if config.data_check_only:
        run_data_check(config)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.final_train:
        settings = selected_unfreeze_levels(config)
        if len(settings) != 1:
            raise ValueError("--final-train expects exactly one --settings value.")
        row = run_final_training(settings[0], config, device)
        append_csv(config.output_dir / "final_results.csv", row, list(row.keys()))
        (config.output_dir / "final_results.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
        write_final_tables(config.output_dir, row)
        print(f"Finished final training: val_auc={row['val_auc']:.4f}, test_auc={row['test_auc']:.4f}")
        print("Saved final performance table:", config.output_dir / "final_performance_table.csv")
        print("Saved final resource table:", config.output_dir / "final_resource_table.csv")
        return

    results_csv = config.output_dir / "fold_auc_results.csv"
    existing = load_fold_rows(results_csv)
    completed = {(row["unfreeze_name"], row["fold"]) for row in existing}

    fieldnames = [
        "unfreeze_name",
        "unfreeze_label",
        "fold",
        "val_auc",
        "best_history_val_auc",
        "last_history_val_auc",
        "trainable_base_layers",
        "unfrozen_blocks",
        "train_images",
        "val_images",
        "positive_class",
        "run_seed",
    ]
    for setting in selected_unfreeze_levels(config):
        for fold in config.folds:
            key = (setting["name"], fold)
            if not config.no_skip_existing and key in completed:
                print(f"Skip existing result: {setting['name']} fold {fold}")
                continue
            print("=" * 80)
            print(f"Run setting={setting['name']} fold={fold}")
            print(setting["description"])
            row = run_one_fold(setting, fold, config, device)
            append_csv(results_csv, row, fieldnames)
            print(f"Finished {setting['name']} fold {fold}: val_auc={row['val_auc']:.4f}")
    save_summary(config)
    print("Saved fold results:", results_csv)


if __name__ == "__main__":
    main()
