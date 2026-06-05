import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def resolve_data_dir() -> str:
    # Find the DDSM ROI data directory. Environment variables have priority.
    candidates = [
        os.environ.get("DDSM_DATA_DIR"),
        "E:/DDSM",
        os.environ.get("DDSM_ROI_DIR"),
        "/mnt/e/DDSM/ROI",
        "/content/drive/MyDrive/DDSM/ROI-data",
        "E:/DDSM/ROI",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find ROI dataset. Set DDSM_ROI_DIR or place data in one of: "
        "/mnt/e/DDSM/ROI, /content/drive/MyDrive/DDSM/ROI-data, E:/DDSM/ROI"
    )


# Image formats included in the metadata tables.
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}

# Exclude annotation or overlay images; these should not be used as normal inputs.
EXCLUDED_NAME_PARTS = ("mask", "overlay")


def infer_patient_id(class_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(class_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]

    # Fallback for flat folders, for example C_0029_1.LEFT_CC.jpg -> 0029.
    stem = image_path.stem.replace("-", "_")
    parts = [part for part in stem.split("_") if part]
    for part in parts:
        if part.isdigit():
            return part
    return stem


def find_image_records(data_dir: str):
    # Scan class folders and build one metadata record for every image.
    # The later split is patient-level, so all images from one patient stay together.
    root = Path(data_dir)
    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected class folders under {root}, for example Benign/ and Cancer/."
        )

    # Keep labels stable when the expected folders exist:
    # Benign -> 0, Cancer -> 1.
    class_names = [path.name for path in class_dirs]
    if "Benign" in class_names and "Cancer" in class_names:
        class_names = ["Benign", "Cancer"]
        class_dirs = [root / name for name in class_names]

    records = []
    patient_labels = {}
    for label, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lower_name = image_path.name.lower()
            if any(part in lower_name for part in EXCLUDED_NAME_PARTS):
                continue

            # Prefix patient_id with the class folder to avoid collisions such as
            # Benign/0029 and Cancer/0029.
            patient_id = f"{class_dir.name}/{infer_patient_id(class_dir, image_path)}"
            if patient_id in patient_labels and patient_labels[patient_id] != label:
                raise ValueError(f"Patient appears in multiple classes: {patient_id}")

            patient_labels[patient_id] = label
            records.append(
                {
                    "path": str(image_path),
                    "patient_id": patient_id,
                    "label": label,
                    "class_name": class_names[label],
                }
            )

    if not records:
        raise ValueError(f"No image files found under {root}.")
    return records, class_names


def build_patient_label_map(records):
    # Convert image-level records to patient-level labels.
    patient_label_map = {}
    for record in records:
        patient_id = record["patient_id"]
        label = record["label"]
        if patient_id in patient_label_map and patient_label_map[patient_id] != label:
            raise ValueError(f"Patient appears in multiple classes: {patient_id}")
        patient_label_map[patient_id] = label
    return patient_label_map


def split_train_test_patient_level(patient_ids, patient_labels, seed, test_fraction=0.2):
    # First split: hold out an independent test set by patient.
    # This split is shuffled and stratified, not an order-based split.

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=seed,
    )
    train_idx, test_idx = next(
        splitter.split(np.zeros(len(patient_ids)), patient_labels)
    )
    train_ids = [patient_ids[idx] for idx in train_idx]
    test_ids = [patient_ids[idx] for idx in test_idx]
    return train_ids, test_ids


def make_patient_folds(train_patient_ids, patient_label_map, seed, n_splits=5):
    # Second split: split only the training patients into K folds.
    # Each fold has its own fold_train and fold_val patients.
    train_patient_ids = np.array(sorted(train_patient_ids))
    train_patient_labels = np.array(
        [patient_label_map[patient_id] for patient_id in train_patient_ids],
        dtype="int32",
    )

    class_counts = Counter(train_patient_labels.tolist())
    min_class_count = min(class_counts.values())
    if min_class_count < n_splits:
        raise ValueError(
            f"Cannot create {n_splits}-fold CV: the smallest class has only "
            f"{min_class_count} training patients."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        splitter.split(np.zeros(len(train_patient_ids)), train_patient_labels),
        start=1,
    ):
        folds.append(
            {
                "fold": fold_idx,
                "train_patient_ids": train_patient_ids[fold_train_idx].tolist(),
                "val_patient_ids": train_patient_ids[fold_val_idx].tolist(),
            }
        )
    return folds


def records_for_patients(records, patient_ids):
    # Filter image-level records by patient IDs.
    patient_ids = set(patient_ids)
    return [record for record in records if record["patient_id"] in patient_ids]


def summarize_records(records, class_names):
    # Count images per class.
    counts = Counter(record["label"] for record in records)
    return {class_names[idx]: counts.get(idx, 0) for idx in range(len(class_names))}


def summarize_patients(patient_ids, patient_label_map, class_names):
    # Count patients per class.
    counts = Counter(patient_label_map[patient_id] for patient_id in patient_ids)
    return {class_names[idx]: counts.get(idx, 0) for idx in range(len(class_names))}


def assert_patient_disjoint(*patient_id_groups):
    # Make sure no patient appears in more than one split within the same check.
    seen = set()
    for group in patient_id_groups:
        group = set(group)
        overlap = seen & group
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Patient leakage detected: {examples}")
        seen.update(group)


def write_patient_csv(path, patient_ids, patient_label_map, class_names):
    # Save a patient-level metadata table.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "class_name"])
        for patient_id in sorted(patient_ids):
            label = patient_label_map[patient_id]
            writer.writerow([patient_id, label, class_names[label]])


def write_image_csv(path, records):
    # Save an image-level metadata table.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "patient_id", "label", "class_name"])
        for record in records:
            writer.writerow(
                [
                    record["path"],
                    record["patient_id"],
                    record["label"],
                    record["class_name"],
                ]
            )


def write_fold_assignment_csv(path, folds, patient_label_map, class_names):
    # Save one compact table showing which validation fold each training patient belongs to.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "class_name", "val_fold"])
        for fold in folds:
            for patient_id in sorted(fold["val_patient_ids"]):
                label = patient_label_map[patient_id]
                writer.writerow([patient_id, label, class_names[label], fold["fold"]])


def save_fold_files(folds, records, patient_label_map, class_names, folds_dir):
    # For each fold, save both patient-level and image-level train/validation tables.
    fold_summaries = []
    folds_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        fold_idx = fold["fold"]
        fold_train_patient_ids = fold["train_patient_ids"]
        fold_val_patient_ids = fold["val_patient_ids"]
        assert_patient_disjoint(fold_train_patient_ids, fold_val_patient_ids)

        fold_train_records = records_for_patients(records, fold_train_patient_ids)
        fold_val_records = records_for_patients(records, fold_val_patient_ids)

        fold_prefix = folds_dir / f"fold_{fold_idx}"
        write_patient_csv(
            Path(f"{fold_prefix}_train_patients.csv"),
            fold_train_patient_ids,
            patient_label_map,
            class_names,
        )
        write_patient_csv(
            Path(f"{fold_prefix}_val_patients.csv"),
            fold_val_patient_ids,
            patient_label_map,
            class_names,
        )
        write_image_csv(Path(f"{fold_prefix}_train_images.csv"), fold_train_records)
        write_image_csv(Path(f"{fold_prefix}_val_images.csv"), fold_val_records)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_patients": len(fold_train_patient_ids),
                "val_patients": len(fold_val_patient_ids),
                "train_images": len(fold_train_records),
                "val_images": len(fold_val_records),
                "train_patient_counts": summarize_patients(
                    fold_train_patient_ids,
                    patient_label_map,
                    class_names,
                ),
                "val_patient_counts": summarize_patients(
                    fold_val_patient_ids,
                    patient_label_map,
                    class_names,
                ),
                "train_image_counts": summarize_records(fold_train_records, class_names),
                "val_image_counts": summarize_records(fold_val_records, class_names),
            }
        )

    return fold_summaries


def main():
    # Fixed seed makes the train/test split and CV folds reproducible.
    seed = int(os.environ.get("DDSM_SPLIT_SEED", "123"))
    random.seed(seed)
    np.random.seed(seed)

    # Defaults: 80% training, 20% independent test, then 5-fold CV inside training.
    test_fraction = float(os.environ.get("DDSM_TEST_FRACTION", "0.2"))
    n_folds = int(os.environ.get("DDSM_CV_FOLDS", "5"))
    if not 0 < test_fraction < 1:
        raise ValueError("DDSM_TEST_FRACTION must be between 0 and 1.")
    if n_folds < 2:
        raise ValueError("DDSM_CV_FOLDS must be at least 2.")

    data_dir = resolve_data_dir()
    records, class_names = find_image_records(data_dir)
    patient_label_map = build_patient_label_map(records)

    patient_ids = np.array(sorted(patient_label_map))
    patient_labels = np.array(
        [patient_label_map[patient_id] for patient_id in patient_ids],
        dtype="int32",
    )

    print("Using data_dir:", data_dir)
    print("Classes:", class_names)
    print("Total images:", len(records))
    print("Total patients:", len(patient_ids))
    print("Image label counts:", summarize_records(records, class_names))
    print(
        "Patient label counts:",
        summarize_patients(patient_ids, patient_label_map, class_names),
    )

    # Hold out the independent test patients first.
    train_patient_ids, test_patient_ids = split_train_test_patient_level(
        patient_ids,
        patient_labels,
        seed,
        test_fraction=test_fraction,
    )
    assert_patient_disjoint(train_patient_ids, test_patient_ids)

    train_records = records_for_patients(records, train_patient_ids)
    test_records = records_for_patients(records, test_patient_ids)

    # Create 5-fold CV only from training patients.
    folds = make_patient_folds(
        train_patient_ids,
        patient_label_map,
        seed,
        n_splits=n_folds,
    )

    print("\n===== Train/Test Split Summary =====")
    print("Train patients:", len(train_patient_ids))
    print("Test patients:", len(test_patient_ids))
    print("Train images:", len(train_records))
    print("Test images:", len(test_records))
    print("Train image counts:", summarize_records(train_records, class_names))
    print("Test image counts:", summarize_records(test_records, class_names))

    split_dir = Path("results") / "splits"
    folds_dir = split_dir / "folds"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save the outer 80/20 split metadata.
    write_patient_csv(
        split_dir / "train_patients.csv",
        train_patient_ids,
        patient_label_map,
        class_names,
    )
    write_patient_csv(
        split_dir / "test_patients.csv",
        test_patient_ids,
        patient_label_map,
        class_names,
    )
    write_image_csv(split_dir / "train_images.csv", train_records)
    write_image_csv(split_dir / "test_images.csv", test_records)

    # Save the inner K-fold metadata based only on training patients.
    write_fold_assignment_csv(
        split_dir / "train_cv_fold_assignments.csv",
        folds,
        patient_label_map,
        class_names,
    )
    fold_summaries = save_fold_files(
        folds,
        records,
        patient_label_map,
        class_names,
        folds_dir,
    )

    print(f"\n===== {n_folds}-Fold CV Summary, Training Data Only =====")
    for item in fold_summaries:
        print(
            f"Fold {item['fold']}: "
            f"train patients={item['train_patients']}, "
            f"val patients={item['val_patients']}, "
            f"train images={item['train_images']}, "
            f"val images={item['val_images']}"
        )

    # Save one JSON summary for quick checking and reproducibility.
    summary = {
        "data_dir": data_dir,
        "seed": seed,
        "test_fraction": test_fraction,
        "cv_folds": n_folds,
        "classes": class_names,
        "total_images": len(records),
        "total_patients": len(patient_ids),
        "train_images": len(train_records),
        "test_images": len(test_records),
        "train_patients": len(train_patient_ids),
        "test_patients": len(test_patient_ids),
        "all_image_counts": summarize_records(records, class_names),
        "train_image_counts": summarize_records(train_records, class_names),
        "test_image_counts": summarize_records(test_records, class_names),
        "all_patient_counts": summarize_patients(
            patient_ids,
            patient_label_map,
            class_names,
        ),
        "train_patient_counts": summarize_patients(
            train_patient_ids,
            patient_label_map,
            class_names,
        ),
        "test_patient_counts": summarize_patients(
            test_patient_ids,
            patient_label_map,
            class_names,
        ),
        "folds": fold_summaries,
    }
    summary_path = split_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved split files to:", split_dir)
    print("Saved:", split_dir / "train_images.csv")
    print("Saved:", split_dir / "test_images.csv")
    print("Saved:", split_dir / "train_cv_fold_assignments.csv")
    print("Saved fold files to:", folds_dir)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
