import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
BASE_SEED = 123
FOLDS = [1, 2, 3, 4, 5]
SELECTED_UNFREEZE_NAMES = None
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
DATA_CHECK_ONLY = False
FINAL_TRAIN = False

SPLIT_DIR = Path("results") / "splits"
FOLDS_DIR = SPLIT_DIR / "folds"
FINAL_SPLIT_DIR = SPLIT_DIR / "final_80_10_10"
DEFAULT_OUTPUT_DIR = Path("ResNet50") / "results" / "resnet50_unfreeze_5fold"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

RESULTS_CSV = OUTPUT_DIR / "fold_auc_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "unfreeze_auc_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "unfreeze_auc_summary.json"
PLOT_PATH = OUTPUT_DIR / "unfreeze_auc_5fold.png"
FINAL_RESULTS_CSV = OUTPUT_DIR / "final_results.csv"
FINAL_RESULTS_JSON = OUTPUT_DIR / "final_results.json"
FINAL_PERFORMANCE_CSV = OUTPUT_DIR / "final_performance_table.csv"
FINAL_PERFORMANCE_MD = OUTPUT_DIR / "final_performance_table.md"
FINAL_RESOURCE_CSV = OUTPUT_DIR / "final_resource_table.csv"
FINAL_RESOURCE_MD = OUTPUT_DIR / "final_resource_table.md"

# Keep this True to match the old notebook: augmentation is inside the model and
# is active only during training.
USE_DATA_AUGMENTATION = True

# The old notebook used simple Rescaling(1/255) before ResNet50. Keep that here
# for a fair comparison with the previous experiment.
USE_RESNET_PREPROCESS_INPUT = False

RESNET_WEIGHTS = "imagenet"
DROPOUT_RATE = 0.5
L2_REG = 1e-4

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 50
STAGE1_LR = 1e-4
STAGE2_LR = 1e-5
STAGE2_WEIGHT_DECAY = 1e-5

STAGE1_PATIENCE = 3
STAGE2_PATIENCE = 6
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-7

# Set to False if you want to rerun completed fold/config combinations.
SKIP_EXISTING_RESULTS = True

# Set to True if you also want to save a full .keras checkpoint for every run.
# This can create many large files: len(UNFREEZE_LEVELS) * 5 checkpoints.
SAVE_BEST_MODELS = False

# BatchNorm layers are usually kept frozen for stable fine-tuning on small
# medical image datasets.
FREEZE_BATCH_NORM = True

# Optional manual path replacements for image paths stored in the split CSVs.
# Example for Colab:
# PATH_REPLACEMENTS = [(r"E:\DDSM\ROI", "/content/drive/MyDrive/DDSM/ROI")]
PATH_REPLACEMENTS = []

UNFREEZE_LEVELS = [
    {
        "name": "head_only",
        "label": "Head only",
        "prefixes": [],
        "description": "ResNet50 frozen; train classifier head only.",
    },
    {
        "name": "conv5",
        "label": "conv5",
        "prefixes": ["conv5_"],
        "description": "Unfreeze ResNet50 conv5 block.",
    },
    {
        "name": "conv4_5",
        "label": "conv4-5",
        "prefixes": ["conv4_", "conv5_"],
        "description": "Unfreeze ResNet50 conv4 and conv5 blocks.",
    },
    {
        "name": "conv3_5",
        "label": "conv3-5",
        "prefixes": ["conv3_", "conv4_", "conv5_"],
        "description": "Unfreeze ResNet50 conv3 through conv5 blocks.",
    },
    {
        "name": "conv2_5",
        "label": "conv2-5",
        "prefixes": ["conv2_", "conv3_", "conv4_", "conv5_"],
        "description": "Unfreeze ResNet50 conv2 through conv5 blocks.",
    },
    {
        "name": "all_non_bn",
        "label": "All non-BN",
        "prefixes": "all",
        "description": "Unfreeze all non-BatchNorm ResNet50 layers.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 5-fold fine-tuning experiments with AUC tracking."
    )
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="Base random seed. Final training uses this value directly.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        choices=[setting["name"] for setting in UNFREEZE_LEVELS],
        help="Subset of unfreeze settings to run.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for this experiment's logs, CSV summaries, and plots. "
            "Use a new directory for comparisons such as batch size 32."
        ),
    )
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--dropout-rate", type=float, default=DROPOUT_RATE)
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=STAGE2_LR,
        help="Learning rate for the fine-tuning stage after unfreezing ResNet layers.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--resnet-weights",
        choices=["imagenet", "none"],
        default=RESNET_WEIGHTS,
        help="Use 'none' for an offline smoke test without downloading ImageNet weights.",
    )
    parser.add_argument(
        "--resnet-preprocess-input",
        action="store_true",
        help=(
            "Use keras.applications.resnet50.preprocess_input instead of "
            "simple Rescaling(1/255). This matches ImageNet-pretrained ResNet50."
        ),
    )
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--save-best-models", action="store_true")
    parser.add_argument(
        "--final-train",
        action="store_true",
        help=(
            "Train on results/splits/final_80_10_10/final_train_images.csv, "
            "use final_val for early stopping, and report final_test AUC."
        ),
    )
    parser.add_argument(
        "--final-split-dir",
        type=Path,
        default=FINAL_SPLIT_DIR,
        help="Directory containing final_train/final_val/final_test CSV files.",
    )
    parser.add_argument(
        "--data-check-only",
        action="store_true",
        help="Load split CSVs, verify image paths, print TensorFlow GPU info, then exit.",
    )
    return parser.parse_args()


def configure_from_args(args):
    global OUTPUT_DIR
    global RESULTS_CSV
    global SUMMARY_CSV
    global SUMMARY_JSON
    global PLOT_PATH
    global FINAL_RESULTS_CSV
    global FINAL_RESULTS_JSON
    global FINAL_PERFORMANCE_CSV
    global FINAL_PERFORMANCE_MD
    global FINAL_RESOURCE_CSV
    global FINAL_RESOURCE_MD
    global BATCH_SIZE
    global BASE_SEED
    global FOLDS
    global SELECTED_UNFREEZE_NAMES
    global INITIAL_EPOCHS
    global FINE_TUNE_EPOCHS
    global MAX_TRAIN_SAMPLES
    global MAX_VAL_SAMPLES
    global RESNET_WEIGHTS
    global DROPOUT_RATE
    global STAGE2_LR
    global USE_RESNET_PREPROCESS_INPUT
    global SKIP_EXISTING_RESULTS
    global SAVE_BEST_MODELS
    global DATA_CHECK_ONLY
    global FINAL_TRAIN
    global FINAL_SPLIT_DIR

    OUTPUT_DIR = args.output_dir
    RESULTS_CSV = OUTPUT_DIR / "fold_auc_results.csv"
    SUMMARY_CSV = OUTPUT_DIR / "unfreeze_auc_summary.csv"
    SUMMARY_JSON = OUTPUT_DIR / "unfreeze_auc_summary.json"
    PLOT_PATH = OUTPUT_DIR / "unfreeze_auc_5fold.png"
    FINAL_RESULTS_CSV = OUTPUT_DIR / "final_results.csv"
    FINAL_RESULTS_JSON = OUTPUT_DIR / "final_results.json"
    FINAL_PERFORMANCE_CSV = OUTPUT_DIR / "final_performance_table.csv"
    FINAL_PERFORMANCE_MD = OUTPUT_DIR / "final_performance_table.md"
    FINAL_RESOURCE_CSV = OUTPUT_DIR / "final_resource_table.csv"
    FINAL_RESOURCE_MD = OUTPUT_DIR / "final_resource_table.md"

    BATCH_SIZE = args.batch_size
    BASE_SEED = args.base_seed
    FOLDS = args.folds
    SELECTED_UNFREEZE_NAMES = set(args.settings) if args.settings else None
    INITIAL_EPOCHS = args.initial_epochs
    FINE_TUNE_EPOCHS = args.fine_tune_epochs
    MAX_TRAIN_SAMPLES = args.max_train_samples
    MAX_VAL_SAMPLES = args.max_val_samples
    RESNET_WEIGHTS = None if args.resnet_weights == "none" else args.resnet_weights
    DROPOUT_RATE = args.dropout_rate
    STAGE2_LR = args.stage2_lr
    USE_RESNET_PREPROCESS_INPUT = args.resnet_preprocess_input
    SKIP_EXISTING_RESULTS = not args.no_skip_existing
    SAVE_BEST_MODELS = SAVE_BEST_MODELS or args.save_best_models
    DATA_CHECK_ONLY = args.data_check_only
    FINAL_TRAIN = args.final_train
    FINAL_SPLIT_DIR = args.final_split_dir


def selected_unfreeze_levels():
    if SELECTED_UNFREEZE_NAMES is None:
        return UNFREEZE_LEVELS
    return [
        setting for setting in UNFREEZE_LEVELS
        if setting["name"] in SELECTED_UNFREEZE_NAMES
    ]


# ---------------------------------------------------------------------------
# Reproducibility and data loading
# ---------------------------------------------------------------------------

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def configure_tensorflow_runtime():
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("TensorFlow GPU devices: none")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("TensorFlow GPU devices:", ", ".join(gpu.name for gpu in gpus))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def read_image_csv(csv_path):
    records = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "path": normalize_image_path(row["path"]),
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "patient_id": row["patient_id"],
                }
            )
    if not records:
        raise ValueError(f"No image records found in {csv_path}")
    return records


def limit_records(records, limit):
    if limit is None or limit >= len(records):
        return records

    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    if len(by_label) < 2:
        return records[:limit]

    labels = sorted(by_label)
    selected = []
    positions = {label: 0 for label in labels}
    while len(selected) < limit:
        added_this_round = False
        for label in labels:
            position = positions[label]
            label_records = by_label[label]
            if position < len(label_records) and len(selected) < limit:
                selected.append(label_records[position])
                positions[label] += 1
                added_this_round = True
        if not added_this_round:
            break
    return selected


def verify_records_exist(records, label):
    missing = [record["path"] for record in records if not Path(record["path"]).exists()]
    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{label}: {len(missing)} image paths do not exist. Examples:\n{sample}"
        )
    print(f"{label}: verified {len(records)} image paths")


def normalize_image_path(path):
    normalized = path.strip()
    for source, target in PATH_REPLACEMENTS:
        if normalized.startswith(source):
            normalized = target + normalized[len(source):]

    # Split files generated on Windows store paths like E:\DDSM\ROI\...
    # In WSL/Linux, map those to /mnt/e/DDSM/ROI/... automatically.
    if os.name != "nt" and len(normalized) >= 3 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[3:].replace("\\", "/")
        normalized = f"/mnt/{drive}/{rest}"
    elif os.name != "nt":
        normalized = normalized.replace("\\", "/")

    return normalized


def infer_class_names(records):
    label_to_name = {}
    for record in records:
        label_to_name[record["label"]] = record["class_name"]
    return [label_to_name[idx] for idx in sorted(label_to_name)]


def decode_and_resize(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(records, training, seed):
    paths = [record["path"] for record in records]
    labels = [record["label"] for record in records]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_data_augmentation():
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.10),
            keras.layers.RandomZoom(0.20),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.RandomContrast(0.20),
        ],
        name="data_augmentation",
    )


def build_model():
    base_model = keras.applications.ResNet50(
        weights=RESNET_WEIGHTS,
        include_top=False,
        input_shape=IMG_SIZE + (3,),
    )

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    if USE_RESNET_PREPROCESS_INPUT:
        x = keras.applications.resnet50.preprocess_input(inputs)
    else:
        x = keras.layers.Rescaling(1.0 / 255, name="rescaling")(inputs)

    if USE_DATA_AUGMENTATION:
        x = build_data_augmentation()(x)

    # training=False keeps BatchNorm in inference mode during fine-tuning.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(L2_REG),
        name="cancer_probability",
    )(x)

    model = keras.Model(inputs, outputs, name="resnet50_roi_auc")
    return model, base_model


def compile_model(model, learning_rate, weight_decay=None):
    if weight_decay is not None:
        try:
            optimizer = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        except Exception:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def layer_matches_setting(layer_name, prefixes):
    if prefixes == "all":
        return True
    return any(layer_name.startswith(prefix) for prefix in prefixes)


def apply_unfreeze_setting(base_model, setting):
    prefixes = setting["prefixes"]
    base_model.trainable = True

    trainable_count = 0
    for layer in base_model.layers:
        layer.trainable = layer_matches_setting(layer.name, prefixes)
        if FREEZE_BATCH_NORM and isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        trainable_count += int(layer.trainable)

    return trainable_count


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def make_callbacks(run_dir, monitor="val_auc"):
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=STAGE2_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]
    if SAVE_BEST_MODELS:
        callbacks.insert(
            0,
            keras.callbacks.ModelCheckpoint(
                str(run_dir / "best_model.keras"),
                monitor=monitor,
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
        )
    return callbacks


def collect_predictions(model, dataset):
    y_true = []
    y_prob = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_true.append(labels.numpy().reshape(-1))
        y_prob.append(probs)
    return np.concatenate(y_true).astype("int32"), np.concatenate(y_prob)


def exact_auc(model, dataset):
    y_true, y_prob = collect_predictions(model, dataset)
    return float(roc_auc_score(y_true, y_prob))


def collect_predictions_timed(model, dataset):
    y_true = []
    y_prob = []
    start = time.perf_counter()
    for images, labels in dataset:
        probs = model(images, training=False).numpy().reshape(-1)
        y_true.append(labels.numpy().reshape(-1))
        y_prob.append(probs)
    elapsed = time.perf_counter() - start
    return np.concatenate(y_true).astype("int32"), np.concatenate(y_prob), elapsed


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype("int32")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(y_true, y_prob))

    return {
        "accuracy": float(accuracy),
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def count_params(weights):
    return int(sum(np.prod(weight.shape) for weight in weights))


def fmt_metric(value):
    return f"{float(value):.4f}"


def fmt_seconds(value):
    return f"{float(value):.2f}"


def history_best(history, metric_name):
    values = history.history.get(metric_name, [])
    if not values:
        return None
    return float(np.max(values))


def history_last(history, metric_name):
    values = history.history.get(metric_name, [])
    if not values:
        return None
    return float(values[-1])


def run_one_fold(setting, fold):
    run_seed = BASE_SEED + fold
    set_global_seed(run_seed)

    train_csv = FOLDS_DIR / f"fold_{fold}_train_images.csv"
    val_csv = FOLDS_DIR / f"fold_{fold}_val_images.csv"
    train_records = limit_records(read_image_csv(train_csv), MAX_TRAIN_SAMPLES)
    val_records = limit_records(read_image_csv(val_csv), MAX_VAL_SAMPLES)
    verify_records_exist(train_records[: min(len(train_records), 20)], f"fold {fold} train sample")
    verify_records_exist(val_records[: min(len(val_records), 20)], f"fold {fold} val sample")
    class_names = infer_class_names(train_records + val_records)

    train_ds = make_dataset(train_records, training=True, seed=run_seed)
    val_ds = make_dataset(val_records, training=False, seed=run_seed)

    model, base_model = build_model()

    # Stage 1: train only the classification head.
    base_model.trainable = False
    compile_model(model, learning_rate=STAGE1_LR)
    stage1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=STAGE1_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    # Stage 2: train the selected ResNet50 blocks. BatchNorm stays frozen.
    trainable_base_layers = apply_unfreeze_setting(base_model, setting)
    compile_model(
        model,
        learning_rate=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )

    run_dir = OUTPUT_DIR / setting["name"] / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    callbacks = make_callbacks(run_dir)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    val_auc = exact_auc(model, val_ds)
    best_history_val_auc = history_best(history2, "val_auc")
    if best_history_val_auc is None:
        best_history_val_auc = history_best(history1, "val_auc")
    last_history_val_auc = history_last(history2, "val_auc")
    if last_history_val_auc is None:
        last_history_val_auc = history_last(history1, "val_auc")

    result = {
        "unfreeze_name": setting["name"],
        "unfreeze_label": setting["label"],
        "fold": fold,
        "val_auc": val_auc,
        "best_history_val_auc": best_history_val_auc,
        "last_history_val_auc": last_history_val_auc,
        "trainable_base_layers": trainable_base_layers,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "positive_class": class_names[1],
        "run_seed": run_seed,
    }

    keras.backend.clear_session()
    return result


def run_final_training(setting):
    run_seed = BASE_SEED
    set_global_seed(run_seed)

    train_csv = FINAL_SPLIT_DIR / "final_train_images.csv"
    val_csv = FINAL_SPLIT_DIR / "final_val_images.csv"
    test_csv = FINAL_SPLIT_DIR / "final_test_images.csv"

    train_records = limit_records(read_image_csv(train_csv), MAX_TRAIN_SAMPLES)
    val_records = limit_records(read_image_csv(val_csv), MAX_VAL_SAMPLES)
    test_records = read_image_csv(test_csv)

    verify_records_exist(
        train_records[: min(len(train_records), 20)],
        "final train sample",
    )
    verify_records_exist(
        val_records[: min(len(val_records), 20)],
        "final val sample",
    )
    verify_records_exist(
        test_records[: min(len(test_records), 20)],
        "final test sample",
    )
    class_names = infer_class_names(train_records + val_records + test_records)

    train_ds = make_dataset(train_records, training=True, seed=run_seed)
    val_ds = make_dataset(val_records, training=False, seed=run_seed)
    test_ds = make_dataset(test_records, training=False, seed=run_seed)

    model, base_model = build_model()

    base_model.trainable = False
    compile_model(model, learning_rate=STAGE1_LR)
    stage1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=STAGE1_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    train_start = time.perf_counter()
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    trainable_base_layers = apply_unfreeze_setting(base_model, setting)
    compile_model(
        model,
        learning_rate=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )

    run_dir = OUTPUT_DIR / "final" / setting["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    callbacks = make_callbacks(run_dir)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    training_time_s = time.perf_counter() - train_start
    epochs_ran = len(history1.epoch) + len(history2.epoch)
    avg_epoch_time_s = training_time_s / epochs_ran if epochs_ran else 0.0

    val_auc = exact_auc(model, val_ds)
    y_true, y_prob, test_inference_time_s = collect_predictions_timed(model, test_ds)
    test_metrics = binary_metrics(y_true, y_prob)
    test_auc = test_metrics["auc"]
    test_steps = int(np.ceil(len(test_records) / BATCH_SIZE)) if test_records else 0
    test_ms_per_step = test_inference_time_s * 1000.0 / test_steps if test_steps else 0.0
    best_history_val_auc = history_best(history2, "val_auc")
    if best_history_val_auc is None:
        best_history_val_auc = history_best(history1, "val_auc")
    last_history_val_auc = history_last(history2, "val_auc")
    if last_history_val_auc is None:
        last_history_val_auc = history_last(history1, "val_auc")

    final_model_path = run_dir / "final_model.keras"
    model.save(final_model_path)
    model_size_mb = final_model_path.stat().st_size / (1024 * 1024)

    result = {
        "unfreeze_name": setting["name"],
        "unfreeze_label": setting["label"],
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_tp": test_metrics["tp"],
        "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "best_history_val_auc": best_history_val_auc,
        "last_history_val_auc": last_history_val_auc,
        "trainable_base_layers": trainable_base_layers,
        "total_params": model.count_params(),
        "trainable_params": count_params(model.trainable_weights),
        "non_trainable_params": count_params(model.non_trainable_weights),
        "model_size_mb": model_size_mb,
        "training_time_s": training_time_s,
        "avg_epoch_time_s": avg_epoch_time_s,
        "test_inference_time_s": test_inference_time_s,
        "test_ms_per_step": test_ms_per_step,
        "epochs_ran": epochs_ran,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "test_images": len(test_records),
        "positive_class": class_names[1],
        "run_seed": run_seed,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "stage2_lr": STAGE2_LR,
        "use_resnet_preprocess_input": USE_RESNET_PREPROCESS_INPUT,
        "final_split_dir": str(FINAL_SPLIT_DIR),
        "final_model_path": str(final_model_path),
    }

    keras.backend.clear_session()
    return result


# ---------------------------------------------------------------------------
# Result files and plotting
# ---------------------------------------------------------------------------

def read_existing_results():
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def completed_key(row):
    return row["unfreeze_name"], int(row["fold"])


def append_result(row):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unfreeze_name",
        "unfreeze_label",
        "fold",
        "val_auc",
        "best_history_val_auc",
        "last_history_val_auc",
        "trainable_base_layers",
        "train_images",
        "val_images",
        "positive_class",
        "run_seed",
    ]
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_final_result(row):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unfreeze_name",
        "unfreeze_label",
        "val_auc",
        "test_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_tp",
        "test_tn",
        "test_fp",
        "test_fn",
        "best_history_val_auc",
        "last_history_val_auc",
        "trainable_base_layers",
        "total_params",
        "trainable_params",
        "non_trainable_params",
        "model_size_mb",
        "training_time_s",
        "avg_epoch_time_s",
        "test_inference_time_s",
        "test_ms_per_step",
        "epochs_ran",
        "train_images",
        "val_images",
        "test_images",
        "positive_class",
        "run_seed",
        "batch_size",
        "dropout_rate",
        "stage2_lr",
        "use_resnet_preprocess_input",
        "final_split_dir",
        "final_model_path",
    ]
    mode = "a"
    write_header = not FINAL_RESULTS_CSV.exists()
    if FINAL_RESULTS_CSV.exists():
        with FINAL_RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fieldnames:
            mode = "w"
            write_header = True

    with FINAL_RESULTS_CSV.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    existing = []
    if FINAL_RESULTS_JSON.exists():
        existing = json.loads(FINAL_RESULTS_JSON.read_text(encoding="utf-8"))
    existing.append(row)
    FINAL_RESULTS_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    write_final_tables(row)


def write_simple_markdown_table(path, headers, values):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_tables(row):
    model_name = (
        f"ResNet50-{row['unfreeze_label']} "
        f"(preprocess, dropout={row['dropout_rate']})"
    )

    performance_headers = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1"]
    performance_row = {
        "Model": model_name,
        "Accuracy": fmt_metric(row["test_accuracy"]),
        "AUC": fmt_metric(row["test_auc"]),
        "Precision": fmt_metric(row["test_precision"]),
        "Recall": fmt_metric(row["test_recall"]),
        "F1": fmt_metric(row["test_f1"]),
    }
    with FINAL_PERFORMANCE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=performance_headers)
        writer.writeheader()
        writer.writerow(performance_row)
    write_simple_markdown_table(
        FINAL_PERFORMANCE_MD,
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
    with FINAL_RESOURCE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resource_headers)
        writer.writeheader()
        writer.writerow(resource_row)
    write_simple_markdown_table(
        FINAL_RESOURCE_MD,
        resource_headers,
        [resource_row[header] for header in resource_headers],
    )


def load_results_for_summary():
    rows = read_existing_results()
    clean_rows = []
    for row in rows:
        clean_rows.append(
            {
                "unfreeze_name": row["unfreeze_name"],
                "unfreeze_label": row["unfreeze_label"],
                "fold": int(row["fold"]),
                "val_auc": float(row["val_auc"]),
                "trainable_base_layers": int(row["trainable_base_layers"]),
            }
        )
    return clean_rows


def summarize_results(rows):
    by_setting = {}
    for row in rows:
        by_setting.setdefault(row["unfreeze_name"], []).append(row)

    summaries = []
    for setting in UNFREEZE_LEVELS:
        setting_rows = sorted(by_setting.get(setting["name"], []), key=lambda r: r["fold"])
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
    return summaries


def save_summary(summaries):
    if not summaries:
        return
    fieldnames = [
        "unfreeze_name",
        "unfreeze_label",
        "folds_completed",
        "mean_val_auc",
        "std_val_auc",
        "min_val_auc",
        "max_val_auc",
        "trainable_base_layers",
        "description",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    SUMMARY_JSON.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def plot_results(rows, summaries):
    if not rows or not summaries:
        print("No results available for plotting yet.")
        return

    label_by_name = {setting["name"]: setting["label"] for setting in UNFREEZE_LEVELS}
    ordered_names = [item["unfreeze_name"] for item in summaries]
    x_positions = np.arange(len(ordered_names))

    plt.figure(figsize=(11, 6))

    for x, name in zip(x_positions, ordered_names):
        setting_rows = sorted(
            [row for row in rows if row["unfreeze_name"] == name],
            key=lambda row: row["fold"],
        )
        aucs = np.array([row["val_auc"] for row in setting_rows], dtype=float)
        folds = [row["fold"] for row in setting_rows]
        if len(aucs) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.14, 0.14, len(aucs))

        plt.scatter(
            np.full(len(aucs), x) + jitter,
            aucs,
            s=54,
            alpha=0.85,
            color="#3662a3",
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

    mean_values = [item["mean_val_auc"] for item in summaries]
    std_values = [item["std_val_auc"] for item in summaries]
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

    plt.xticks(x_positions, [label_by_name[name] for name in ordered_names])
    plt.ylabel("Validation AUC")
    plt.xlabel("ResNet50 unfreezing setting")
    plt.title("ResNet50 unfreezing level selection by 5-fold validation AUC")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


def run_data_check():
    if FINAL_TRAIN:
        for split_name in ["final_train", "final_val", "final_test"]:
            image_csv = FINAL_SPLIT_DIR / f"{split_name}_images.csv"
            records = read_image_csv(image_csv)
            verify_records_exist(records, split_name)
        return

    for fold in FOLDS:
        train_csv = FOLDS_DIR / f"fold_{fold}_train_images.csv"
        val_csv = FOLDS_DIR / f"fold_{fold}_val_images.csv"
        train_records = limit_records(read_image_csv(train_csv), MAX_TRAIN_SAMPLES)
        val_records = limit_records(read_image_csv(val_csv), MAX_VAL_SAMPLES)
        verify_records_exist(train_records, f"fold {fold} train")
        verify_records_exist(val_records, f"fold {fold} val")


def main():
    args = parse_args()
    configure_from_args(args)
    configure_tensorflow_runtime()

    if DATA_CHECK_ONLY:
        run_data_check()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FINAL_TRAIN:
        final_settings = selected_unfreeze_levels()
        if len(final_settings) != 1:
            names = ", ".join(setting["name"] for setting in final_settings)
            raise ValueError(
                "Final training expects exactly one --settings value. "
                f"Got: {names}"
            )

        setting = final_settings[0]
        print("=" * 80)
        print(f"Run final training setting={setting['name']}")
        print(setting["description"])
        print(f"Final split dir: {FINAL_SPLIT_DIR}")
        result = run_final_training(setting)
        append_final_result(result)
        print(
            f"Finished final training {setting['name']}: "
            f"val_auc={result['val_auc']:.4f}, "
            f"test_auc={result['test_auc']:.4f}, "
            f"trainable_base_layers={result['trainable_base_layers']}"
        )
        print("\nSaved final results:", FINAL_RESULTS_CSV)
        print("Saved final results JSON:", FINAL_RESULTS_JSON)
        print("Saved final performance table:", FINAL_PERFORMANCE_CSV)
        print("Saved final resource table:", FINAL_RESOURCE_CSV)
        return

    existing_rows = read_existing_results()
    completed = {completed_key(row) for row in existing_rows}

    for setting in selected_unfreeze_levels():
        for fold in FOLDS:
            key = (setting["name"], fold)
            if SKIP_EXISTING_RESULTS and key in completed:
                print(f"Skip existing result: {setting['name']} fold {fold}")
                continue

            print("=" * 80)
            print(f"Run setting={setting['name']} fold={fold}")
            print(setting["description"])
            result = run_one_fold(setting, fold)
            append_result(result)
            completed.add(key)
            print(
                f"Finished {setting['name']} fold {fold}: "
                f"val_auc={result['val_auc']:.4f}, "
                f"trainable_base_layers={result['trainable_base_layers']}"
            )

    rows = load_results_for_summary()
    summaries = summarize_results(rows)
    save_summary(summaries)
    plot_results(rows, summaries)

    print("\nSaved fold AUC results:", RESULTS_CSV)
    print("Saved summary:", SUMMARY_CSV)
    print("Saved plot:", PLOT_PATH)

    if summaries:
        best = max(summaries, key=lambda item: item["mean_val_auc"])
        print(
            "Best mean validation AUC: "
            f"{best['unfreeze_label']} = {best['mean_val_auc']:.4f} "
            f"+/- {best['std_val_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
