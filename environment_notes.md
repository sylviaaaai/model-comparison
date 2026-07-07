# Environment Notes

This repository is a cleaned final-analysis package for manuscript review, not
a single-command reproduction package. The package list in `requirements.txt`
records the main Python dependencies, while this file records the experimental
environment information that is relevant for interpreting the results and
resource tables.

## Quick Environment Summary

Python version: 3.11.15 in both recorded framework environments.

CUDA / GPU: CUDA-enabled local workstation; NVIDIA GeForce RTX 5090 with
32607 MiB GPU memory reported by `nvidia-smi`. PyTorch reported CUDA 12.8 via
`torch` 2.11.0+cu128.

TensorFlow version: 2.20.0 in the TensorFlow/Keras environment.

PyTorch version: 2.11.0+cu128 in the PyTorch/timm environment.

timm version: 1.0.27 in the PyTorch/timm environment.

scikit-learn version: 1.8.0 in both recorded framework environments.

OpenCV / Pillow version: OpenCV (`opencv-python`) was listed as a project
dependency but was not installed or not detected in the recorded framework
environments during this audit. Pillow was 12.2.0 in both recorded framework
environments.

## Recorded Experimental Hardware

| Item | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU memory | 32607 MiB reported by `nvidia-smi` |
| NVIDIA driver | 596.49 |
| CUDA / GPU execution | CUDA-enabled local workstation runs were used for model training, final evaluation, saliency export, and resource timing. |
| Resource timing scope | Resource tables should be interpreted as within-project comparisons under the shared RTX 5090 environment, not as hardware-independent runtime benchmarks. |

The four-model resource table is:

```text
1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/final_resource_comparison.csv
```

## Python Package Environment

The project used separate framework environments for the TensorFlow/Keras and
PyTorch/timm workflows. The versions below were read from the local virtual
environments available with this project package.

### PyTorch / timm Environment

Environment path:

```text
/mnt/e/SW整合/.venv-torch5090
```

This environment was used for the PyTorch/timm models and saliency export
workflows, including ViT-B/16, ConvNeXt-Base, and Swin-Base.

| Component | Recorded version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| timm | 1.0.27 |
| scikit-learn | 1.8.0 |
| Pillow | 12.2.0 |
| NumPy | 2.4.4 |
| pandas | 3.0.3 |
| matplotlib | 3.10.9 |
| TensorFlow | Not installed in this environment |
| OpenCV (`opencv-python`) | Not installed or not detected in this environment |

### TensorFlow / Keras Environment

Environment path:

```text
/mnt/e/SW整合/.venv-wsl
```

This environment was used for the TensorFlow/Keras workflow, including the
ResNet50 model.

| Component | Recorded version |
| --- | --- |
| Python | 3.11.15 |
| TensorFlow | 2.20.0 |
| scikit-learn | 1.8.0 |
| Pillow | 12.2.0 |
| NumPy | 2.4.6 |
| matplotlib | 3.10.9 |
| PyTorch / torchvision / timm | Not installed in this environment |
| pandas | Not installed or not detected in this environment |
| OpenCV (`opencv-python`) | Not installed or not detected in this environment |

These are the framework-level environments that were available during this
documentation audit. If a manuscript requires a stricter archival record, keep
the generated package freeze files from these environments with the submitted
code supplement.

## Minimum Dependency List

The unpinned dependency list is kept in:

```text
requirements.txt
```

It records the main libraries needed to audit or rerun scripts:

```text
numpy
pandas
scikit-learn
matplotlib
Pillow
torch
torchvision
timm
tensorflow
opencv-python
```

OpenCV (`opencv-python`) is included because some image-processing,
mask-processing, and saliency-visualization utilities may depend on it.

## How To Pin The Original Environment

To create stricter archival records, run the following from each virtual
environment and save the outputs as supplements:

```bash
python --version
python -m pip freeze > requirements_frozen_ENV_NAME.txt
python - <<'PY'
import importlib.metadata as m
packages = [
    "tensorflow",
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "opencv-python",
    "Pillow",
    "numpy",
    "pandas",
    "matplotlib",
]
for package in packages:
    try:
        print(f"{package}=={m.version(package)}")
    except Exception:
        print(f"{package}: not installed or not detected")
PY
```

For GPU/CUDA reporting, also record:

```bash
nvidia-smi
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", exc)

try:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
    print("tensorflow gpus:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("tensorflow check failed:", exc)
PY
```

## Current Audit Notes

During this documentation update, the default active shell environment was not
the original training environment:

```text
default python3: 3.14.4
default shell project dependencies: not installed
```

The framework versions above were therefore read explicitly from:

```text
/mnt/e/SW整合/.venv-torch5090/bin/python
/mnt/e/SW整合/.venv-wsl/bin/python
```

The GPU information was read with `nvidia-smi` after sandbox approval.
