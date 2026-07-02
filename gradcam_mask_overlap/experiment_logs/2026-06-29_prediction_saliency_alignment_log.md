# Experiment Log: Prediction and Saliency Alignment Before Overlap

Date: 2026-06-29
Workspace: `/mnt/e/SW整合`

## Execution Environment

The prediction/saliency alignment work used the project Torch environment:

```text
.venv-torch5090/bin/python = Python 3.11.15
torch = 2.11.0+cu128
torchvision = 0.26.0+cu128
CUDA build = 12.8
timm = 1.0.27
numpy = 2.4.4
Pillow = 12.2.0
```

GPU details when elevated execution is allowed:

```text
GPU = NVIDIA GeForce RTX 5090
driver = 596.49
memory = 32607 MiB
```

Default sandbox note:

```text
GPU/NVML access is blocked in normal sandboxed commands. GPU jobs require an
elevated execution request. The full Swin and ViT recomputation/export jobs were
run with elevated GPU access.
```

## Goal

Before computing lesion-mask overlap metrics, re-check that all final model
prediction files, patient-level bootstrap CI tables, saliency maps, and
analysis-ready metadata refer to the same model checkpoints and inference
outputs.

## Prediction Cache Checks

The four final prediction CSV files were checked against
`gradcam_mask_overlap/metadata/model_predictions.csv`.

| Model | Rows | Metadata rows | Max probability diff | Label mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 290 | 290 | 0.0 | 0 | 0 |
| ViT-B/16 | 290 | 290 | 0.0 | 0 | 0 |
| ConvNeXt-Base | 290 | 290 | 0.0 | 0 | 0 |
| Swin-Base | 290 | 290 | 0.0 | 0 | 0 |

Stale prediction backups kept for traceability:

```text
results/patient_level_bootstrap_ci/swin_base_test_predictions.stale_20260629.csv
results/patient_level_bootstrap_ci/vit_b16_test_predictions.stale_20260629.csv
```

## Swin Issue and Resolution

Swin initially showed a small but real mismatch between the old prediction CSV
and probabilities recomputed from the current `final_model.pt`.

Resolution:

1. The old Swin prediction CSV was backed up.
2. Swin test predictions were regenerated from the current final checkpoint.
3. Patient-level bootstrap CI was rerun.
4. The model-comparison README and CI summary tables were updated.
5. `export_swin_raw_gradcam.py` was aligned with the original training script
   model builder and transform pipeline.

Current Swin probability agreement in the saliency manifest:

```text
max abs(source_cancer_probability - recomputed_cancer_probability) = 1.08e-6
```

## ViT Issue and Resolution

ViT had a smaller mismatch:

```text
old cache vs current final_model.pt:
max abs diff  = 0.0004709363
mean abs diff = 0.0000496114
prediction mismatches = 0
```

This was traced to the historical prediction cache being numerically different
from the current checkpoint/inference path, not to an incorrect model path,
dropout setting, split order, or preprocessing mismatch.

Resolution:

1. The old ViT prediction CSV was backed up.
2. ViT test predictions were regenerated from the current
   `VIT/results/vit_b16_torch_final_all_blocks_dropout04/final/all_blocks/final_model.pt`.
3. Patient-level bootstrap CI was rerun.
4. `model_predictions.csv`, `saliency_manifest.csv`,
   `analysis_ready_overlap_manifest.csv`, the CI summary tables, and
   `model-comparison-repo/README.md` were updated.
5. Standard ViT Attention Rollout was selected for overlap:

```text
method = rollout
discard_ratio = 0.0
percentile = 99.0
```

The sharper `attention_rollout_magma_discard09` results are retained as
historical/auxiliary visualization outputs, but they are not the current overlap
analysis path.

Current ViT probability agreement in the saliency manifest:

```text
max abs(source_cancer_probability - recomputed_cancer_probability) = 1.23e-6
```

## Explainability Methods in the Current Analysis

| Model | Main saliency used for overlap | Rows | Raw map status |
| --- | --- | ---: | --- |
| ResNet50 | `gradcam_malignant` | 290 | `raw_map_available` |
| ConvNeXt-Base | `gradcam_malignant` | 290 | `raw_map_available` |
| Swin-Base | `gradcam_malignant` | 290 | `raw_map_available` |
| ViT-B/16 | `attention_rollout` | 290 | `raw_map_available` |

Predicted-class Grad-CAM maps are also available for ResNet50, ConvNeXt, and
Swin:

```text
resnet50:gradcam = 290
convnext_base:gradcam = 290
swin_base:gradcam = 290
```

## Raw Saliency Map Audit

All exported raw maps are normalized 224 x 224 maps with matching PNG previews.

| Saliency folder | NPY | PNG |
| --- | ---: | ---: |
| `resnet50/gradcam_predclass` | 290 | 290 |
| `resnet50/gradcam_malignant` | 290 | 290 |
| `convnext_base/gradcam_predclass` | 290 | 290 |
| `convnext_base/gradcam_malignant` | 290 | 290 |
| `swin_base/gradcam_predclass` | 290 | 290 |
| `swin_base/gradcam_malignant` | 290 | 290 |
| `vit_b16/attention_rollout_predclass` | 290 | 290 |

Final saliency probability agreement:

| Model | Saliency rows | Max probability diff |
| --- | ---: | ---: |
| ResNet50 | 580 | 4.7e-7 |
| ViT-B/16 | 290 | 1.23e-6 |
| ConvNeXt-Base | 580 | 2.17e-5 |
| Swin-Base | 580 | 1.08e-6 |

## Analysis-Ready Metadata

`gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv` now has:

```text
analysis_manifest_rows = 2030
ready_for_overlap_rows = 2030
saliency_status_counts = {"raw_map_available": 2030}
```

Rows by model:

```text
resnet50 = 580
convnext_base = 580
swin_base = 580
vit_b16 = 290
```

Ready rows by method:

```text
resnet50:gradcam = 290
resnet50:gradcam_malignant = 290
convnext_base:gradcam = 290
convnext_base:gradcam_malignant = 290
swin_base:gradcam = 290
swin_base:gradcam_malignant = 290
vit_b16:attention_rollout = 290
```

## Current CI Table

The local CI summary and the GitHub summary table match.

| Model | AUC (95% CI) | Accuracy | TP/TN/FP/FN |
| --- | --- | ---: | --- |
| ResNet50 | 0.7014 [0.6239, 0.7769] | 0.6241 | 89/92/51/58 |
| ViT-B/16 | 0.8190 [0.7546, 0.8775] | 0.7759 | 100/125/18/47 |
| ConvNeXt-Base | 0.7910 [0.7298, 0.8468] | 0.7276 | 103/108/35/44 |
| Swin-Base | 0.7813 [0.7101, 0.8450] | 0.7621 | 90/131/12/57 |

## Remaining Work

`gradcam_mask_overlap/metadata/overlap_metrics.csv` still contains only the
header. The next step is to compute overlap metrics from the 2030 ready rows:

- IoU
- Dice
- CAM energy inside lesion mask
- mask coverage by CAM
- CAM peak location and whether the peak lies inside the mask
- threshold-specific overlap summaries

Recommended primary comparison:

```text
ResNet50 / ConvNeXt / Swin: gradcam_malignant
ViT-B/16: attention_rollout
```
