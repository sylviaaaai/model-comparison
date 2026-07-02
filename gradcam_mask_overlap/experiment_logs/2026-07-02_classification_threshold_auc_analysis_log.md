# Experiment Log: Classification Threshold and Paired AUC Analysis

Date: 2026-07-02
Workspace: `/mnt/e/SW整合`

## Purpose

This analysis is a classification companion to the Grad-CAM / mask-overlap
analysis. It answers two separate questions:

```text
1. If thresholds are selected on the final validation set, how do the locked
   thresholds perform on the final test set?
2. On the final test set, are model AUC differences robust under paired
   patient-level bootstrap resampling?
```

Important design rule:

```text
Threshold selection used validation predictions only.
Final conclusions are reported on the test set only.
```

This analysis does not replace the overlap/localization analysis. It uses all
final test images for classification performance, not only
`usable_for_overlap = yes`.

## Inputs

Validation split:

```text
results/splits/final_80_10_10/final_val_images.csv
n = 280 images
```

Test split:

```text
results/splits/final_80_10_10/final_test_images.csv
n = 290 images
```

Existing test prediction CSVs:

```text
results/patient_level_bootstrap_ci/resnet50_test_predictions.csv
results/patient_level_bootstrap_ci/vit_b16_test_predictions.csv
results/patient_level_bootstrap_ci/convnext_base_test_predictions.csv
results/patient_level_bootstrap_ci/swin_base_test_predictions.csv
```

Validation prediction CSVs generated in this run:

```text
gradcam_mask_overlap/classification_analysis/validation_predictions_raw/resnet50_test_predictions.csv
gradcam_mask_overlap/classification_analysis/validation_predictions_raw/vit_b16_test_predictions.csv
gradcam_mask_overlap/classification_analysis/validation_predictions_raw/convnext_base_test_predictions.csv
gradcam_mask_overlap/classification_analysis/validation_predictions_raw/swin_base_test_predictions.csv
```

Note:

```text
The validation filenames retain the source export script's
*_test_predictions.csv suffix. Their directory and downstream split column
identify them as validation predictions.
```

## Validation Prediction Export

PyTorch models were exported with CPU inference using:

```text
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/matplotlib \
/mnt/e/SW整合/.venv-torch5090/bin/python tools/patient_level_bootstrap_ci.py \
  --split-csv results/splits/final_80_10_10/final_val_images.csv \
  --output-dir gradcam_mask_overlap/classification_analysis/validation_predictions_raw \
  --n-bootstrap 10 \
  --seed 123 \
  --num-workers 0 \
  --device cpu \
  --models vit_b16 convnext_base swin_base \
  --overwrite-predictions
```

ResNet50 was exported separately with TensorFlow CPU using:

```text
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/matplotlib TF_CPP_MIN_LOG_LEVEL=2 \
/mnt/e/SW整合/.venv-wsl/bin/python tools/export_resnet50_final_predictions.py \
  --split-csv results/splits/final_80_10_10/final_val_images.csv \
  --output-csv gradcam_mask_overlap/classification_analysis/validation_predictions_raw/resnet50_test_predictions.csv \
  --batch-size 16 \
  --seed 123
```

GPU note:

```text
No GPU was used. TensorFlow printed a CUDA initialization warning but reported
TensorFlow GPU devices: none.
```

## Main Analysis Script

Script added:

```text
gradcam_mask_overlap/tools/analyze_classification_thresholds_and_auc.py
```

Command:

```text
/mnt/e/SW整合/.venv-torch5090/bin/python \
gradcam_mask_overlap/tools/analyze_classification_thresholds_and_auc.py \
  --n-bootstrap 5000 \
  --seed 20260702
```

Validation threshold selection:

```text
For each model, sweep observed validation probability thresholds plus
0.0, 0.5, and 1.0.

Select threshold by:
  max validation Youden's J
  tie-break 1: higher F1
  tie-break 2: higher balanced accuracy
  tie-break 3: lower threshold
```

Test evaluation:

```text
Apply the selected threshold unchanged to test predictions.
Also report fixed-threshold 0.5 as a comparator.
```

Bootstrap:

```text
n_bootstrap = 5000
seed = 20260702
resampling unit = patient_id
AUC pairwise comparison = paired bootstrap on the same resampled patients
```

## Outputs

Output folder:

```text
gradcam_mask_overlap/classification_analysis/threshold_auc/
```

Generated files:

```text
classification_validation_predictions_long.csv
classification_test_predictions_long.csv
classification_validation_threshold_sweep.csv
classification_validation_selected_thresholds.csv
classification_test_locked_threshold_performance.csv
classification_test_fixed_0p5_performance.csv
classification_test_auc_by_model.csv
classification_test_auc_pairwise_bootstrap.csv
classification_threshold_auc_summary.md
classification_threshold_auc_summary.json
```

Paper-readable tables generated after the first run:

```text
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_readable_tables.md
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_validation_thresholds_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_test_locked_thresholds_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_test_fixed_0p5_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_paired_auc_readable.csv
```

Readable-table script:

```text
gradcam_mask_overlap/tools/build_paper_ready_classification_tables.py
```

## How To Read The Tables

`classification_validation_threshold_sweep.csv`

```text
One row per model x validation threshold.
Use this only to audit how threshold selection behaved.
Do not use it for final test conclusions.
```

`classification_validation_selected_thresholds.csv`

```text
One row per model.
Records the validation-selected threshold and validation performance at that
threshold.
This is the source for locked test thresholds.
```

`classification_test_locked_threshold_performance.csv`

```text
One row per model on the final test set.
Uses the validation-selected threshold.
This is the main threshold-dependent classification table.
```

`classification_test_fixed_0p5_performance.csv`

```text
One row per model on the final test set.
Uses threshold = 0.5 for comparison with the original/default operating point.
```

`classification_test_auc_by_model.csv`

```text
One row per model.
Reports threshold-free test AUC and patient-level bootstrap 95% CI.
```

`classification_test_auc_pairwise_bootstrap.csv`

```text
One row per model pair.
delta_auc_a_minus_b = test AUC(model_a) - test AUC(model_b).
Positive delta favors model_a.
CI crossing 0 means the paired AUC difference should be interpreted
conservatively.
```

## Key Results

Validation-selected thresholds:

| Model | Threshold | Validation AUC | Validation sensitivity | Validation specificity | Validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 0.6070 | 0.774 | 0.658 | 0.806 | 0.716 |
| ViT-B/16 | 0.5581 | 0.834 | 0.678 | 0.888 | 0.762 |
| ConvNeXt-Base | 0.6931 | 0.812 | 0.651 | 0.896 | 0.745 |
| Swin-Base | 0.4614 | 0.803 | 0.685 | 0.836 | 0.746 |

Locked-threshold test performance:

| Model | Threshold | Test AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) |
| --- | ---: | --- | --- | --- | --- | --- |
| ResNet50 | 0.6070 | 0.701 [0.623, 0.777] | 0.655 [0.590, 0.724] | 0.544 [0.442, 0.645] | 0.769 [0.686, 0.851] | 0.615 [0.522, 0.698] |
| ViT-B/16 | 0.5581 | 0.819 [0.758, 0.877] | 0.779 [0.718, 0.837] | 0.667 [0.567, 0.761] | 0.895 [0.841, 0.942] | 0.754 [0.675, 0.820] |
| ConvNeXt-Base | 0.6931 | 0.791 [0.734, 0.845] | 0.759 [0.702, 0.813] | 0.633 [0.546, 0.719] | 0.888 [0.833, 0.939] | 0.727 [0.655, 0.790] |
| Swin-Base | 0.4614 | 0.781 [0.717, 0.846] | 0.752 [0.689, 0.812] | 0.619 [0.523, 0.717] | 0.888 [0.827, 0.943] | 0.717 [0.635, 0.789] |

Paired AUC comparison:

| Pair | Delta AUC | 95% CI | Interpretation |
| --- | ---: | --- | --- |
| ResNet50 - ViT-B/16 | -0.1175 | [-0.1834, -0.0559] | ResNet50 lower |
| ResNet50 - ConvNeXt-Base | -0.0895 | [-0.1633, -0.0222] | ResNet50 lower |
| ResNet50 - Swin-Base | -0.0798 | [-0.1441, -0.0200] | ResNet50 lower |
| ViT-B/16 - ConvNeXt-Base | +0.0280 | [-0.0098, +0.0635] | CI crosses 0 |
| ViT-B/16 - Swin-Base | +0.0377 | [-0.0060, +0.0794] | CI crosses 0 |
| ConvNeXt-Base - Swin-Base | +0.0097 | [-0.0282, +0.0494] | CI crosses 0 |

## Interpretation

The classification result is not identical to the localization result:

```text
Classification AUC:
  ViT-B/16 is numerically highest on the final test set.
  ConvNeXt-Base and Swin-Base are close behind.
  ResNet50 is clearly lower than the other three under paired bootstrap.

Overlap/localization:
  ConvNeXt-Base remains the strongest model for mask-overlap localization.
```

Therefore the manuscript should separate these claims:

```text
ViT-B/16 had the highest numerical test AUC, although paired AUC differences
among ViT-B/16, ConvNeXt-Base, and Swin-Base crossed zero.

ConvNeXt-Base showed the strongest lesion localization / mask-overlap behavior.
```

Locked validation-selected thresholds made the operating points more specific
for ViT-B/16, ConvNeXt-Base, and Swin-Base, but sensitivity remained moderate.
This should be reported as operating-point behavior, not as a replacement for
threshold-free AUC.

## Conclusion For Manuscript Planning

Use this analysis as a classification-performance companion table or
supplementary table. It is useful because it shows the classification ranking
and proves that thresholds were not tuned on the test set.

Suggested framing:

```text
Primary classification performance was assessed by test AUC with patient-level
paired bootstrap comparisons. Operating-point metrics were reported using
thresholds selected only on the validation set by Youden's J and locked before
test-set evaluation.
```
