# Classification Results

This folder collects the classification-performance tables needed for the
manuscript.

## What This Covers

```text
1. Original four-model classification comparison
2. Patient-level bootstrap CI for final test AUC
3. Validation-selected operating thresholds
4. Locked-threshold final test performance
5. Paired final test AUC comparison
6. Resource comparison
```

## First Tables To Read

```text
tables/best_final_models_comparison.md
tables/image_level_patient_bootstrap_ci_summary.md
tables/tableS_classification_readable_tables.md
```

## Main Interpretation

```text
ViT-B/16 has the highest numerical final test AUC.
ConvNeXt-Base and Swin-Base are close behind.
ResNet50 is lower than the other three under paired patient-level bootstrap.
```

Use conservative wording for ViT-B/16 vs ConvNeXt-Base vs Swin-Base because
their paired AUC differences cross 0.

## Table Guide

```text
best_final_models_comparison.*
  Original final selected model comparison.

image_level_patient_bootstrap_ci_summary.*
  Final test metrics with patient-level bootstrap confidence intervals.

tableS_classification_readable_tables.md
  Human-readable combined table for validation thresholds, locked test
  performance, fixed 0.5 performance, and paired AUC differences.

tableS_classification_validation_thresholds_readable.csv
  Model-specific thresholds selected on the validation set.

tableS_classification_test_locked_thresholds_readable.csv
  Final test operating-point metrics using validation-selected thresholds.

tableS_classification_test_fixed_0p5_readable.csv
  Fixed 0.5 threshold comparator.

tableS_classification_paired_auc_readable.csv
  Paired patient-level bootstrap AUC differences.

final_resource_comparison.*
  Model size and inference/resource comparison.
```

## Methods Note

For operating-point analysis, model-specific thresholds were selected on the
validation set by maximizing Youden's J and then locked before evaluation on
the independent final test set.

