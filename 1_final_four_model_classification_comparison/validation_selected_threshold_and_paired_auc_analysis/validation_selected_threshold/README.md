# Validation-Selected Threshold Analysis

This folder contains the threshold-dependent classification analysis for the
four final image-level models. The purpose is to define each model's operating
threshold using validation data only, then apply that threshold once to the
independent final test set.

This analysis answers a different question from paired AUC comparison. AUC is
threshold-free, while the tables here report operating-point metrics such as
accuracy, sensitivity, specificity, precision, and F1 after a single threshold
has been selected.

## Reading Order

Start with the reader-ready tables:

```text
results/tableS_classification_validation_thresholds_readable.csv
results/tableS_classification_test_locked_thresholds_readable.csv
results/tableS_classification_test_fixed_0p5_readable.csv
```

Recommended order:

1. `tableS_classification_validation_thresholds_readable.csv`: shows the
   threshold selected for each model on the validation split.
2. `tableS_classification_test_locked_thresholds_readable.csv`: main
   threshold-dependent test result, using the validation-selected threshold.
3. `tableS_classification_test_fixed_0p5_readable.csv`: comparator using a
   fixed threshold of 0.5.
4. For exact numeric values and additional columns, use the corresponding
   machine-readable CSV files listed below.

For threshold-free AUC model comparison, read the sibling paired AUC section:

```text
../paired_auc_analysis/
```

## Result Files

| File | Meaning | Best use |
| --- | --- | --- |
| `results/tableS_classification_validation_thresholds_readable.csv` | Reader-friendly validation table with selected threshold, validation AUC, sensitivity, specificity, F1, and confusion counts. | Start here to see how each threshold was chosen. |
| `results/tableS_classification_test_locked_thresholds_readable.csv` | Reader-friendly final test table using each model's validation-selected threshold. Includes AUC, accuracy, sensitivity, specificity, precision, F1, and TP/TN/FP/FN. | Main threshold-dependent test result. |
| `results/tableS_classification_test_fixed_0p5_readable.csv` | Reader-friendly final test table using fixed threshold 0.5 for every model. | Comparator against the validation-selected operating point. |
| `results/classification_validation_selected_thresholds.csv` | Full validation threshold-selection output with exact thresholds and validation metrics. | Reproducibility and exact values. |
| `results/classification_test_locked_threshold_performance.csv` | Full final test output using validation-selected thresholds, including patient-level bootstrap 95% CIs. | Exact locked-threshold test statistics. |
| `results/classification_test_fixed_0p5_performance.csv` | Full final test output using fixed threshold 0.5, including patient-level bootstrap 95% CIs. | Exact fixed-threshold comparator statistics. |

## Experimental Procedure

The analysis used the final validation split to choose operating thresholds and
the final held-out test split to evaluate locked-threshold performance.

Validation threshold selection:

1. For each model, sweep candidate probability thresholds on validation
   predictions.
2. Compute threshold-dependent metrics at each candidate threshold.
3. Select the threshold that maximizes validation Youden's J:

```text
Youden's J = sensitivity + specificity - 1
```

4. If multiple thresholds tie, break ties by higher F1.
5. If still tied, break ties by higher balanced accuracy.
6. If still tied, choose the lower threshold.

After threshold selection, the chosen threshold was locked for each model and
applied to the independent final test split. The test set was not used to tune
or select the threshold.

Test-set uncertainty:

- Test metrics were estimated on 290 final test images from 135 patients.
- The test set contains 147 malignant and 143 benign images.
- Confidence intervals in the test tables use patient-level bootstrap
  resampling with 5000 valid bootstrap draws.
- Patient-level resampling preserves the grouping of images within patients.

## How to Read the Tables

- `Threshold` is the probability cutoff used to convert predicted malignant
  probability into a binary class label.
- `AUC` is threshold-free and therefore is unchanged between the
  locked-threshold and fixed-0.5 test tables.
- `Accuracy`, `sensitivity`, `specificity`, `precision`, `NPV`, `F1`,
  `balanced_accuracy`, and `Youden's J` depend on the chosen threshold.
- `TP/TN/FP/FN` reports image-level confusion counts.
- In the locked-threshold table, `threshold_source` is
  `validation_selected_youden_j`, meaning the cutoff came from validation-set
  threshold selection.
- In the fixed-threshold table, `threshold_source` is `fixed_0.5`, meaning all
  models used the same probability cutoff of 0.5.

## Result Snapshot

| Model | Validation-selected threshold | Locked test AUC | Locked test accuracy | Sensitivity | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 0.607 | 0.701 | 0.655 | 0.544 | 0.769 | 0.615 |
| ViT-B/16 | 0.558 | 0.819 | 0.779 | 0.667 | 0.895 | 0.754 |
| ConvNeXt-Base | 0.693 | 0.791 | 0.759 | 0.633 | 0.888 | 0.727 |
| Swin-Base | 0.461 | 0.781 | 0.752 | 0.619 | 0.888 | 0.717 |

## Interpretation

The locked-threshold results keep all threshold tuning on the validation set.
ViT-B/16 has the highest locked-threshold test AUC, accuracy, and F1 in this
analysis. ConvNeXt-Base and Swin-Base are close behind, while ResNet50 has lower
test AUC and lower sensitivity at its validation-selected operating point.

The fixed 0.5 table is included as a comparator, not as the primary
threshold-dependent result. It shows how performance changes if all models are
forced to use the same default cutoff instead of model-specific thresholds
selected on validation data.

For model-to-model threshold-free AUC differences and paired bootstrap
confidence intervals, use:

```text
../paired_auc_analysis/
```
