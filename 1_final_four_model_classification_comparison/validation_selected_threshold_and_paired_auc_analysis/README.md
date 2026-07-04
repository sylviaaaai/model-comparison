# Validation-Selected Threshold and Paired AUC Analysis

This folder contains companion classification analyses for the four final
image-level models. Together, they report both threshold-dependent operating
point performance and threshold-free paired AUC comparisons.

The analyses are separated because they answer different questions:

- **Validation-selected threshold analysis**: after choosing one operating
  threshold per model on validation data only, how does each model perform on
  the final test set?
- **Paired AUC analysis**: when comparing models on the same final test
  patients, how large are the threshold-free AUC differences?

## Start Here

For a one-file reader-friendly summary, open:

```text
classification_threshold_auc_readable_tables.md
```

This file combines validation-selected thresholds, locked-threshold test
performance, fixed-0.5 comparator results, test AUC by model, and paired AUC
differences.

For more detail, read the two subfolder README files in this order:

1. `validation_selected_threshold/README.md`
2. `paired_auc_analysis/README.md`

## Folder Structure

```text
validation_selected_threshold_and_paired_auc_analysis/
├── README.md
├── classification_threshold_auc_readable_tables.md
├── results_readable_tables.md
├── validation_selected_threshold/
│   ├── README.md
│   └── results/
│       ├── classification_validation_selected_thresholds.csv
│       ├── classification_test_locked_threshold_performance.csv
│       ├── classification_test_fixed_0p5_performance.csv
│       ├── tableS_classification_validation_thresholds_readable.csv
│       ├── tableS_classification_test_locked_thresholds_readable.csv
│       └── tableS_classification_test_fixed_0p5_readable.csv
├── paired_auc_analysis/
│   ├── README.md
│   └── results/
│       ├── classification_test_auc_by_model.csv
│       ├── classification_test_auc_pairwise_bootstrap.csv
│       └── tableS_classification_paired_auc_readable.csv
└── tools/
    ├── analyze_classification_thresholds_and_auc.py
    └── build_paper_ready_classification_tables.py
```

## Table Reading Order

For manuscript or supervisor review, use the reader-ready files first:

| Step | File | What it shows |
| ---: | --- | --- |
| 1 | `classification_threshold_auc_readable_tables.md` | Combined Markdown summary of threshold and paired AUC results. |
| 2 | `validation_selected_threshold/results/tableS_classification_validation_thresholds_readable.csv` | The threshold selected for each model on the validation split. |
| 3 | `validation_selected_threshold/results/tableS_classification_test_locked_thresholds_readable.csv` | Main threshold-dependent final test result using validation-selected thresholds. |
| 4 | `validation_selected_threshold/results/tableS_classification_test_fixed_0p5_readable.csv` | Fixed 0.5 comparator on the same final test set. |
| 5 | `paired_auc_analysis/results/tableS_classification_paired_auc_readable.csv` | Pairwise paired-bootstrap AUC differences and interpretation. |

Use the longer machine-readable CSV files when exact numeric values, additional
columns, or reproducibility checks are needed.

## Validation-Selected Threshold Analysis

This analysis selects one threshold per model using validation predictions
only. The selected threshold is then locked and applied once to the final test
set.

Threshold selection rule:

1. Sweep candidate probability thresholds on the validation split.
2. Select the threshold that maximizes validation Youden's J.
3. Break ties by higher F1.
4. Break remaining ties by higher balanced accuracy.
5. If a tie remains, choose the lower threshold.

Main files:

| File | Meaning |
| --- | --- |
| `validation_selected_threshold/results/classification_validation_selected_thresholds.csv` | Full validation threshold-selection output for each model. |
| `validation_selected_threshold/results/classification_test_locked_threshold_performance.csv` | Final test performance using validation-selected thresholds. |
| `validation_selected_threshold/results/classification_test_fixed_0p5_performance.csv` | Final test performance using a fixed threshold of 0.5 for every model. |

The locked-threshold table is the primary threshold-dependent test result. The
fixed-0.5 table is included as a comparator.

## Paired AUC Analysis

This analysis compares threshold-free test AUCs between models using paired
patient-level bootstrap resampling. For each bootstrap draw, patients are
sampled with replacement and the same sampled patients are used for both models
in a pair.

The paired difference is:

```text
delta AUC = AUC(model_a) - AUC(model_b)
```

A positive delta favors `model_a`; a negative delta favors `model_b`. A
confidence interval crossing 0 should be interpreted conservatively.

Main files:

| File | Meaning |
| --- | --- |
| `paired_auc_analysis/results/classification_test_auc_by_model.csv` | Test AUC and patient-level bootstrap 95% CI for each model. |
| `paired_auc_analysis/results/classification_test_auc_pairwise_bootstrap.csv` | Pairwise AUC differences with bootstrap 95% CIs. |
| `paired_auc_analysis/results/tableS_classification_paired_auc_readable.csv` | Reader-ready paired AUC comparison table. |

## Result Snapshot

| Model | Validation-selected threshold | Locked test AUC | Locked test accuracy | Locked test F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | 0.607 | 0.701 | 0.655 | 0.615 |
| ViT-B/16 | 0.558 | 0.819 | 0.779 | 0.754 |
| ConvNeXt-Base | 0.693 | 0.791 | 0.759 | 0.727 |
| Swin-Base | 0.461 | 0.781 | 0.752 | 0.717 |

The paired AUC analysis supports a clear AUC gap between ResNet50 and the three
stronger models. Differences among ViT-B/16, ConvNeXt-Base, and Swin-Base are
smaller, and their paired-bootstrap confidence intervals cross 0.

## Notes

- The final test set contains 290 images from 135 patients.
- The final test class distribution is 147 malignant and 143 benign images.
- Test-set confidence intervals use patient-level bootstrap resampling.
- Values in README snapshots are rounded for readability; CSV files retain the
  exact numeric results.
