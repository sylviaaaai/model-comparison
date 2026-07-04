# Paired AUC Analysis

This folder contains the paired test-set AUC comparison for the four final
classification models. The goal is to compare model AUCs on the same final test
patients while preserving the paired structure of the data.

Unlike threshold-dependent metrics, AUC does not depend on choosing a single
classification threshold. This analysis is therefore the main place to compare
threshold-free discrimination between models.

## Reading Order

Start with the reader-ready table:

```text
results/tableS_classification_paired_auc_readable.csv
```

Then use the supporting machine-readable files if more detail is needed:

```text
results/classification_test_auc_by_model.csv
results/classification_test_auc_pairwise_bootstrap.csv
```

Recommended order:

1. `tableS_classification_paired_auc_readable.csv`: quick interpretation of all
   pairwise AUC differences.
2. `classification_test_auc_by_model.csv`: per-model test AUCs and bootstrap
   confidence intervals.
3. `classification_test_auc_pairwise_bootstrap.csv`: full pairwise bootstrap
   output with exact numeric values.

## Result Files

| File | Meaning | Best use |
| --- | --- | --- |
| `results/tableS_classification_paired_auc_readable.csv` | Reader-friendly pairwise AUC comparison table. It includes AUC for model A, AUC for model B, the paired AUC difference, 95% CI, and plain-language interpretation. | Start here for manuscript review or supervisor discussion. |
| `results/classification_test_auc_by_model.csv` | One row per model. Reports final test AUC, patient-level bootstrap 95% CI, number of images, number of patients, and class counts. | Use when reporting each model's absolute test AUC. |
| `results/classification_test_auc_pairwise_bootstrap.csv` | One row per model pair. Reports exact AUC values, delta AUC as `AUC(model_a) - AUC(model_b)`, 95% bootstrap CI for the delta, and bootstrap count. | Use for exact pairwise statistics and reproducibility checks. |

## Experimental Procedure

The analysis uses the final held-out test set:

- 290 test images
- 135 patients
- 147 malignant images
- 143 benign images

For each model, the final test-set prediction scores were aligned by patient
and image. AUC was computed from the malignant-class probability scores.

Paired uncertainty was estimated with patient-level bootstrap resampling:

1. Sample patients with replacement from the final test patient set.
2. Include all test images belonging to the sampled patients.
3. For the sampled set, compute AUC for both models in a model pair.
4. Compute the paired difference:

```text
delta AUC = AUC(model_a) - AUC(model_b)
```

5. Repeat the bootstrap procedure 5000 valid times.
6. Use the bootstrap distribution of delta AUC to estimate the 95% confidence
   interval.

Patient-level resampling is used because images from the same patient are not
fully independent. Keeping the same sampled patients for both models preserves
the paired comparison.

## How to Read the Pairwise Table

In `classification_test_auc_pairwise_bootstrap.csv`:

- `model_a_display` and `model_b_display` define the comparison.
- `auc_a` and `auc_b` are the test AUCs for those two models.
- `delta_auc_a_minus_b` is `auc_a - auc_b`.
- A positive delta favors `model_a`.
- A negative delta favors `model_b`.
- `delta_auc_bootstrap_ci95_low` and `delta_auc_bootstrap_ci95_high` are the
  patient-level bootstrap 95% CI for the paired difference.
- If the CI crosses 0, the observed AUC difference should be interpreted
  conservatively.

## Result Snapshot

| Comparison | Delta AUC, model A - model B (95% CI) | Interpretation |
| --- | ---: | --- |
| ResNet50 - ViT-B/16 | -0.118 [-0.183, -0.056] | ViT-B/16 higher |
| ResNet50 - ConvNeXt-Base | -0.090 [-0.163, -0.022] | ConvNeXt-Base higher |
| ResNet50 - Swin-Base | -0.080 [-0.144, -0.020] | Swin-Base higher |
| ViT-B/16 - ConvNeXt-Base | 0.028 [-0.010, 0.064] | ViT-B/16 numerically higher; CI crosses 0 |
| ViT-B/16 - Swin-Base | 0.038 [-0.006, 0.079] | ViT-B/16 numerically higher; CI crosses 0 |
| ConvNeXt-Base - Swin-Base | 0.010 [-0.028, 0.049] | ConvNeXt-Base numerically higher; CI crosses 0 |

## Interpretation

The paired bootstrap supports a clear AUC gap between ResNet50 and the three
stronger models. The confidence intervals for ResNet50 versus ViT-B/16,
ConvNeXt-Base, and Swin-Base are entirely below 0 when the comparison is written
as `ResNet50 - other model`, indicating that the other model has higher AUC.

Among ViT-B/16, ConvNeXt-Base, and Swin-Base, the point estimates favor
ViT-B/16, followed by ConvNeXt-Base and Swin-Base. However, the paired
bootstrap confidence intervals for these differences cross 0, so these
differences should be described as numerical rather than clearly separated.

## Relationship to Threshold Analysis

This paired AUC analysis is threshold-free. It should be read together with the
validation-selected threshold analysis in:

```text
../validation_selected_threshold/
```

The threshold analysis reports operating-point metrics such as accuracy,
sensitivity, specificity, precision, and F1 after thresholds are selected on
validation data. The paired AUC analysis here answers a different question:
whether the models differ in test-set ranking performance independent of any
single threshold.
