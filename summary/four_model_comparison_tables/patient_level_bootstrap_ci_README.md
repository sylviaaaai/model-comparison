# Patient-Level Bootstrap 95% Confidence Intervals

This folder includes an additional final comparison table with 95% confidence
intervals:

- `image_level_patient_bootstrap_ci_summary.csv`
- `image_level_patient_bootstrap_ci_summary.md`

## Reporting Target

The reported performance is **image-level performance with patient-level
bootstrap confidence intervals**.

This means each metric is computed over test images, matching the original
final test evaluation tables, but the bootstrap resampling unit is the patient
rather than the individual image. This is important because one patient can
contribute multiple images, so treating every image as fully independent would
underestimate uncertainty.

## Test Set

The final test split contains:

- 290 test images
- 135 patients
- 147 Cancer images
- 143 Benign images

All four final models were evaluated on the same final test split.

## Final Models

The confidence interval table uses the same final model selections as the main
four-model comparison:

| Model | Final setting |
| --- | --- |
| ResNet50 | `conv3_5`, ResNet preprocessing, dropout `0.4`, batch size `16` |
| ViT-B/16 | `all_blocks`, dropout `0.4`, batch size `16` |
| ConvNeXt-Base | `stage3_4`, dropout `0.4`, batch size `32` |
| Swin-Base | `stage2_4`, dropout `0.4`, batch size `16` |

## Bootstrap Procedure

For each final model:

1. The saved final model was used to generate a full final-test prediction file
   with one row per image.
2. Each prediction row contained the image path, `patient_id`, true label,
   predicted cancer probability, and thresholded prediction.
3. Bootstrap resampling was performed at the `patient_id` level.
4. In each bootstrap replicate, 135 patients were sampled with replacement.
5. When a patient was sampled, all images from that patient were included in
   the replicate.
6. Image-level metrics were recomputed on the resampled image set.
7. This was repeated for 2000 bootstrap replicates using seed `123`.
8. The 95% confidence interval was computed with the percentile method, using
   the 2.5th and 97.5th percentiles of the bootstrap distribution.

The classification threshold was `0.5` for all models.

## Metrics

The CI table reports:

- AUC
- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1
- TP / TN / FP / FN

The confusion matrix values are the original full-test image-level counts and
match the main final comparison table.

## ResNet50 AUC Note

The ResNet50 AUC point estimate is reported as `0.7016`, matching the original
final result table. Its 95% CI bounds come from the patient-level bootstrap
prediction file generated from the saved final ResNet50 model.

The regenerated ResNet50 prediction file produced the same thresholded
confusion matrix as the original final table:

```text
TP/TN/FP/FN = 89/92/51/58
```

The regenerated probability-based AUC differed only at the fourth decimal place
because of TensorFlow inference/runtime floating-point differences, so the
published point estimate was kept consistent with the original final table.

## Interpretation

These intervals quantify uncertainty in the final test-set performance while
accounting for within-patient correlation. They should be read together with
the original model-selection summaries and final resource comparison tables.
