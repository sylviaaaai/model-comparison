# Subgroup Analysis

This exploratory analysis asks where the main overlap signal comes from by stratifying overlap metrics by true label and by each model's own confusion group.

## Subgroups

The analysis reports overlap by `true_label`, by model-specific `confusion_group` (TP, FN, FP, TN), and by related subgroup counts. The confusion group is model-specific, so a case can belong to different groups for different models.

## Results

| Table | Meaning |
| --- | --- |
| `results/00_true_label_key_metrics_readable.csv` | Main readable table for malignant vs benign subgroups. |
| `results/01_confusion_group_dice_readable.csv` | Main readable table for TP/FN/FP/TN top-20% Dice. |
| `results/02_subgroup_counts.csv` | Denominator table for subgroup interpretation. |

## Conclusion

ConvNeXt-Base maintains relatively high lesion-mask overlap in both malignant and benign images, while ViT-B/16 remains low in both strata. TP cases generally have higher Dice than FN cases. FP cases can still show measurable overlap with annotated benign findings, suggesting that some false-positive predictions are spatially aligned with annotated benign abnormalities. These subgroup results are descriptive because some FP/FN denominators are small.
