# Subgroup Analysis

## Purpose

This exploratory analysis asks where the main saliency-mask overlap signal comes from after stratifying the test images by label and prediction outcome.

It does not replace the primary `main_yes` and `sensitivity_yes_cautious` model comparisons. Instead, it helps interpret whether the overlap pattern is similar in malignant vs benign images, and whether overlap differs across each model's true-positive, false-negative, false-positive, and true-negative cases.

## Analysis Design

The analysis uses the overlap metrics from the main usable-mask scopes:

| Scope | Rule | Role |
| --- | --- | --- |
| `main_yes` | `usable_for_overlap == yes` | Primary strict usable-mask subgroup analysis. |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | Sensitivity subgroup analysis including cautious masks. |

Two subgrouping strategies are used:

| Stratifier | Meaning | Important note |
| --- | --- | --- |
| `true_label` | Groups images into malignant and benign cases. | The subgroup is the same for every model. |
| `confusion_group` | Groups images by each model's prediction outcome: TP, FN, FP, TN. | This subgroup is model-specific, so the same image can be TP for one model and FN/FP/TN for another. |

The confusion groups mean:

| Group | True label | Model prediction | Interpretation |
| --- | --- | --- | --- |
| `TP` | Malignant | Malignant | Correct malignant prediction. |
| `FN` | Malignant | Benign | Missed malignant case. |
| `FP` | Benign | Malignant | False-positive malignant prediction. |
| `TN` | Benign | Benign | Correct benign prediction. |

Because FP and FN groups can be small, especially for some models, these subgroup results should be read descriptively rather than as definitive statistical comparisons.

## Metrics

The full subgroup analysis computes the same seven compact overlap metrics used elsewhere:

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `energy_inside_mask` | Fraction of saliency energy inside the lesion mask. | Higher is better. |
| `normalized_energy` | Mask energy normalized against expected/background energy. | Higher is better. |
| `peak_in_mask` | Whether the saliency peak falls inside the lesion mask. | Higher is better. |
| `peak_distance_px` | Pixel distance from saliency peak to lesion mask/target. | Lower is better. |
| `mask_coverage_top20` | Fraction of lesion mask covered by the top 20% saliency region. | Higher is better. |
| `iou_top20` | IoU between lesion mask and top 20% saliency region. | Higher is better. |
| `dice_top20` | Dice score between lesion mask and top 20% saliency region. | Higher is better. |

The paper-facing readable tables focus on a smaller set:

- `normalized_energy`
- `peak_in_mask`
- `iou_top20`
- `dice_top20`

For confusion groups, the readable output focuses on `dice_top20`.

## Results

| File | Meaning |
| --- | --- |
| `results/00_true_label_key_metrics_readable.csv` | Main readable table for malignant vs benign subgroups. Includes key metrics and mean/median values for all four models. |
| `results/01_confusion_group_dice_readable.csv` | Main readable table for model-specific TP/FN/FP/TN groups using top-20% Dice. |
| `results/02_subgroup_counts.csv` | Denominator and composition table for subgroup interpretation. Includes counts by scope, model, true label, confusion group, marker status, and zero-saliency flag. |

Recommended reading order:

1. Start with `results/02_subgroup_counts.csv` to check subgroup denominators.
2. Read `results/00_true_label_key_metrics_readable.csv` to compare malignant vs benign overlap patterns.
3. Read `results/01_confusion_group_dice_readable.csv` to inspect TP/FN/FP/TN Dice patterns.

## Column Guide

### `00_true_label_key_metrics_readable.csv`

| Column | Meaning |
| --- | --- |
| `scope` | Analysis scope: `main_yes` or `sensitivity_yes_cautious`. |
| `true_label` | Malignant or benign subgroup. |
| `n_images` | Number of images in that scope and true-label subgroup. |
| `metric` | Reported overlap metric. |
| `<model>_mean` | Mean metric value for that model in the subgroup. |
| `<model>_median` | Median metric value for that model in the subgroup. |

### `01_confusion_group_dice_readable.csv`

| Column | Meaning |
| --- | --- |
| `scope` | Analysis scope. |
| `model` | Model whose prediction outcome defines TP/FN/FP/TN. |
| `metric` | Always `dice_top20` in this readable table. |
| `TP_n`, `FN_n`, `FP_n`, `TN_n` | Number of images in each model-specific confusion group. |
| `TP_mean`, `FN_mean`, `FP_mean`, `TN_mean` | Mean top-20% Dice in each confusion group. |
| `TP_median`, `FN_median`, `FP_median`, `TN_median` | Median top-20% Dice in each confusion group. |

### `02_subgroup_counts.csv`

| Column | Meaning |
| --- | --- |
| `stratifier` | Which subgrouping was used: `true_label`, `confusion_group`, or `true_label_and_confusion_group`. |
| `scope` | Analysis scope. |
| `model` | Model name. |
| `true_label` | Malignant/benign label when relevant. |
| `confusion_group` | TP/FN/FP/TN group when relevant. |
| `n_images` | Number of images in the subgroup. |
| `n_malignant`, `n_benign` | Label composition of the subgroup. |
| `TP`, `FN`, `FP`, `TN` | Confusion-group counts for that model. |
| `marker_positive` | Number of images with marker/artifact annotations. |
| `zero_saliency_any_model` | Number of images where at least one model has zero saliency. |

## Main Result Snapshot

### True-Label Subgroups

Mean top-20% Dice by true label:

| Scope | True label | ResNet50 | ViT_B16 | ConvNeXt_Base | Swin_Base |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `malignant` | 0.427 | 0.123 | 0.444 | 0.359 |
| `main_yes` | `benign` | 0.366 | 0.106 | 0.389 | 0.193 |
| `sensitivity_yes_cautious` | `malignant` | 0.410 | 0.114 | 0.438 | 0.356 |
| `sensitivity_yes_cautious` | `benign` | 0.343 | 0.100 | 0.399 | 0.198 |

Key reading:

- ConvNeXt-Base remains high in both malignant and benign strata.
- ResNet50 is also relatively high, especially in malignant images.
- ViT_B16 remains low in both malignant and benign strata.
- Swin_Base is stronger in malignant images than benign images.

### Model-Specific Confusion Groups

Mean top-20% Dice by model-specific confusion group:

| Scope | Model | TP | FN | FP | TN |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | ResNet50 | 0.498 | 0.294 | 0.475 | 0.287 |
| `main_yes` | ViT_B16 | 0.138 | 0.086 | 0.126 | 0.103 |
| `main_yes` | ConvNeXt_Base | 0.484 | 0.354 | 0.456 | 0.368 |
| `main_yes` | Swin_Base | 0.464 | 0.196 | 0.422 | 0.173 |
| `sensitivity_yes_cautious` | ResNet50 | 0.486 | 0.302 | 0.483 | 0.263 |
| `sensitivity_yes_cautious` | ViT_B16 | 0.129 | 0.083 | 0.129 | 0.096 |
| `sensitivity_yes_cautious` | ConvNeXt_Base | 0.472 | 0.360 | 0.458 | 0.379 |
| `sensitivity_yes_cautious` | Swin_Base | 0.455 | 0.206 | 0.436 | 0.179 |

Key reading:

- TP groups generally show higher Dice than FN groups, consistent with stronger lesion alignment when the malignant prediction is correct.
- FP groups can still have measurable overlap with annotated benign findings. This suggests some false-positive malignant predictions are spatially aligned with annotated benign abnormalities rather than being completely off-mask.
- TN groups generally have lower overlap than FP groups for ResNet50, ConvNeXt-Base, and Swin-Base.
- ViT_B16 stays low across all confusion groups.

## Caveats

- This analysis is exploratory and descriptive.
- The confusion group is model-specific, so TP/FN/FP/TN rows are not the same image sets across models.
- Some FP/FN denominators are small. For example, in `main_yes`, Swin-Base has only 6 FP cases and ViT_B16 has 11 FP cases.
- ResNet50 zero-saliency rows can reduce valid `peak_distance_px` counts, especially in subgroups with more zero-saliency cases.
- Subgroup differences should not be interpreted as causal evidence about model decision-making.

## Conclusion

The subgroup analysis supports the main overlap conclusion while adding context. ConvNeXt-Base maintains relatively strong lesion-mask overlap in both malignant and benign images, ResNet50 is the closest comparator, Swin-Base is intermediate, and ViT_B16 remains low. TP cases tend to have stronger overlap than FN cases, while FP cases often still show nontrivial overlap with annotated benign findings. Because subgroup denominators can be small, these findings should be framed as descriptive interpretation rather than confirmatory statistical evidence.

