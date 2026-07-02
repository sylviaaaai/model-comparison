# Overlap Subgroup Analysis

This is an exploratory subgroup analysis. It explains where the main overlap signal is coming from; it does not replace the primary `main_yes` and `yes_cautious` model comparisons.

## Output Tables

- `overlap_by_true_label_summary.csv`: scope x true_label x model x metric.
- `overlap_by_confusion_group_summary.csv`: scope x model-specific TP/FN/FP/TN x metric.
- `overlap_by_true_label_and_confusion_group_summary.csv`: scope x true_label x model-specific confusion group x metric.
- `overlap_subgroup_counts.csv`: denominators for the subgroup tables.

## True-Label Subgroups

Mean top-20% Dice by true label:

| Scope | True label | ResNet50 | ViT_B16 | ConvNeXt_Base | Swin_Base |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `malignant` | 0.427275 | 0.122827 | 0.444076 | 0.358865 |
| `main_yes` | `benign` | 0.366122 | 0.106356 | 0.388726 | 0.192647 |
| `sensitivity_yes_cautious` | `malignant` | 0.410223 | 0.11365 | 0.437833 | 0.355718 |
| `sensitivity_yes_cautious` | `benign` | 0.342983 | 0.100072 | 0.398832 | 0.198072 |

## Model-Specific Confusion Groups

Mean top-20% Dice by model-specific confusion group:

| Scope | Model | TP | FN | FP | TN |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `ResNet50` | 0.497607 | 0.294213 | 0.475056 | 0.286897 |
| `main_yes` | `ViT_B16` | 0.137697 | 0.0863692 | 0.126475 | 0.102951 |
| `main_yes` | `ConvNeXt_Base` | 0.484293 | 0.353893 | 0.455702 | 0.36794 |
| `main_yes` | `Swin_Base` | 0.464191 | 0.195862 | 0.422254 | 0.172966 |
| `sensitivity_yes_cautious` | `ResNet50` | 0.485913 | 0.301908 | 0.483169 | 0.263332 |
| `sensitivity_yes_cautious` | `ViT_B16` | 0.128551 | 0.0828763 | 0.12931 | 0.0956866 |
| `sensitivity_yes_cautious` | `ConvNeXt_Base` | 0.472061 | 0.359822 | 0.457968 | 0.378737 |
| `sensitivity_yes_cautious` | `Swin_Base` | 0.454501 | 0.205778 | 0.436227 | 0.179466 |

## Interpretation Notes

- `true_label` subgrouping asks whether overlap behavior differs between malignant and benign images.
- `confusion_group` subgrouping is model-specific: each model has its own TP/FN/FP/TN labels.
- TP/FN are malignant-image groups; FP/TN are benign-image groups.
- Small FP/FN subgroup denominators should be interpreted cautiously.
- ResNet50 zero-saliency rows can reduce valid `peak_distance_px` counts in TN/FN-heavy subgroups.
