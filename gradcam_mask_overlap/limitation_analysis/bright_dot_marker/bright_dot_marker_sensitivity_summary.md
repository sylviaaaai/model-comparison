# Bright-Dot / Marker Sensitivity Summary

This analysis compares marker-positive and marker-negative images using the current image-level master table.

Important scope note:

```text
This is the bright-dot / marker artifact track. It does not use
bright_dot_or_marker to decide mask usability. Because usable_for_overlap
is still pending, this version summarizes all 290 test images. The same
analysis should be rerun on the final mask-usable set after
usable_for_overlap is completed.
```

## Files

- `bright_dot_marker_classification_sensitivity_by_model.csv`
- `bright_dot_marker_classification_performance_by_model.csv`
- `bright_dot_marker_classification_performance_delta_by_model.csv`
- `bright_dot_marker_overlap_sensitivity_by_model.csv`
- `bright_dot_marker_exclusion_delta_by_model.csv`

## Classification Summary

| Model | Scope | N | TP | FN | FP | TN | Benign FP rate | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | `all_images` | 290 | 89 | 58 | 51 | 92 | 0.356643 | 0.643357 |
| `ResNet50` | `marker_negative` | 265 | 73 | 55 | 48 | 89 | 0.350365 | 0.649635 |
| `ResNet50` | `marker_positive` | 25 | 16 | 3 | 3 | 3 | 0.5 | 0.5 |
| `ViT_B16` | `all_images` | 290 | 100 | 47 | 18 | 125 | 0.125874 | 0.874126 |
| `ViT_B16` | `marker_negative` | 265 | 84 | 44 | 16 | 121 | 0.116788 | 0.883212 |
| `ViT_B16` | `marker_positive` | 25 | 16 | 3 | 2 | 4 | 0.333333 | 0.666667 |
| `ConvNeXt_Base` | `all_images` | 290 | 103 | 44 | 35 | 108 | 0.244755 | 0.755245 |
| `ConvNeXt_Base` | `marker_negative` | 265 | 86 | 42 | 30 | 107 | 0.218978 | 0.781022 |
| `ConvNeXt_Base` | `marker_positive` | 25 | 17 | 2 | 5 | 1 | 0.833333 | 0.166667 |
| `Swin_Base` | `all_images` | 290 | 90 | 57 | 12 | 131 | 0.0839161 | 0.916084 |
| `Swin_Base` | `marker_negative` | 265 | 73 | 55 | 10 | 127 | 0.0729927 | 0.927007 |
| `Swin_Base` | `marker_positive` | 25 | 17 | 2 | 2 | 4 | 0.333333 | 0.666667 |

## Classification Performance

| Model | Scope | AUC | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | `all_images` | 0.701441 | 0.624138 | 0.605442 | 0.643357 | 0.635714 | 0.620209 |
| `ResNet50` | `marker_negative` | 0.680657 | 0.611321 | 0.570312 | 0.649635 | 0.603306 | 0.586345 |
| `ResNet50` | `marker_positive` | 0.824561 | 0.76 | 0.842105 | 0.5 | 0.842105 | 0.842105 |
| `ViT_B16` | `all_images` | 0.818991 | 0.775862 | 0.680272 | 0.874126 | 0.847458 | 0.754717 |
| `ViT_B16` | `marker_negative` | 0.812044 | 0.773585 | 0.65625 | 0.883212 | 0.84 | 0.736842 |
| `ViT_B16` | `marker_positive` | 0.850877 | 0.8 | 0.842105 | 0.666667 | 0.888889 | 0.864865 |
| `ConvNeXt_Base` | `all_images` | 0.790971 | 0.727586 | 0.70068 | 0.755245 | 0.746377 | 0.722807 |
| `ConvNeXt_Base` | `marker_negative` | 0.782904 | 0.728302 | 0.671875 | 0.781022 | 0.741379 | 0.704918 |
| `ConvNeXt_Base` | `marker_positive` | 0.798246 | 0.72 | 0.894737 | 0.166667 | 0.772727 | 0.829268 |
| `Swin_Base` | `all_images` | 0.781266 | 0.762069 | 0.612245 | 0.916084 | 0.882353 | 0.722892 |
| `Swin_Base` | `marker_negative` | 0.768761 | 0.754717 | 0.570312 | 0.927007 | 0.879518 | 0.691943 |
| `Swin_Base` | `marker_positive` | 0.868421 | 0.84 | 0.894737 | 0.666667 | 0.894737 | 0.894737 |

## Classification Performance Delta

Delta is computed as `marker_negative - all_images`.

| Model | AUC delta | Accuracy delta | Sensitivity delta | Specificity delta | Precision delta | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.020784 | -0.012817 | -0.03513 | 0.006278 | -0.032408 | -0.033864 |
| `ViT_B16` | -0.006947 | -0.002277 | -0.024022 | 0.009086 | -0.007458 | -0.017875 |
| `ConvNeXt_Base` | -0.008067 | 0.000716 | -0.028805 | 0.025777 | -0.004998 | -0.017889 |
| `Swin_Base` | -0.012505 | -0.007352 | -0.041933 | 0.010923 | -0.002835 | -0.030949 |

## Marker Exclusion Delta

Delta is computed as `marker_negative - all_images`, so a negative FP-rate delta means FP rate decreases after marker-positive images are excluded.

| Model | FP-rate delta | Specificity delta | Mean iou_top20 delta | Mean dice_top20 delta | Mean normalized_energy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.006278 | 0.006278 | -0.002897 | -0.004294 | -0.0097 |
| `ViT_B16` | -0.009086 | 0.009086 | -0.0014596 | -0.00244 | -0.000771 |
| `ConvNeXt_Base` | -0.025777 | 0.025777 | 0.002877 | 0.003312 | 0.01507 |
| `Swin_Base` | -0.0109234 | 0.010923 | -0.005598 | -0.008163 | -0.0139 |

## Interpretation

The marker-positive subset is small and imbalanced, so the analysis should be treated as a sensitivity / limitation check rather than a definitive causal test.

ConvNeXt_Base remains the strongest artifact-sensitivity concern because it combines frequent marker attention with a high benign marker false-positive rate.

ViT_B16 also frequently attends to markers, but the marker-positive benign false-positive count is lower than ConvNeXt_Base.

ResNet50 and Swin_Base show lower overall marker-attention rates. However, marker attention is still present in their marker-positive false-positive cases, so the issue is relevant across architectures.
