# Bright-Dot / Marker Sensitivity Summary

This analysis compares marker-positive and marker-negative images using the current image-level master table.

Important scope note:

```text
This is the bright-dot / marker artifact track. It does not use
bright_dot_or_marker to decide mask usability.
usable_scope = yes_cautious
source_rows = 290
analyzed_rows = 279
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
| `ResNet50` | `all_images` | 279 | 83 | 58 | 50 | 88 | 0.362319 | 0.637681 |
| `ResNet50` | `marker_negative` | 257 | 70 | 55 | 47 | 85 | 0.356061 | 0.643939 |
| `ResNet50` | `marker_positive` | 22 | 13 | 3 | 3 | 3 | 0.5 | 0.5 |
| `ViT_B16` | `all_images` | 279 | 95 | 46 | 18 | 120 | 0.130435 | 0.869565 |
| `ViT_B16` | `marker_negative` | 257 | 82 | 43 | 16 | 116 | 0.121212 | 0.878788 |
| `ViT_B16` | `marker_positive` | 22 | 13 | 3 | 2 | 4 | 0.333333 | 0.666667 |
| `ConvNeXt_Base` | `all_images` | 279 | 98 | 43 | 35 | 103 | 0.253623 | 0.746377 |
| `ConvNeXt_Base` | `marker_negative` | 257 | 84 | 41 | 30 | 102 | 0.227273 | 0.772727 |
| `ConvNeXt_Base` | `marker_positive` | 22 | 14 | 2 | 5 | 1 | 0.833333 | 0.166667 |
| `Swin_Base` | `all_images` | 279 | 85 | 56 | 10 | 128 | 0.0724638 | 0.927536 |
| `Swin_Base` | `marker_negative` | 257 | 71 | 54 | 8 | 124 | 0.0606061 | 0.939394 |
| `Swin_Base` | `marker_positive` | 22 | 14 | 2 | 2 | 4 | 0.333333 | 0.666667 |

## Classification Performance

| Model | Scope | AUC | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | `all_images` | 0.685322 | 0.612903 | 0.588652 | 0.637681 | 0.62406 | 0.605839 |
| `ResNet50` | `marker_negative` | 0.668667 | 0.603113 | 0.56 | 0.643939 | 0.598291 | 0.578512 |
| `ResNet50` | `marker_positive` | 0.791667 | 0.727273 | 0.8125 | 0.5 | 0.8125 | 0.8125 |
| `ViT_B16` | `all_images` | 0.815089 | 0.770609 | 0.673759 | 0.869565 | 0.840708 | 0.748031 |
| `ViT_B16` | `marker_negative` | 0.811939 | 0.770428 | 0.656 | 0.878788 | 0.836735 | 0.735426 |
| `ViT_B16` | `marker_positive` | 0.822917 | 0.772727 | 0.8125 | 0.666667 | 0.866667 | 0.83871 |
| `ConvNeXt_Base` | `all_images` | 0.783637 | 0.72043 | 0.695035 | 0.746377 | 0.736842 | 0.715328 |
| `ConvNeXt_Base` | `marker_negative` | 0.779576 | 0.723735 | 0.672 | 0.772727 | 0.736842 | 0.702929 |
| `ConvNeXt_Base` | `marker_positive` | 0.760417 | 0.681818 | 0.875 | 0.166667 | 0.736842 | 0.8 |
| `Swin_Base` | `all_images` | 0.779217 | 0.763441 | 0.602837 | 0.927536 | 0.894737 | 0.720339 |
| `Swin_Base` | `marker_negative` | 0.771636 | 0.758755 | 0.568 | 0.939394 | 0.898734 | 0.696078 |
| `Swin_Base` | `marker_positive` | 0.84375 | 0.818182 | 0.875 | 0.666667 | 0.875 | 0.875 |

## Classification Performance Delta

Delta is computed as `marker_negative - all_images`.

| Model | AUC delta | Accuracy delta | Sensitivity delta | Specificity delta | Precision delta | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.016655 | -0.00979 | -0.028652 | 0.006258 | -0.025769 | -0.027327 |
| `ViT_B16` | -0.00315 | -0.000181 | -0.017759 | 0.009223 | -0.003973 | -0.012605 |
| `ConvNeXt_Base` | -0.004061 | 0.003305 | -0.023035 | 0.02635 | 0 | -0.012399 |
| `Swin_Base` | -0.007581 | -0.004686 | -0.034837 | 0.011858 | 0.003997 | -0.024261 |

## Marker Exclusion Delta

Delta is computed as `marker_negative - all_images`, so a negative FP-rate delta means FP rate decreases after marker-positive images are excluded.

| Model | FP-rate delta | Specificity delta | Mean iou_top20 delta | Mean dice_top20 delta | Mean normalized_energy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.006258 | 0.006258 | -0.00208 | -0.003196 | -0.00768 |
| `ViT_B16` | -0.009223 | 0.009223 | -0.0006307 | -0.001058 | 0.000977 |
| `ConvNeXt_Base` | -0.02635 | 0.02635 | 0.002233 | 0.00249 | 0.01224 |
| `Swin_Base` | -0.0118577 | 0.011858 | -0.004868 | -0.007059 | -0.0122 |

## Interpretation

The marker-positive subset is small and imbalanced, so the analysis should be treated as a sensitivity / limitation check rather than a definitive causal test.

ConvNeXt_Base remains the strongest artifact-sensitivity concern because it combines frequent marker attention with a high benign marker false-positive rate.

ViT_B16 also frequently attends to markers, but the marker-positive benign false-positive count is lower than ConvNeXt_Base.

ResNet50 and Swin_Base show lower overall marker-attention rates. However, marker attention is still present in their marker-positive false-positive cases, so the issue is relevant across architectures.
