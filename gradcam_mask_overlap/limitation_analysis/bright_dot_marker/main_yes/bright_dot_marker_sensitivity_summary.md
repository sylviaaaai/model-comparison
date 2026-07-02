# Bright-Dot / Marker Sensitivity Summary

This analysis compares marker-positive and marker-negative images using the current image-level master table.

Important scope note:

```text
This is the bright-dot / marker artifact track. It does not use
bright_dot_or_marker to decide mask usability.
usable_scope = main_yes
source_rows = 290
analyzed_rows = 183
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
| `ResNet50` | `all_images` | 183 | 70 | 37 | 32 | 44 | 0.421053 | 0.578947 |
| `ResNet50` | `marker_negative` | 165 | 59 | 34 | 30 | 42 | 0.416667 | 0.583333 |
| `ResNet50` | `marker_positive` | 18 | 11 | 3 | 2 | 2 | 0.5 | 0.5 |
| `ViT_B16` | `all_images` | 183 | 76 | 31 | 11 | 65 | 0.144737 | 0.855263 |
| `ViT_B16` | `marker_negative` | 165 | 65 | 28 | 9 | 63 | 0.125 | 0.875 |
| `ViT_B16` | `marker_positive` | 18 | 11 | 3 | 2 | 2 | 0.5 | 0.5 |
| `ConvNeXt_Base` | `all_images` | 183 | 74 | 33 | 18 | 58 | 0.236842 | 0.763158 |
| `ConvNeXt_Base` | `marker_negative` | 165 | 62 | 31 | 15 | 57 | 0.208333 | 0.791667 |
| `ConvNeXt_Base` | `marker_positive` | 18 | 12 | 2 | 3 | 1 | 0.75 | 0.25 |
| `Swin_Base` | `all_images` | 183 | 65 | 42 | 6 | 70 | 0.0789474 | 0.921053 |
| `Swin_Base` | `marker_negative` | 165 | 53 | 40 | 4 | 68 | 0.0555556 | 0.944444 |
| `Swin_Base` | `marker_positive` | 18 | 12 | 2 | 2 | 2 | 0.5 | 0.5 |

## Classification Performance

| Model | Scope | AUC | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | `all_images` | 0.704378 | 0.622951 | 0.654206 | 0.578947 | 0.686275 | 0.669856 |
| `ResNet50` | `marker_negative` | 0.68638 | 0.612121 | 0.634409 | 0.583333 | 0.662921 | 0.648352 |
| `ResNet50` | `marker_positive` | 0.839286 | 0.722222 | 0.785714 | 0.5 | 0.846154 | 0.814815 |
| `ViT_B16` | `all_images` | 0.831407 | 0.770492 | 0.71028 | 0.855263 | 0.873563 | 0.783505 |
| `ViT_B16` | `marker_negative` | 0.830944 | 0.775758 | 0.698925 | 0.875 | 0.878378 | 0.778443 |
| `ViT_B16` | `marker_positive` | 0.803571 | 0.722222 | 0.785714 | 0.5 | 0.846154 | 0.814815 |
| `ConvNeXt_Base` | `all_images` | 0.778775 | 0.721311 | 0.691589 | 0.763158 | 0.804348 | 0.743719 |
| `ConvNeXt_Base` | `marker_negative` | 0.774791 | 0.721212 | 0.666667 | 0.791667 | 0.805195 | 0.729412 |
| `ConvNeXt_Base` | `marker_positive` | 0.732143 | 0.722222 | 0.857143 | 0.25 | 0.8 | 0.827586 |
| `Swin_Base` | `all_images` | 0.777668 | 0.737705 | 0.607477 | 0.921053 | 0.915493 | 0.730337 |
| `Swin_Base` | `marker_negative` | 0.772401 | 0.733333 | 0.569892 | 0.944444 | 0.929825 | 0.706667 |
| `Swin_Base` | `marker_positive` | 0.803571 | 0.777778 | 0.857143 | 0.5 | 0.857143 | 0.857143 |

## Classification Performance Delta

Delta is computed as `marker_negative - all_images`.

| Model | AUC delta | Accuracy delta | Sensitivity delta | Specificity delta | Precision delta | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.017998 | -0.01083 | -0.019797 | 0.004386 | -0.023354 | -0.021504 |
| `ViT_B16` | -0.000463 | 0.005266 | -0.011355 | 0.019737 | 0.004815 | -0.005062 |
| `ConvNeXt_Base` | -0.003984 | -9.9e-05 | -0.024922 | 0.028509 | 0.000847 | -0.014307 |
| `Swin_Base` | -0.005267 | -0.004372 | -0.037585 | 0.023391 | 0.014332 | -0.02367 |

## Marker Exclusion Delta

Delta is computed as `marker_negative - all_images`, so a negative FP-rate delta means FP rate decreases after marker-positive images are excluded.

| Model | FP-rate delta | Specificity delta | Mean iou_top20 delta | Mean dice_top20 delta | Mean normalized_energy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.004386 | 0.004386 | -0.001586 | -0.002391 | -0.0093 |
| `ViT_B16` | -0.019737 | 0.019737 | -0.0009161 | -0.001536 | 0.001672 |
| `ConvNeXt_Base` | -0.028509 | 0.028509 | 0.002889 | 0.003267 | 0.01405 |
| `Swin_Base` | -0.0233918 | 0.023391 | -0.005945 | -0.008745 | -0.01571 |

## Interpretation

The marker-positive subset is small and imbalanced, so the analysis should be treated as a sensitivity / limitation check rather than a definitive causal test.

ConvNeXt_Base remains the strongest artifact-sensitivity concern because it combines frequent marker attention with a high benign marker false-positive rate.

ViT_B16 also frequently attends to markers, but the marker-positive benign false-positive count is lower than ConvNeXt_Base.

ResNet50 and Swin_Base show lower overall marker-attention rates. However, marker attention is still present in their marker-positive false-positive cases, so the issue is relevant across architectures.
