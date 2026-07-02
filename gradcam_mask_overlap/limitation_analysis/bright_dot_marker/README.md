# Bright-Dot / Marker Limitation Analysis

This folder contains image-level cases where `bright_dot_or_marker != none`.

## Files

- `bright_dot_cases_confusion_wide.csv`: one row per marker case, with four-model prediction groups and top-20% metrics.
- `bright_dot_cases_by_model_long.csv`: one row per image-model pair.
- `bright_dot_confusion_summary_by_model.csv`: TP/FN/FP/TN counts by model and marker location.
- `bright_dot_confusion_counts_simple.csv`: simple TP/FN/FP/TN counts by model for all marker cases.
- `bright_dot_image_figure_index.csv`: one row per marker case with the rendered figure path.
- `bright_dot_attention_review.csv`: manual review for whether each model attends to the bright dot / marker.
- `bright_dot_attention_summary_by_model.csv`: model-level marker attention rates and confusion counts.
- `bright_dot_attention_by_confusion.csv`: marker attention rates stratified by TP/FN/FP/TN.
- `bright_dot_marker_classification_sensitivity_by_model.csv`: all-image, marker-negative, and marker-positive classification summaries by model.
- `bright_dot_marker_classification_performance_by_model.csv`: AUC, accuracy, sensitivity/recall, specificity, precision, and F1 by model and marker subset.
- `bright_dot_marker_classification_performance_delta_by_model.csv`: classification performance change after excluding marker-positive images.
- `bright_dot_marker_overlap_sensitivity_by_model.csv`: all-image, marker-negative, and marker-positive overlap metric summaries by model.
- `bright_dot_marker_exclusion_delta_by_model.csv`: change after excluding marker-positive images.
- `bright_dot_marker_sensitivity_summary.md`: concise sensitivity-analysis report.
- `figures_top20/`: one four-model top-20% QC panel per marker case.

Figure titles include the marker location and four-model confusion groups:

```text
marker=<location> | R=<ResNet50> V=<ViT_B16> C=<ConvNeXt_Base> S=<Swin_Base> | <image_id>
```

## Attention Review Coding

Manual review columns in `bright_dot_attention_review.csv`:

```text
ResNet50_attends_marker
ViT_B16_attends_marker
ConvNeXt_Base_attends_marker
Swin_Base_attends_marker
```

Coding:

```text
1 = the model's top-20% saliency visibly attends to the bright dot / marker
0 = the model's top-20% saliency does not visibly attend to the bright dot / marker
```

The optional note columns were removed because they were not used in this review pass.

## Attention Summary

| Model | N | Attends marker | Attention rate | TP | FN | FP | TN | Benign FP attends marker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | 25 | 12 | 48.0% | 16 | 3 | 3 | 3 | 3/3 |
| `ViT_B16` | 25 | 21 | 84.0% | 16 | 3 | 2 | 4 | 2/2 |
| `ConvNeXt_Base` | 25 | 23 | 92.0% | 17 | 2 | 5 | 1 | 5/5 |
| `Swin_Base` | 25 | 13 | 52.0% | 17 | 2 | 2 | 4 | 2/2 |

By confusion group:

```text
ResNet50: TP 5/16, FN 2/3, FP 3/3, TN 2/3
ViT_B16: TP 13/16, FN 3/3, FP 2/2, TN 3/4
ConvNeXt_Base: TP 15/17, FN 2/2, FP 5/5, TN 1/1
Swin_Base: TP 7/17, FN 1/2, FP 2/2, TN 3/4
```

## Marker Counts

- `inside_mask`: 9
- `on_border`: 2
- `outside_mask`: 13
- `uncertain`: 1

## Overall Confusion Counts

| Model | N | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet50` | 25 | 16 | 3 | 3 | 3 |
| `vit_b16` | 25 | 16 | 3 | 2 | 4 |
| `convnext_base` | 25 | 17 | 2 | 5 | 1 |
| `swin_base` | 25 | 17 | 2 | 2 | 4 |

## Interpretation Note

These cases are collected for limitation and sensitivity review. The marker location does not by itself prove model shortcut learning; it flags images where a small bright dot or marker-like artifact could complicate visual interpretation of saliency/overlap.

## Marker-Exclusion Sensitivity Analysis

Scope:

```text
Current version uses all 290 test images because usable_for_overlap is still
pending. This analysis should be rerun after the final mask-quality usable set
is defined.
```

Classification summary:

| Model | All-image benign FP rate | Marker-negative benign FP rate | Marker-positive benign FP rate | Specificity change after excluding marker |
| --- | ---: | ---: | ---: | ---: |
| `ResNet50` | 0.356643 | 0.350365 | 0.500000 | +0.006278 |
| `ViT_B16` | 0.125874 | 0.116788 | 0.333333 | +0.009086 |
| `ConvNeXt_Base` | 0.244755 | 0.218978 | 0.833333 | +0.025777 |
| `Swin_Base` | 0.083916 | 0.072993 | 0.333333 | +0.010923 |

Classification performance after excluding marker-positive images:

| Model | AUC | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | 0.680657 | 0.611321 | 0.570312 | 0.649635 | 0.603306 | 0.586345 |
| `ViT_B16` | 0.812044 | 0.773585 | 0.656250 | 0.883212 | 0.840000 | 0.736842 |
| `ConvNeXt_Base` | 0.782904 | 0.728302 | 0.671875 | 0.781022 | 0.741379 | 0.704918 |
| `Swin_Base` | 0.768761 | 0.754717 | 0.570312 | 0.927007 | 0.879518 | 0.691943 |

Classification performance delta after excluding marker-positive images:

```text
Delta = marker_negative - all_images
```

| Model | AUC delta | Accuracy delta | Sensitivity delta | Specificity delta | Precision delta | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.020784 | -0.012817 | -0.035130 | +0.006278 | -0.032408 | -0.033864 |
| `ViT_B16` | -0.006947 | -0.002277 | -0.024022 | +0.009086 | -0.007458 | -0.017875 |
| `ConvNeXt_Base` | -0.008067 | +0.000716 | -0.028805 | +0.025777 | -0.004998 | -0.017889 |
| `Swin_Base` | -0.012505 | -0.007352 | -0.041933 | +0.010923 | -0.002835 | -0.030949 |

Overlap sensitivity summary:

```text
Excluding marker-positive images changed mean top-20 IoU/Dice only slightly in
the all-image analysis. The larger signal is behavioral/classification-related:
marker-positive benign cases have elevated FP rates, especially for
ConvNeXt_Base.
```

Interpretation:

```text
The bright-dot / marker artifact appears to be more important as a potential
shortcut or classification-sensitivity factor than as a driver of global
overlap-metric averages. ConvNeXt_Base remains the strongest concern because it
combines high marker attention with the highest benign marker-positive FP rate.
```

## Current Conclusion and Remaining Work

Current conclusion:

```text
The bright-dot / marker analysis is complete for the all-image test set. The
main finding is not that marker exclusion globally improves model performance.
Instead, marker-positive cases are label-imbalanced toward malignant images, so
marker-positive AUC/recall/F1 can appear higher. The more meaningful artifact
signal is that benign marker-positive cases have elevated false-positive rates,
especially for ConvNeXt_Base.
```

What this supports:

```text
1. Bright-dot / marker artifacts should be described as a limitation and sanity
   check, not as definitive causal proof of shortcut learning.
2. ConvNeXt_Base is the strongest shortcut/artifact-sensitivity concern because
   it combines frequent marker attention with the highest benign marker-positive
   FP rate.
3. ViT_B16 frequently attends to markers, but the behavioral FP impact is less
   extreme than ConvNeXt_Base.
4. ResNet50 and Swin_Base show weaker overall marker attention, although their
   marker-positive FP cases still require caution.
5. Global overlap metrics change only slightly after marker-positive images are
   excluded, so the marker issue is primarily a classification/artifact
   sensitivity issue rather than a major driver of average overlap metrics.
```

Evidence chain for this interpretation:

```text
1. Average top-20 overlap metrics changed only slightly after marker-positive
   cases were excluded.

   IoU_top20 delta:
     ResNet50 = -0.002897
     ViT_B16 = -0.001460
     ConvNeXt_Base = +0.002877
     Swin_Base = -0.005598

   Dice_top20 delta:
     ResNet50 = -0.004294
     ViT_B16 = -0.002440
     ConvNeXt_Base = +0.003312
     Swin_Base = -0.008163

   These small absolute changes do not support the claim that bright-dot /
   marker artifacts materially shift average overlap performance.

2. In contrast, benign false-positive behavior differed more clearly between
   marker-negative and marker-positive cases.

   Benign FP rate, marker-negative vs marker-positive:
     ResNet50 = 0.350365 vs 0.500000
     ViT_B16 = 0.116788 vs 0.333333
     ConvNeXt_Base = 0.218978 vs 0.833333
     Swin_Base = 0.072993 vs 0.333333

   This is strongest for ConvNeXt_Base, where benign marker-positive FP rate is
   5/6 = 83.3%.

3. Manual saliency review showed frequent model attention to the marker,
   especially for ConvNeXt_Base and ViT_B16.

   Marker-attention rate:
     ResNet50 = 12/25 = 48.0%
     ViT_B16 = 21/25 = 84.0%
     ConvNeXt_Base = 23/25 = 92.0%
     Swin_Base = 13/25 = 52.0%

   Among marker-positive benign FP cases, all available FP cases attended to
   the marker:
     ResNet50 = 3/3
     ViT_B16 = 2/2
     ConvNeXt_Base = 5/5
     Swin_Base = 2/2
```

Recommended wording:

```text
Bright-dot artifacts did not materially change average top-20 overlap metrics
in the all-image sensitivity analysis, but marker-positive benign cases showed
elevated false-positive rates, especially for ConvNeXt_Base. Together with
manual saliency review showing frequent marker attention, this suggests
artifact sensitivity rather than a major shift in global overlap performance.
```

Remaining work:

```text
No additional manual bright-dot labeling is needed for the current dataset.

After usable_for_overlap is finalized for the mask-quality track, rerun
analyze_bright_dot_marker_sensitivity.py on the final mask-usable subset. This
will provide the publication-ready marker-exclusion sensitivity result for the
main analysis population.
```
