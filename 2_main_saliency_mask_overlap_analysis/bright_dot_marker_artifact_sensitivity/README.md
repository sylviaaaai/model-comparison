# Bright-Dot / Marker Artifact Sensitivity

## Purpose

This folder evaluates whether bright-dot or marker-like artifacts near lesion masks may affect saliency interpretation, classification behavior, or the main saliency-mask overlap conclusion.

The motivation comes from manual image review: some final test images contain small bright dots or marker-like objects near the annotated ROI. These may be clinical markers, acquisition artifacts, or other bright structures. Because saliency maps can visually attend to these regions, this analysis treats marker-positive images as a limitation/sensitivity track.

This analysis should not be interpreted as causal proof of shortcut learning. It asks whether marker-positive cases show warning signs that could complicate interpretation.

## Marker Definitions

The image-level master table contains a `bright_dot_or_marker` field.

| Marker status | Meaning |
| --- | --- |
| Marker-negative | `bright_dot_or_marker` is empty or `none`. |
| Marker-positive | `bright_dot_or_marker` is non-empty and not `none`. |

Marker status is separate from mask usability:

- `bright_dot_or_marker` is not used to decide `usable_for_overlap`.
- Marker-positive images can still be included in `main_yes` or `sensitivity_yes_cautious` if their lesion masks are usable.
- Marker-positive images are analyzed separately as an artifact-sensitivity and limitation check.

There are 25 marker-positive images in the full final test set. Within the overlap scopes used here:

| Scope | Marker-positive images |
| --- | ---: |
| `main_yes` | 18 |
| `sensitivity_yes_cautious` | 22 |

## Analysis Questions

This folder addresses three related questions:

1. **Does model saliency visibly attend to the marker?**
   - The analysis uses rendered top-20% saliency QC panels for marker-positive cases.
   - Manual review codes whether each model's top-20% saliency visibly attends to the marker.
   - Code `1` means attends marker; code `0` means does not attend marker.

2. **Are marker-positive images associated with different classification behavior?**
   - Classification summaries compare marker-positive, marker-negative, and all-image subsets.
   - The main focus is benign false-positive behavior, specificity, and related classification metrics.

3. **Does excluding marker-positive images change the overlap conclusions?**
   - Overlap metrics are recomputed after excluding marker-positive images.
   - The exclusion delta checks whether marker-positive cases materially drive the average overlap metrics or model ordering.

## Analysis Scopes

Two overlap-analysis scopes are summarized:

| Scope used in output file names | Meaning | Role |
| --- | --- | --- |
| `main_yes` | `usable_for_overlap == yes` | Strict primary overlap set. |
| `yes_cautious` | `usable_for_overlap in {yes, cautious}` | Broader mask-QC sensitivity set. |

Within each scope, rows are also split into:

| Subset | Meaning |
| --- | --- |
| `all_images` | All images in that scope. |
| `marker_negative` | Images without a marker flag. |
| `marker_positive` | Images with a marker flag. |
| `marker_positive_benign` | Marker-positive benign images. |
| `marker_positive_malignant` | Marker-positive malignant images. |

## Results

| File | Meaning |
| --- | --- |
| `results/00_bright_dot_marker_key_summary.csv` | Compact key summary combining marker attention, marker-positive classification behavior, and marker-exclusion overlap/classification deltas. Best first table to read. |
| `results/01_marker_attention_summary_by_model.csv` | Manual marker-attention review summarized by model, label, and prediction type. |
| `results/02_marker_attention_by_confusion.csv` | Marker-attention review summarized by model-specific TP/FN/FP/TN group. |
| `results/03_main_yes_classification_sensitivity.csv` | Classification behavior in the strict `main_yes` set by marker subset. |
| `results/04_yes_cautious_classification_sensitivity.csv` | Classification behavior in the broader `yes+cautious` set by marker subset. |
| `results/05_main_yes_marker_exclusion_delta.csv` | Main-set classification and overlap deltas after excluding marker-positive images. |
| `results/06_yes_cautious_marker_exclusion_delta.csv` | Yes+cautious classification and overlap deltas after excluding marker-positive images. |
| `results/07_main_yes_overlap_sensitivity.csv` | Main-set overlap metrics by marker subset. |
| `results/08_yes_cautious_overlap_sensitivity.csv` | Yes+cautious overlap metrics by marker subset. |

Recommended reading order:

1. Start with `results/00_bright_dot_marker_key_summary.csv` for the compact result.
2. Read `results/01_marker_attention_summary_by_model.csv` and `results/02_marker_attention_by_confusion.csv` to understand manual marker-attention patterns.
3. Read `results/03_main_yes_classification_sensitivity.csv` and `results/04_yes_cautious_classification_sensitivity.csv` to compare classification behavior by marker subset.
4. Read `results/05_main_yes_marker_exclusion_delta.csv` and `results/06_yes_cautious_marker_exclusion_delta.csv` to see whether excluding marker-positive images changes classification or overlap metrics.
5. Use `results/07_main_yes_overlap_sensitivity.csv` and `results/08_yes_cautious_overlap_sensitivity.csv` for the full overlap metric summaries by marker subset.

## How To Read The Key Fields

### Marker Attention

| Field | Meaning |
| --- | --- |
| `n_marker_images` | Number of marker-positive images reviewed for that model. |
| `attends_marker` | Number of marker-positive images where top-20% saliency visibly attends to the marker. |
| `attention_rate_percent` | `attends_marker / n_marker_images * 100`. |
| `benign_FP_attends_marker` | Marker-positive benign false-positive cases where saliency attends to the marker. |
| `benign_FP_attention_rate_percent` | Marker-attention rate among marker-positive benign false positives. |

### Classification Sensitivity

| Field | Meaning |
| --- | --- |
| `benign_fp_rate` | False-positive rate among benign images: `FP / (FP + TN)`. |
| `malignant_fn_rate` | False-negative rate among malignant images: `FN / (TP + FN)`. |
| `specificity` | True-negative rate: `TN / (TN + FP)`. |
| `sensitivity` | True-positive rate: `TP / (TP + FN)`. |

### Marker-Exclusion Delta

In the exclusion-delta tables, the main delta is:

```text
delta_after_excluding_marker = marker_negative - all_images
```

Interpretation examples:

- A negative `benign_fp_rate` delta means the benign FP rate decreases after marker-positive images are excluded.
- A positive `specificity` delta means specificity increases after marker-positive images are excluded.
- A near-zero `dice_top20` or `iou_top20` delta means marker-positive images do not materially shift the average overlap metric.

The `marker_positive_minus_marker_negative` column compares marker-positive images directly with marker-negative images.

## Result Snapshot

| Model | Marker-positive test images | Marker attention rate | Main marker-positive benign FP rate | Main FP-rate delta after marker exclusion | Main Dice top-20 delta after marker exclusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 25 | 48.0% | 0.50 | -0.004 | -0.002 |
| ViT_B16 | 25 | 84.0% | 0.50 | -0.020 | -0.002 |
| ConvNeXt_Base | 25 | 92.0% | 0.75 | -0.029 | 0.003 |
| Swin_Base | 25 | 52.0% | 0.50 | -0.023 | -0.009 |

Key reading:

- ConvNeXt-Base and ViT_B16 show the highest manual marker-attention rates.
- In marker-positive benign cases, ConvNeXt-Base has the highest false-positive rate in the strict `main_yes` set.
- Excluding marker-positive images slightly lowers benign FP rate and slightly increases specificity, especially for ConvNeXt-Base and Swin-Base.
- The overlap deltas after marker exclusion are small. Marker-positive images do not materially drive the global saliency-mask overlap ranking.

## Caveats

- The marker-positive subset is small: 25 images in the full final test set, 18 in `main_yes`, and 22 in `yes+cautious`.
- Marker attention was manually coded from rendered top-20% saliency panels, so it is a visual-review variable, not an automatically measured geometric overlap with marker coordinates.
- Marker status is not a controlled intervention. This analysis cannot prove the model used a marker artifact causally.
- Marker-positive cases may differ from marker-negative cases in label mix, mask type, lesion appearance, or acquisition context.
- The strongest evidence here is a limitation/sanity-check pattern: frequent marker attention and elevated benign false-positive rates in marker-positive subsets, not a definitive shortcut-learning claim.

## Conclusion

Marker artifacts do not materially shift average overlap metrics or drive the global overlap ranking. ConvNeXt-Base remains the top lesion-overlap model after marker-positive exclusion.

The stronger signal is classification/artifact sensitivity: marker-positive benign cases show elevated false-positive behavior, especially for ConvNeXt-Base, and manual saliency review shows frequent marker attention for ConvNeXt-Base and ViT_B16. This should be framed as a limitation and interpretability caution rather than causal proof of shortcut learning.

