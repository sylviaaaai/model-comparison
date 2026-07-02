# Experiment Log: Master-Table Overlap and Marker Sensitivity

Date: 2026-07-01
Workspace: `/mnt/e/SW整合`

## Purpose

This log records the current image-level master-table overlap analysis after
manual mask usability coding and the 0019 correction.

The analysis keeps two tracks separate:

```text
1. Lesion-mask usability for overlap analysis
2. Bright-dot / marker artifact sensitivity
```

Important rule:

```text
bright_dot_or_marker is not used to decide usable_for_overlap.
Bright-dot / marker cases are analyzed as a separate artifact-sensitivity
limitation track.
```

## Input Table

Source:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Status at this run:

```text
rows = 290
one row = one test image

usable_for_overlap:
  yes = 183
  cautious = 96
  no = 11

included_in_main_overlap_analysis:
  True = 183
  False = 107

missing_reason:
  blank = 183
  cautious_sensitivity_only = 96
  not_usable_for_overlap = 11
```

Manual-QC rule summary:

```text
yes:
  manual_mask_type in {localized_mass_like, localized_malignant_annotation}
  AND mask_qc_type = tight_or_reasonable

cautious:
  manual_mask_type in {calcification_region, architectural_distortion_like,
  uncertain_annotation}
  OR mask_qc_type in {overinclusive_annotation, border_touching, diffuse_large}

no:
  mask_qc_type in {foreground_like_or_contour, cropped_or_truncated,
  bad_or_mismatched}
```

## Step 1: Main and Sensitivity Overlap Analysis

Script:

```text
gradcam_mask_overlap/tools/analyze_master_overlap_by_usable_set.py
```

Command:

```text
python3 gradcam_mask_overlap/tools/analyze_master_overlap_by_usable_set.py
```

Output folder:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/
```

Generated files:

```text
overlap_usable_scope_counts.csv
overlap_usable_scope_model_metric_summary.csv
overlap_usable_scope_model_metric_ranks.csv
overlap_peak_distance_validity_by_scope.csv
overlap_usable_scope_summary.json
overlap_usable_scope_summary.md
```

Analysis scopes:

```text
main_yes:
  usable_for_overlap == yes

sensitivity_yes_cautious:
  usable_for_overlap in {yes, cautious}

excluded_no:
  usable_for_overlap == no

all_images:
  all 290 images; context only
```

Metrics summarized:

```text
energy_inside_mask
normalized_energy
peak_in_mask
peak_distance_px
mask_coverage_top20
iou_top20
dice_top20
```

## How to Read the Overlap Tables

`overlap_usable_scope_counts.csv`:

```text
One row per analysis scope.
Use this first to confirm denominators.

Important columns:
  n_images
  n_malignant
  n_benign
  marker_positive
  marker_negative
```

`overlap_usable_scope_model_metric_summary.csv`:

```text
One row per scope x model x metric.
Use this for descriptive statistics and plotting.

Important columns:
  scope
  model
  metric
  n_valid
  mean
  median
  sd
  min
  max
```

`overlap_usable_scope_model_metric_ranks.csv`:

```text
One row per scope x metric x model rank.
Use this for model ordering.

direction = higher_is_better for all metrics except:
  peak_distance_px = lower_is_better
```

`overlap_peak_distance_validity_by_scope.csv`:

```text
One row per model x scope.
Use this table to report the denominator for peak_distance_px.

Important columns:
  n_images
  n_valid_peak_distance
  n_invalid
  reason
```

Note:

```text
For peak_distance_px, ResNet50 can have lower n_valid because zero-saliency
maps have no valid peak coordinate and therefore no peak-to-mask distance.
```

## Scope Counts

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_counts.csv
```

| Scope | N | Malignant | Benign | Marker-positive | Marker-negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| `main_yes` | 183 | 107 | 76 | 18 | 165 |
| `sensitivity_yes_cautious` | 279 | 141 | 138 | 22 | 257 |
| `excluded_no` | 11 | 6 | 5 | 3 | 8 |
| `all_images` | 290 | 147 | 143 | 25 | 265 |

Interpretation:

```text
The primary overlap analysis uses 183 strict mask-usable cases.
The sensitivity analysis adds 96 cautious cases, giving 279 total images.
Only 11 images are excluded as not usable for overlap.
```

## Main Overlap Results: usable_for_overlap = yes

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
```

Ranking uses the mean metric value. Higher is better except
`peak_distance_px`.

| Metric | Rank 1 | Rank 2 | Interpretation |
| --- | --- | --- | --- |
| `energy_inside_mask` | `ConvNeXt_Base` 0.754565 | `ResNet50` 0.671732 | ConvNeXt concentrates more saliency energy inside the mask. |
| `normalized_energy` | `ConvNeXt_Base` 1.40647 | `ResNet50` 1.26175 | ConvNeXt has the strongest mask-normalized enrichment. |
| `peak_in_mask` | `ConvNeXt_Base` 0.792350 | `ResNet50` 0.743169 | ConvNeXt most often places the peak inside the mask. |
| `peak_distance_px` | `ResNet50` 3.31036 | `ConvNeXt_Base` 3.35891 | ResNet50 is fractionally better; the two are very close. |
| `mask_coverage_top20` | `ConvNeXt_Base` 0.290899 | `ResNet50` 0.278347 | ConvNeXt top-20% saliency covers slightly more mask area. |
| `iou_top20` | `ConvNeXt_Base` 0.272865 | `ResNet50` 0.263546 | ConvNeXt has the highest top-20% IoU. |
| `dice_top20` | `ConvNeXt_Base` 0.421089 | `ResNet50` 0.401878 | ConvNeXt has the highest top-20% Dice. |

Main-set ordering:

```text
Most metrics:
  ConvNeXt_Base > ResNet50 > Swin_Base > ViT_B16

peak_distance_px:
  ResNet50 ~= ConvNeXt_Base << Swin_Base < ViT_B16
```

## Sensitivity Results: usable_for_overlap in {yes, cautious}

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
```

| Metric | Rank 1 | Rank 2 | Interpretation |
| --- | --- | --- | --- |
| `energy_inside_mask` | `ConvNeXt_Base` 0.744003 | `ResNet50` 0.636575 | ConvNeXt remains highest. |
| `normalized_energy` | `ConvNeXt_Base` 1.41026 | `ResNet50` 1.21138 | ConvNeXt remains highest. |
| `peak_in_mask` | `ConvNeXt_Base` 0.792115 | `ResNet50` 0.702509 | ConvNeXt remains highest. |
| `peak_distance_px` | `ConvNeXt_Base` 3.21054 | `ResNet50` 3.96829 | ConvNeXt becomes best after adding cautious masks. |
| `mask_coverage_top20` | `ConvNeXt_Base` 0.290894 | `ResNet50` 0.262150 | ConvNeXt remains highest. |
| `iou_top20` | `ConvNeXt_Base` 0.271164 | `ResNet50` 0.245369 | ConvNeXt remains highest. |
| `dice_top20` | `ConvNeXt_Base` 0.418542 | `ResNet50` 0.376964 | ConvNeXt remains highest. |

Interpretation:

```text
Adding cautious masks does not reverse the model ranking.
ConvNeXt_Base remains the strongest localization model across all seven
metrics in the yes+cautious sensitivity analysis.
```

## Peak-Distance Validity Check

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_peak_distance_validity_by_scope.csv
```

| Model | Scope | N images | Valid peak distance | Invalid | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| `ResNet50` | `main_yes` | 183 | 167 | 16 | Peak distance was computed only for valid non-zero saliency maps. |
| `ViT_B16` | `main_yes` | 183 | 183 | 0 |  |
| `ConvNeXt_Base` | `main_yes` | 183 | 183 | 0 |  |
| `Swin_Base` | `main_yes` | 183 | 183 | 0 |  |
| `ResNet50` | `sensitivity_yes_cautious` | 279 | 250 | 29 | Peak distance was computed only for valid non-zero saliency maps. |
| `ViT_B16` | `sensitivity_yes_cautious` | 279 | 279 | 0 |  |
| `ConvNeXt_Base` | `sensitivity_yes_cautious` | 279 | 279 | 0 |  |
| `Swin_Base` | `sensitivity_yes_cautious` | 279 | 279 | 0 |  |
| `ResNet50` | `excluded_no` | 11 | 11 | 0 |  |
| `ViT_B16` | `excluded_no` | 11 | 11 | 0 |  |
| `ConvNeXt_Base` | `excluded_no` | 11 | 11 | 0 |  |
| `Swin_Base` | `excluded_no` | 11 | 11 | 0 |  |
| `ResNet50` | `all_images` | 290 | 261 | 29 | Peak distance was computed only for valid non-zero saliency maps. |
| `ViT_B16` | `all_images` | 290 | 290 | 0 |  |
| `ConvNeXt_Base` | `all_images` | 290 | 290 | 0 |  |
| `Swin_Base` | `all_images` | 290 | 290 | 0 |  |

Validation:

```text
All invalid ResNet50 peak_distance_px rows were nan values.
All invalid ResNet50 rows matched:
  zero_saliency_any_model = True
  ResNet50_energy_inside_mask = 0

Counts:
  main_yes invalid = 16/183
  yes_cautious invalid = 29/279
  all_images invalid = 29/290
```

Reporting implication:

```text
Do not describe ResNet50 as clearly superior for peak_distance_px. In the main
set, ResNet50 and ConvNeXt_Base are very close, and ResNet50's mean is computed
only among valid non-zero saliency maps.

Use conservative wording:
  ResNet50 and ConvNeXt_Base had similarly small peak-to-mask distances among
  images with valid saliency peaks.
```

## Excluded Set

The excluded set is:

```text
usable_for_overlap == no
n = 11
```

Interpretation:

```text
The excluded set is not appropriate for primary model comparison. Broad,
cropped, truncated, or contour-like masks can produce misleading numerical
overlap values. These rows should be discussed as mask-quality limitations.
```

## Step 2: Bright-Dot / Marker Sensitivity Rerun

Script:

```text
gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py
```

Updated option:

```text
--usable-scope all
--usable-scope main_yes
--usable-scope yes_cautious
```

Commands:

```text
python3 gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py \
  --usable-scope yes_cautious \
  --output-dir gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious

python3 gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py \
  --usable-scope main_yes \
  --output-dir gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes
```

Output folders:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious/
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes/
```

Each folder contains:

```text
bright_dot_marker_classification_sensitivity_by_model.csv
bright_dot_marker_classification_performance_by_model.csv
bright_dot_marker_classification_performance_delta_by_model.csv
bright_dot_marker_overlap_sensitivity_by_model.csv
bright_dot_marker_exclusion_delta_by_model.csv
bright_dot_marker_sensitivity_summary.md
```

## How to Read the Bright-Dot Tables

`bright_dot_marker_classification_sensitivity_by_model.csv`:

```text
Within each output folder, all_images means all rows after the usable-scope
prefilter, not necessarily all 290 test images.

Scopes:
  all_images = all rows in the selected usable scope
  marker_negative = bright_dot_or_marker == none
  marker_positive = bright_dot_or_marker != none
  marker_positive_benign = marker-positive benign rows only
  marker_positive_malignant = marker-positive malignant rows only
```

`bright_dot_marker_exclusion_delta_by_model.csv`:

```text
comparison = marker_negative_minus_all_images

delta_after_excluding_marker:
  marker_negative - all_images

Use this table to decide whether excluding marker-positive images changes
classification behavior or average overlap metrics.
```

`bright_dot_marker_overlap_sensitivity_by_model.csv`:

```text
One row per scope x model x overlap metric.
Use this table to compare marker-negative and marker-positive overlap metrics.
```

## Bright-Dot Counts After Filtering

| Usable scope | Analyzed rows | Marker-positive | Marker-negative |
| --- | ---: | ---: | ---: |
| `main_yes` | 183 | 18 | 165 |
| `yes_cautious` | 279 | 22 | 257 |
| full table context | 290 | 25 | 265 |

Interpretation:

```text
Three marker-positive rows are in usable_for_overlap = no and are therefore
excluded from the yes+cautious rerun.
```

## Bright-Dot Classification Results: main_yes

Read from:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes/bright_dot_marker_classification_sensitivity_by_model.csv
```

Marker-positive subset:

| Model | N | TP | FN | FP | TN | Benign FP rate | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | 18 | 11 | 3 | 2 | 2 | 0.50 | 0.50 |
| `ViT_B16` | 18 | 11 | 3 | 2 | 2 | 0.50 | 0.50 |
| `ConvNeXt_Base` | 18 | 12 | 2 | 3 | 1 | 0.75 | 0.25 |
| `Swin_Base` | 18 | 12 | 2 | 2 | 2 | 0.50 | 0.50 |

Key readout:

```text
ConvNeXt_Base has the highest benign false-positive rate on marker-positive
main-set images: 3/4 benign marker-positive images are FP.
```

## Bright-Dot Classification Results: yes_cautious

Read from:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious/bright_dot_marker_classification_sensitivity_by_model.csv
```

Marker-positive subset:

| Model | N | TP | FN | FP | TN | Benign FP rate | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | 22 | 13 | 3 | 3 | 3 | 0.50 | 0.50 |
| `ViT_B16` | 22 | 13 | 3 | 2 | 4 | 0.333333 | 0.666667 |
| `ConvNeXt_Base` | 22 | 14 | 2 | 5 | 1 | 0.833333 | 0.166667 |
| `Swin_Base` | 22 | 14 | 2 | 2 | 4 | 0.333333 | 0.666667 |

Key readout:

```text
ConvNeXt_Base remains the strongest bright-dot / marker artifact-sensitivity
concern in the yes+cautious sensitivity set: 5/6 benign marker-positive images
are FP.
```

## Marker Exclusion Delta

Read from:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes/bright_dot_marker_exclusion_delta_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious/bright_dot_marker_exclusion_delta_by_model.csv
```

Main-set delta after excluding marker-positive rows:

| Model | Benign FP-rate delta | Specificity delta | Mean IoU delta | Mean Dice delta | Mean normalized-energy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.004386 | 0.004386 | -0.001586 | -0.002391 | -0.009300 |
| `ViT_B16` | -0.019737 | 0.019737 | -0.000916 | -0.001536 | 0.001672 |
| `ConvNeXt_Base` | -0.028509 | 0.028509 | 0.002889 | 0.003267 | 0.014050 |
| `Swin_Base` | -0.023392 | 0.023391 | -0.005945 | -0.008745 | -0.015710 |

Yes+cautious delta after excluding marker-positive rows:

| Model | Benign FP-rate delta | Specificity delta | Mean IoU delta | Mean Dice delta | Mean normalized-energy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.006258 | 0.006258 | -0.002080 | -0.003196 | -0.007680 |
| `ViT_B16` | -0.009223 | 0.009223 | -0.000631 | -0.001058 | 0.000977 |
| `ConvNeXt_Base` | -0.026350 | 0.026350 | 0.002233 | 0.002490 | 0.012240 |
| `Swin_Base` | -0.011858 | 0.011858 | -0.004868 | -0.007059 | -0.012200 |

Interpretation:

```text
Marker exclusion improves benign FP rate / specificity slightly, especially
for ConvNeXt_Base. However, the overlap metric deltas are very small.

For ConvNeXt_Base:
  mean IoU delta:
    main_yes = +0.002889
    yes_cautious = +0.002233

  mean Dice delta:
    main_yes = +0.003267
    yes_cautious = +0.002490

These changes are too small to explain the overall localization ranking.
```

## Experimental Analysis

Primary overlap result:

```text
ConvNeXt_Base is the strongest model for lesion-mask localization in the
current master-table analysis. It ranks first for six of seven main-set metrics
and all seven yes+cautious sensitivity metrics.

ResNet50 is consistently second on most metrics and is nearly tied with
ConvNeXt_Base for peak-to-mask distance in the main set.

Swin_Base is generally third.

ViT_B16 attention rollout is consistently weakest for lesion-mask localization,
especially for peak_in_mask and top-20% IoU/Dice.
```

Robustness to mask-QC scope:

```text
The model ordering is stable when expanding from the strict main set
(`usable_for_overlap = yes`, n=183) to the broader sensitivity set
(`usable_for_overlap in {yes, cautious}`, n=279).

Therefore, the main localization conclusion is not driven only by excluding
cautious masks.
```

Bright-dot / marker artifact interpretation:

```text
Bright-dot / marker artifacts remain a real limitation signal, especially for
ConvNeXt_Base, which has the highest marker-positive benign FP rate.

However, marker-positive rows are few:
  main_yes = 18/183
  yes_cautious = 22/279

Excluding marker-positive rows changes mean overlap metrics only minimally and
does not change the overall localization ranking.
```

Mask-quality limitation interpretation:

```text
The 11 usable_for_overlap = no rows should remain excluded from primary overlap
claims. Their masks are not reliable lesion-reference regions, so numerical
overlap can be misleading even when values appear high.

The 96 cautious rows are appropriate for sensitivity analysis, not for the
strict primary overlap denominator.
```

## Conclusion for Reporting

```text
Primary result:
  Use usable_for_overlap = yes (n=183) as the main overlap analysis.

Sensitivity result:
  Report usable_for_overlap in {yes, cautious} (n=279) as a robustness check.

Model comparison:
  ConvNeXt_Base shows the strongest lesion localization by energy concentration,
  peak-in-mask rate, and top-20% IoU/Dice. ResNet50 is the closest comparator.
  Swin_Base is weaker, and ViT_B16 attention rollout is weakest for this
  lesion-mask overlap task.

Artifact limitation:
  Bright-dot / marker artifacts are not the main driver of average overlap
  results, but they are a classification/saliency limitation. ConvNeXt_Base is
  the strongest artifact-sensitivity concern because marker-positive benign
  cases are frequently false positives and manual marker-attention review also
  showed high ConvNeXt marker attention.

Mask-QC limitation:
  Excluded masks should be discussed separately as mask-quality limitations.
  Cautious masks support the same model-ranking conclusion when included in a
  sensitivity analysis.
```

## Verification

Syntax check:

```text
python3 -m py_compile \
  gradcam_mask_overlap/tools/analyze_master_overlap_by_usable_set.py \
  gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py
```

Result:

```text
passed
```
