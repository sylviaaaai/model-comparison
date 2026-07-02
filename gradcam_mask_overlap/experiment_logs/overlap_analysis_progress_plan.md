# Overlap Analysis Progress Plan

Last updated: 2026-07-01 Asia/Shanghai

## Current Strategy

The explainability / overlap analysis will keep two limitation tracks separate:

1. Mask annotation / mask-quality limitation
2. Bright-dot / marker artifact limitation

This separation is intentional. Mask-quality issues affect whether the lesion
mask is a valid reference for overlap metrics. Bright-dot / marker issues are
image-artifact and possible shortcut-sensitivity issues, so they should be
analyzed separately rather than folded directly into mask usability.

## Track 1: Mask Annotation / Mask-Quality Limitation

Purpose:

```text
Assess whether each ROI mask is suitable as the reference region for Grad-CAM /
saliency overlap analysis.
```

Main fields:

```text
mask_area_fraction
manual_mask_type
mask_qc_type
usable_for_overlap
```

Current manual-mask fields:

```text
manual_mask_type:
  localized_mass_like
  calcification_region
  architectural_distortion_like
  localized_malignant_annotation
  uncertain_annotation

mask_qc_type:
  tight_or_reasonable
  overinclusive_annotation
  border_touching
  diffuse_large
  foreground_like_or_contour
  cropped_or_truncated
  bad_or_mismatched
```

Planned `usable_for_overlap` rule:

```text
yes:
  localized_mass_like / localized_malignant_annotation
  + tight_or_reasonable

cautious:
  calcification_region
  architectural_distortion_like
  uncertain_annotation
  overinclusive_annotation
  border_touching
  diffuse_large

no:
  foreground_like_or_contour
  cropped_or_truncated
  bad_or_mismatched
```

Important decision:

```text
bright_dot_or_marker should not directly determine usable_for_overlap.
Bright dots are handled in Track 2.
```

Planned analyses:

```text
Main analysis:
  usable_for_overlap = yes

Sensitivity analysis:
  usable_for_overlap in {yes, cautious}

Excluded / limitation discussion:
  usable_for_overlap = no
```

## Track 2: Bright-Dot / Marker Artifact Limitation

Purpose:

```text
Assess whether small bright-dot / marker-like artifacts attract model saliency
and whether excluding marker-positive cases changes model comparison results.
```

Main fields:

```text
bright_dot_or_marker
ResNet50_attends_marker
ViT_B16_attends_marker
ConvNeXt_Base_attends_marker
Swin_Base_attends_marker
```

Coding:

```text
bright_dot_or_marker:
  none
  inside_mask
  outside_mask
  on_border
  uncertain

*_attends_marker:
  1 = model visibly attends to marker in top-20% saliency panel
  0 = model does not visibly attend to marker
  blank = not applicable because no marker is present
```

Current marker counts:

```text
none = 265
outside_mask = 13
inside_mask = 9
on_border = 2
uncertain = 1
```

Current marker-attention summary:

| Model | Marker cases | Attends marker | Attention rate |
| --- | ---: | ---: | ---: |
| ResNet50 | 25 | 12 | 48.0% |
| ViT_B16 | 25 | 21 | 84.0% |
| ConvNeXt_Base | 25 | 23 | 92.0% |
| Swin_Base | 25 | 13 | 52.0% |

Interpretation so far:

```text
ConvNeXt_Base shows the strongest bright-dot / marker artifact sensitivity
signal. ViT_B16 also frequently attends to markers, but its benign marker
false-positive count is lower than ConvNeXt. ResNet50 and Swin_Base show lower
overall marker-attention rates.

This is a limitation / sanity-check result, not proof that the marker caused
the model prediction.
```

Planned analyses:

```text
All mask-usable cases:
  include bright_dot_or_marker != none

Marker-excluded sensitivity analysis:
  bright_dot_or_marker = none

Bright-dot limitation subset:
  bright_dot_or_marker != none
```

## Master Table Policy

Current master table:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

The master table should contain analysis-ready fields only:

```text
manual_mask_type
mask_qc_type
bright_dot_or_marker
usable_for_overlap
manual_review_notes
ResNet50_attends_marker
ViT_B16_attends_marker
ConvNeXt_Base_attends_marker
Swin_Base_attends_marker
```

Older first-pass manual annotation fields should not be kept in the master
table, to avoid mixing old and current annotation rules. They remain preserved
in:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

## Current Status

Completed:

```text
[x] Built image-level master table.
[x] Generated top-20% overlap / saliency QC figures.
[x] Built manual mask-review template.
[x] Filled manual_mask_type and mask_qc_type from user review notes.
[x] Filled bright_dot_or_marker for all 290 images.
[x] Rendered bright-dot / marker top-20% figure set.
[x] Built and filled bright-dot attention review table.
[x] Generated bright-dot confusion and attention summaries.
[x] Integrated current manual-QC and bright-dot fields into master table.
[x] Ran all-image bright-dot marker-exclusion sensitivity analysis.
[x] Recorded bright-dot interpretation and remaining-work decision.
[x] Filled usable_for_overlap from manual_mask_type and mask_qc_type.
[x] Regenerated master table after usable_for_overlap was filled.
[x] Applied 0019 diffuse_large manual correction after QC figure review.
[x] Ran current master-table main overlap analysis on usable_for_overlap = yes.
[x] Ran current master-table sensitivity analysis on usable_for_overlap in {yes, cautious}.
[x] Reran bright-dot marker sensitivity analysis on usable_for_overlap in {yes, cautious}.
[x] Reran bright-dot marker sensitivity analysis on usable_for_overlap = yes.
[x] Ran paired overlap model comparison with bootstrap CIs.
[x] Ran exploratory overlap subgroup analysis by true label and model-specific confusion group.
[x] Built compact paper-ready subgroup tables.
[x] Ran validation-selected classification threshold analysis and paired test AUC comparison.
```

Pending:

```text
[ ] Manually review usable_for_overlap_review_candidates.csv.
[ ] Regenerate master table after any manual usable_for_overlap edits.
[ ] Compare whether marker exclusion changes model rankings or key metrics on the final usable set.
[ ] Summarize mask-quality limitation and bright-dot artifact limitation for manuscript.
```

Bright-dot track status:

```text
The all-image bright-dot / marker artifact analysis is complete. No additional
manual marker labeling is currently needed. A current mask-usable rerun has
also been generated for usable_for_overlap in {yes, cautious}; rerun it only if
manual usable_for_overlap labels change.
```

Latest mask-usability counts:

```text
usable_for_overlap:
  yes = 183
  cautious = 96
  no = 11

included_in_main_overlap_analysis:
  True = 183
  False = 107
```

Recent mask-QC correction:

```text
0019 LEFT_CC and LEFT_MLO were removed from diffuse_large after manual QC
review, set to localized_malignant_annotation, and included in the yes main-analysis set. 0019 RIGHT_CC and RIGHT_MLO remain diffuse_large
but were changed from no to cautious.
```

Current analysis outputs:

```text
Paper-facing results index:
  gradcam_mask_overlap/paper_draft_materials/00_results_index.md

Experiment closeout:
  gradcam_mask_overlap/paper_draft_materials/01_experiment_closeout.md

Standalone latest-analysis log:
  gradcam_mask_overlap/experiment_logs/2026-07-01_master_table_overlap_and_marker_sensitivity_log.md

Paired model-comparison log:
  gradcam_mask_overlap/experiment_logs/2026-07-01_paired_overlap_model_comparison_log.md

Exploratory subgroup-analysis log:
  gradcam_mask_overlap/experiment_logs/2026-07-02_overlap_subgroup_analysis_log.md

Main/sensitivity overlap summaries:
  gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/
  gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_peak_distance_validity_by_scope.csv

Paired model-comparison summaries:
  gradcam_mask_overlap/overlap_analysis/paired_model_comparisons/

Exploratory subgroup summaries:
  gradcam_mask_overlap/overlap_analysis/subgroup_analyses/

Paper-ready compact subgroup tables:
  gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_true_label_key_metrics.csv
  gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_confusion_group_dice.csv

Classification threshold / paired AUC analysis:
  gradcam_mask_overlap/classification_analysis/threshold_auc/
  gradcam_mask_overlap/experiment_logs/2026-07-02_classification_threshold_auc_analysis_log.md

Mask-usable bright-dot sensitivity reruns:
  gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes/
  gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious/
```

## Next Best Step

```text
Manually review:

gradcam_mask_overlap/metadata/usable_for_overlap_review_candidates.csv

This file contains the 107 cautious/no rows. After any edits, regenerate the
master table before running final overlap analysis.
```
