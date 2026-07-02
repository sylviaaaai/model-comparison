# Experiment Log: Primary Lesion-Mask Overlap Metrics

Date: 2026-06-29
Workspace: `/mnt/e/SW整合`

## Runtime Environment

Related environment notes are also recorded in:

```text
gradcam_mask_overlap/experiment_logs/2026-06-29_overlap_metrics_plan.md
gradcam_mask_overlap/experiment_logs/2026-06-29_prediction_saliency_alignment_log.md
```

Important Python environment note:

```text
System python in the current sandbox does not include the image/scientific
packages needed for rendering QC figures:
  matplotlib: missing
  numpy: missing
  Pillow/PIL: missing

Use one of the project virtual environments for QC rendering:

  /mnt/e/SW整合/.venv-wsl/bin/python
    Python = 3.11.15
    numpy = 2.4.6
    Pillow = 12.2.0
    matplotlib = 3.10.9

  /mnt/e/SW整合/.venv-torch5090/bin/python
    Python = 3.11.15
    numpy = 2.4.4
    Pillow = 12.2.0
    matplotlib = 3.10.9
```

Operational note:

```text
Set MPLCONFIGDIR=/tmp/matplotlib when rendering figures to avoid matplotlib
cache warnings caused by a non-writable /home/owner/.config/matplotlib path.
```

## Related Planning Document

The current analysis roadmap and progress checklist are tracked here:

```text
gradcam_mask_overlap/experiment_logs/overlap_analysis_progress_plan.md
```

This companion plan records the two-track limitation strategy:

```text
1. Mask annotation / mask-quality limitation
2. Bright-dot / marker artifact limitation
```

## Scope

This run computed primary lesion-mask overlap metrics from:

```text
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
```

Primary model-method pairs:

| Model | Saliency method | Explanation family | Class-specific |
| --- | --- | --- | --- |
| ResNet50 | `gradcam_malignant` | Grad-CAM | True |
| ConvNeXt-Base | `gradcam_malignant` | Grad-CAM | True |
| Swin-Base | `gradcam_malignant` | Grad-CAM | True |
| ViT-B/16 | `attention_rollout` | Attention rollout | False |

## Overlay Smoke QC

Generated case-centric QC panels:

```text
gradcam_mask_overlap/qc_figures/overlap_check/
  sampled_cases_index.csv
  TP/
  TN/
  FP/
  FN/
```

Sampling used ResNet50 prediction groups as the reference:

```text
TP = 3 image_id cases
TN = 3 image_id cases
FP = 3 image_id cases
FN = 3 image_id cases
```

Each panel renders all four primary model-method maps for the same image_id:

```text
ROI image
lesion mask
raw heatmap
heatmap overlay
top-10% binary saliency
mask + binary saliency
```

A quick visual spot-check confirmed that generated panels render correctly and
show matching ROI, lesion mask, saliency heatmap, and top-10% binary saliency
layers. Full manual review is still recommended before publication-level
interpretation.

Update after handoff recovery:

```text
2026-06-29T08:28Z UTC
```

The smoke QC panels were regenerated after tightening the zero-saliency
top-fraction handling. If a normalized saliency map is all zero, the top-10%
binary saliency panel is now empty instead of selecting the full image.

FN smoke QC replacement:

```text
2026-06-29T08:44Z UTC
```

The original FN smoke QC cases were the first three ResNet50 FN cases by
`image_id`:

```text
0005_C_0005_1.RIGHT_CC
0005_C_0005_1.RIGHT_MLO
0019_C_0019_1.LEFT_MLO
```

They were replaced with a more varied non-zero-saliency FN set:

```text
0064_C_0064_1.LEFT_CC
1724_A_1724_1.LEFT_CC
3024_B_3024_1.RIGHT_MLO
```

The active `sampled_cases_index.csv` now points to the replacement FN cases.
The original FN PNG files were moved to:

```text
gradcam_mask_overlap/qc_figures/overlap_check/FN/replaced_20260629/
```

## Metric Outputs

Scripts added:

```text
gradcam_mask_overlap/tools/render_overlap_smoke_qc.py
gradcam_mask_overlap/tools/compute_overlap_metrics.py
gradcam_mask_overlap/tools/summarize_overlap_metrics.py
```

Primary long-form output:

```text
gradcam_mask_overlap/metadata/overlap_metrics.csv
gradcam_mask_overlap/metadata/overlap_metrics_summary.json
```

Model-level summaries:

```text
gradcam_mask_overlap/metadata/overlap_continuous_model_summary.csv
gradcam_mask_overlap/metadata/overlap_threshold_model_summary.csv
gradcam_mask_overlap/metadata/overlap_best_fixed_thresholds.csv
gradcam_mask_overlap/metadata/overlap_model_summary.json
```

## Integrity Checks

Expected and observed row counts matched:

| Check | Observed | Expected |
| --- | ---: | ---: |
| Source image-method pairs | 1160 | 1160 |
| Output metric rows | 15080 | 15080 |
| Fixed-threshold rows | 10440 | 10440 |
| Top-fraction rows | 4640 | 4640 |

Rows by model-method pair:

| Model-method | Source pairs |
| --- | ---: |
| `convnext_base:gradcam_malignant` | 290 |
| `resnet50:gradcam_malignant` | 290 |
| `swin_base:gradcam_malignant` | 290 |
| `vit_b16:attention_rollout` | 290 |

Quality flags:

| Quality note | Image-method pairs |
| --- | ---: |
| `ok` | 1131 |
| `zero_saliency` | 29 |

The 29 zero-saliency image-method pairs are all from
`resnet50:gradcam_malignant`:

```text
TN = 21
FN = 8
```

These should be retained as quality-flagged rows, but model-level interpretation
should explicitly mention them because they reduce ResNet50 malignant Grad-CAM
continuous saliency metrics and thresholded overlap metrics.

Zero-saliency top-fraction handling:

```text
For zero-saliency maps, top_fraction binary maps are empty.
This avoids treating an all-zero map as if every pixel were tied for top-k.
```

Post-fix validation:

```text
metric_rows = 15080
source_image_method_pairs = 1160
fixed rows = 10440
top_fraction rows = 4640
zero-saliency image-method pairs = 29
zero-saliency top_fraction cam_area_px values = {0}
```

## Main Continuous Summary

| Model | Mean energy in mask | Mean normalized energy | Peak in mask rate | Zero saliency pairs |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-Base | 0.7433 | 1.4100 | 0.7862 | 0 |
| ResNet50 | 0.6409 | 1.2202 | 0.7069 | 29 |
| Swin-Base | 0.5773 | 1.0922 | 0.5310 | 0 |
| ViT-B/16 | 0.3812 | 0.7135 | 0.1276 | 0 |

Interpretation note: normalized energy greater than 1 means saliency energy is
more concentrated inside the lesion annotation than expected under a spatially
uniform heatmap.

## Best Fixed Threshold Summary

Best fixed thresholds are descriptive sensitivity results, not the primary
claim, because selecting the best threshold on the test set has a test-set
tuning flavor.

| Model | Best fixed threshold | Mean IoU |
| --- | ---: | ---: |
| ConvNeXt-Base | 0.1 | 0.3644 |
| ResNet50 | 0.1 | 0.3237 |
| Swin-Base | 0.1 | 0.4584 |
| ViT-B/16 | 0.1 | 0.4005 |

## Current Takeaway

For continuous localization concentration, ConvNeXt-Base has the strongest mean
normalized energy in mask among the four primary methods, followed by ResNet50
and Swin-Base. ViT-B/16 attention rollout is less concentrated inside the lesion
mask and has a much lower peak-in-mask rate.

For fixed-threshold IoU, the best threshold among the swept values is 0.1 for
all four methods; Swin-Base has the highest mean IoU at this threshold in the
current summary table.

## Handoff Summary After Interactive Review

Timestamp:

```text
2026-06-29T10:35Z UTC
```

Current state of the overlap stage:

1. The primary overlap analysis has been computed for the four selected
   model-method pairs:

```text
resnet50:gradcam_malignant
convnext_base:gradcam_malignant
swin_base:gradcam_malignant
vit_b16:attention_rollout
```

2. The output metrics include continuous localization metrics and binary
   overlap metrics:

```text
continuous:
  cam_energy_in_mask
  normalized_energy_in_mask
  peak_in_mask
  peak_x / peak_y
  peak_distance_to_mask_px
  mask_area_px
  mask_area_fraction
  cam_sum

thresholded:
  fixed thresholds = 0.1 through 0.9
  top_fraction = top_10pct, top_20pct, top_30pct, top_40pct
  iou
  dice
  mask_coverage_by_cam
  saliency_precision
```

3. `normalized_energy_in_mask` is retained as the main continuous metric.
   Best fixed threshold results are retained only as supplementary/sensitivity
   results, not as the main conclusion.

4. The zero-saliency top-fraction edge case was fixed in both metric computation
   and smoke QC rendering:

```text
gradcam_mask_overlap/tools/compute_overlap_metrics.py
gradcam_mask_overlap/tools/render_overlap_smoke_qc.py
```

For all-zero saliency maps, top-fraction binary maps are now empty rather than
selecting the full image due to tied zero values.

5. Smoke QC panels were generated case-centrically in:

```text
gradcam_mask_overlap/qc_figures/overlap_check/
```

The active smoke QC set contains 12 PNG panels:

```text
TP = 3
TN = 3
FP = 3
FN = 3
```

The FN smoke QC cases were manually replaced because the initial first-three
FN sample was not representative enough. The active FN cases are now:

```text
0064_C_0064_1.LEFT_CC
1724_A_1724_1.LEFT_CC
3024_B_3024_1.RIGHT_MLO
```

The replaced original FN panels are archived at:

```text
gradcam_mask_overlap/qc_figures/overlap_check/FN/replaced_20260629/
```

6. Current trusted output files:

```text
gradcam_mask_overlap/metadata/overlap_metrics.csv
gradcam_mask_overlap/metadata/overlap_metrics_summary.json
gradcam_mask_overlap/metadata/overlap_continuous_model_summary.csv
gradcam_mask_overlap/metadata/overlap_threshold_model_summary.csv
gradcam_mask_overlap/metadata/overlap_best_fixed_thresholds.csv
gradcam_mask_overlap/metadata/overlap_model_summary.json
gradcam_mask_overlap/qc_figures/overlap_check/sampled_cases_index.csv
```

7. Final row-count validation:

```text
source image-method pairs = 1160
metric rows = 15080
fixed-threshold rows = 10440
top-fraction rows = 4640
active smoke QC PNG panels = 12
```

8. Remaining manual step:

```text
Review the active 12 smoke QC panels for obvious ROI/mask/saliency spatial
misalignment before using the numerical summaries for publication-style
interpretation or figures.
```

## Selected Reporting Tables

Timestamp:

```text
2026-06-29T10:40Z UTC
```

After manual review, the active smoke QC panels were considered acceptable for
continuing with summary-table and figure preparation.

A compact selected-metrics table was added for easier plotting and
publication-style reporting:

```text
gradcam_mask_overlap/tools/build_selected_overlap_tables.py
```

This script extracts the representative binary overlap settings:

```text
fixed_0.1
top_10pct
top_20pct
```

and merges them with continuous image-method metrics:

```text
cam_energy_in_mask
normalized_energy_in_mask
peak_in_mask
peak_distance_to_mask_px
mask_area_px
mask_area_fraction
cam_sum
```

Outputs:

```text
gradcam_mask_overlap/metadata/overlap_selected_metrics_wide.csv
gradcam_mask_overlap/metadata/overlap_selected_model_summary.csv
gradcam_mask_overlap/metadata/overlap_selected_summary.json
```

Validation:

```text
source metric rows = 15080
selected wide rows = 1160
selected model summary rows = 4
```

Use `overlap_selected_metrics_wide.csv` as the preferred plotting/statistics
input when the analysis only needs continuous metrics plus `fixed_0.1`,
`top_10pct`, and `top_20pct`. Keep `overlap_metrics.csv` as the full long-form
audit table.

## ResNet50 Zero-Saliency Distribution

Timestamp:

```text
2026-06-29T11:08Z UTC
```

The 29 zero-saliency image-method pairs are all from the primary
`resnet50:gradcam_malignant` analysis path. They are not randomly distributed
across ResNet50 confusion-matrix groups.

Distribution:

| Group | Zero-saliency pairs | ResNet50 group total | Group rate |
| --- | ---: | ---: | ---: |
| TP | 0 | 89 | 0.0% |
| TN | 21 | 92 | 22.8% |
| FP | 0 | 51 | 0.0% |
| FN | 8 | 58 | 13.8% |

Class-direction view:

```text
Benign -> Benign = 21  (TN)
Cancer -> Benign = 8   (FN)
Benign -> Cancer = 0   (FP)
Cancer -> Cancer = 0   (TP)
```

Interpretation note:

```text
All 29 zero-saliency ResNet50 malignant Grad-CAM cases come from images that
ResNet50 predicted as Benign. This is consistent with cancer-targeted Grad-CAM:
when the model does not produce positive localization evidence for the Cancer
score at the selected layer, the post-ReLU Grad-CAM map can be all zero.
```

Rendered examples for visual inspection:

```text
gradcam_mask_overlap/qc_figures/zero_saliency_examples/TN/0234_C_0234_1.RIGHT_MLO_overlap_smoke_qc.png
gradcam_mask_overlap/qc_figures/zero_saliency_examples/FN/1632_A_1632_1.RIGHT_MLO_overlap_smoke_qc.png
```

## Zero-CAM Handling Finalization

Timestamp:

```text
2026-06-29T11:42Z UTC
```

The zero-saliency handling was tightened after review.

Final rules:

```text
For all-zero saliency maps:
  cam_sum = 0
  cam_energy_in_mask = 0
  normalized_energy_in_mask = 0
  peak_in_mask = False
  peak_x / peak_y = blank
  peak_distance_to_mask_px = nan
  quality_note = zero_saliency
```

For fixed thresholds and top-fraction thresholds:

```text
binary_saliency = all False
iou = 0
dice = 0
mask_coverage_by_cam = 0
saliency_precision = 0
cam_area_fraction = 0
intersection_area_px = 0
union_area_px = mask_area_px
```

Top-fraction binary masks now use exact top-k index selection instead of
percentile/quantile thresholding:

```text
k = ceil(top_fraction * n_pixels)
top_idx = argpartition(flat_cam, -k)[-k:]
```

This avoids tie inflation when many pixels share the same value.

Updated scripts:

```text
gradcam_mask_overlap/tools/compute_overlap_metrics.py
gradcam_mask_overlap/tools/render_overlap_smoke_qc.py
```

Regenerated outputs:

```text
gradcam_mask_overlap/metadata/overlap_metrics.csv
gradcam_mask_overlap/metadata/overlap_metrics_summary.json
gradcam_mask_overlap/metadata/overlap_continuous_model_summary.csv
gradcam_mask_overlap/metadata/overlap_threshold_model_summary.csv
gradcam_mask_overlap/metadata/overlap_best_fixed_thresholds.csv
gradcam_mask_overlap/metadata/overlap_model_summary.json
gradcam_mask_overlap/metadata/overlap_selected_metrics_wide.csv
gradcam_mask_overlap/metadata/overlap_selected_model_summary.csv
gradcam_mask_overlap/metadata/overlap_selected_summary.json
gradcam_mask_overlap/qc_figures/overlap_check/
gradcam_mask_overlap/qc_figures/zero_saliency_examples/
```

Validation:

```text
source image-method pairs = 1160
metric rows = 15080
zero-saliency image-method pairs = 29
zero-saliency top-fraction rows = 116
zero-saliency top-fraction cam_area_px values = {0}
zero-saliency peak_distance_to_mask_px values = {nan}
zero-saliency peak_x / peak_y values = blank
```

ResNet50 top-fraction area sanity check:

```text
top_10pct_cam_area_fraction_mean = 0.09000717471
top_20pct_cam_area_fraction_mean = 0.1800143495
```

These values are expected because 261 non-zero ResNet50 maps contribute exact
top-10% or top-20% areas, while 29 zero-saliency maps contribute 0 area.

## Top-20% Smoke QC Panels

Timestamp:

```text
2026-06-30 Asia/Shanghai
```

Added a configurable top-fraction option to:

```text
gradcam_mask_overlap/tools/render_overlap_smoke_qc.py
```

The existing active 12 smoke QC cases were re-rendered with:

```text
--top-fraction 0.20
```

Output:

```text
gradcam_mask_overlap/qc_figures/overlap_check_top20/
  sampled_cases_index.csv
  TP/
  TN/
  FP/
  FN/
```

Validation:

```text
Rendered 12 top-20% QC panels.
One TP panel was visually spot-checked and showed the expected
top-20% saliency column label and mask/saliency overlay.
```

## Image-Level Master Table

Timestamp:

```text
2026-06-30 Asia/Shanghai
```

Added a reproducible builder:

```text
gradcam_mask_overlap/tools/build_test_image_master_table.py
```

Output:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

The table is one row per test image and horizontally joins:

```text
test_images_manifest.csv
test_roi_mask_manifest.csv
model_predictions.csv
analysis_ready_overlap_manifest.csv
overlap_selected_metrics_wide.csv
```

It includes ROI image/mask metadata, manual annotation fields, rule-based
`mask_type` and `usable_for_overlap_3level`, four-model probabilities and
prediction groups, primary heatmap paths, continuous localization metrics, and
top-20% overlap metrics.

Rule-based annotation mapping:

```text
mass_like -> mass_like
small_cyst_like -> mass_like
calcification_like -> calcification_region
architectural_distortion -> diffuse
diffuse_whole_breast_or_contour -> diffuse

usable_for_overlap_manual:
  yes -> yes
  cautious -> cautious
  exclude_or_separate_analysis -> cautious
```

Validation:

```text
image-level rows = 290
has_prediction_all_models = True for 290 rows
has_heatmap_all_models = True for 290 rows
has_mask = True for 290 rows
included_in_main_overlap_analysis = True for 290 rows
zero_saliency_any_model = True for 29 rows
missing_reason is blank for 290 rows
```

## Manual Mask Review Template

Timestamp:

```text
2026-06-30 Asia/Shanghai
```

Started a new manual mask review table instead of editing the old
semi-manual/default labels in place.

Builder:

```text
gradcam_mask_overlap/tools/build_manual_mask_review_table.py
```

Output:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
gradcam_mask_overlap/metadata/manual_mask_review_template_README.md
```

The table is one row per test image and includes image/mask references:

```text
roi_image_path
roi_mask_path
qc_overlay_path
original_image_path
original_mask_path
binary_filled_mask_path
```

The five new manual review columns are intentionally blank for fresh review:

```text
manual_mask_type
mask_qc_type
bright_dot_or_marker
usable_for_overlap
notes
```

Validation:

```text
review rows = 290
missing roi_image_path = 0
missing roi_mask_path = 0
missing qc_overlay_path = 0
new manual review columns are blank for 290 rows
```

After review, this table should become the source of truth for updating:

```text
test_image_master_table.csv
analysis_ready_overlap_manifest.csv
overlap_selected_metrics_wide.csv
overlap_metrics.csv
```

## Manual Mask Review First-Pass Seeding

Timestamp:

```text
2026-06-30 Asia/Shanghai
```

Seeded the first two new manual review columns from the latest reviewer notes:

```text
gradcam_mask_overlap/tools/seed_manual_mask_review_from_notes.py
```

Updated:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
gradcam_mask_overlap/metadata/manual_mask_review_template_README.md
```

Filled columns:

```text
manual_mask_type
mask_qc_type
notes
```

Intentionally left blank:

```text
bright_dot_or_marker
usable_for_overlap
```

Applied rules:

```text
malignant mass/cyst -> localized_malignant_annotation
benign mass/cyst -> localized_mass_like
calcification -> calcification_region
architectural distortion -> architectural_distortion_like
whole-breast/large diffuse annotations -> uncertain_annotation

large lesion/tissue region -> diffuse_large
whole-breast contour without large-lesion note -> foreground_like_or_contour
border touching but complete -> border_touching
border touching and truncated/incomplete -> cropped_or_truncated
cyst-like overinclusive masks -> overinclusive_annotation
otherwise -> tight_or_reasonable
```

Validation:

```text
review rows = 290
manual_mask_type:
  localized_malignant_annotation = 119
  localized_mass_like = 84
  calcification_region = 49
  uncertain_annotation = 20
  architectural_distortion_like = 18

mask_qc_type:
  tight_or_reasonable = 246
  diffuse_large = 20
  cropped_or_truncated = 9
  border_touching = 7
  overinclusive_annotation = 6
  foreground_like_or_contour = 2

bright_dot_or_marker blank rows = 290
usable_for_overlap blank rows = 290
notes nonblank rows = 54
```

## Manual Mask Review Coding Standard

Current manual-review source of truth:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

This table is image-level, with one row per `image_id`. The old
`manual_annotation_type`, `manual_experiment_group`, and
`usable_for_overlap_manual` columns are retained only as historical references.
The new manual review should be interpreted through these columns:

```text
manual_mask_type
mask_qc_type
bright_dot_or_marker
usable_for_overlap
notes
```

### `manual_mask_type`

This column describes the apparent annotation/lesion morphology, not mask
quality.

Allowed values:

```text
localized_mass_like
calcification_region
architectural_distortion_like
localized_malignant_annotation
uncertain_annotation
```

Coding rules:

```text
benign mass-like annotations -> localized_mass_like
benign cyst-like annotations -> localized_mass_like
malignant mass-like annotations -> localized_malignant_annotation
malignant cyst-like annotations -> localized_malignant_annotation
calcification annotations -> calcification_region
architectural distortion annotations -> architectural_distortion_like
whole-breast / very diffuse / unclear morphology -> uncertain_annotation
```

Rationale:

```text
localized_malignant_annotation is used for malignant localized annotations
because malignant cases were not exhaustively subtyped into benign-style
mass/cyst categories during this review pass.
```

### `mask_qc_type`

This column describes mask usability/geometry problems, independent of lesion
morphology.

Allowed values:

```text
tight_or_reasonable
overinclusive_annotation
border_touching
diffuse_large
foreground_like_or_contour
cropped_or_truncated
bad_or_mismatched
```

Coding rules:

```text
normal localized annotation -> tight_or_reasonable
mask much larger than apparent lesion -> overinclusive_annotation
cyst-like uncertain mask region -> overinclusive_annotation
border touching but lesion appears complete -> border_touching
border touching with incomplete/truncated lesion -> cropped_or_truncated
large lesion/tissue region -> diffuse_large
whole-breast contour-like annotation without explicit large-lesion note -> foreground_like_or_contour
wrong region / mismatched ROI-mask / unusable mask -> bad_or_mismatched
```

Priority rule when multiple issues apply:

```text
bad_or_mismatched
cropped_or_truncated
diffuse_large
foreground_like_or_contour
overinclusive_annotation
border_touching
tight_or_reasonable
```

In practice, if a case is both border-touching and truncated, code it as:

```text
cropped_or_truncated
```

and describe the border side in `notes`.

### `bright_dot_or_marker`

This column is reserved for bright-dot / marker artifacts. It has not been
filled in the current pass.

Allowed values:

```text
none
inside_mask
outside_mask
on_border
uncertain
```

Current status:

```text
blank for all 290 rows
```

### `usable_for_overlap`

This column is the final analysis decision. It has not been filled in the
current pass.

Allowed values:

```text
yes
cautious
no
```

Planned interpretation:

```text
yes:
  localized, clear, and reasonably tight masks

cautious:
  calcification regions
  architectural distortion-like annotations
  mildly overinclusive masks
  mild border touching
  uncertain but still interpretable masks
  bright-dot / marker issues that do not fully invalidate the case

no:
  whole-breast foreground/contour masks
  very diffuse masks with little localization meaning
  cropped/truncated lesions that make overlap unfair
  bad or mismatched masks
```

The `usable_for_overlap` column should be filled after reviewing
`manual_mask_type`, `mask_qc_type`, `bright_dot_or_marker`, and `notes`
together.

### `notes`

This free-text column records case-specific details, especially:

```text
border side
whether the lesion is complete or truncated
whether the mask is much larger than the apparent lesion
whether the annotation is visually uncertain
any marker/bright-dot observations once reviewed
```

Examples:

```text
RIGHT CC right border touching; mass truncated
LEFT CC border touching; mass complete
large lesion/tissue region
small cyst-like annotation; mask likely overinclusive
unclear visually
```

### Downstream Use

The new review table has not yet been propagated into the analysis outputs.
After `bright_dot_or_marker` and `usable_for_overlap` are reviewed, update
downstream tables by joining on:

```text
image_id
```

Target tables:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
gradcam_mask_overlap/metadata/overlap_selected_metrics_wide.csv
gradcam_mask_overlap/metadata/overlap_metrics.csv
```

## Bright-Dot / Marker Review Update

Timestamp:

```text
2026-06-30 Asia/Shanghai
```

Updated `bright_dot_or_marker` in:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

using:

```text
gradcam_mask_overlap/tools/update_bright_dot_marker_review.py
```

The update applied 25 image-level bright-dot / marker annotations from reviewer
notes.

Validation:

```text
bright_dot_or_marker:
  blank = 265
  outside_mask = 13
  inside_mask = 9
  on_border = 2
  uncertain = 1

usable_for_overlap blank rows = 290
manual_mask_type and mask_qc_type blank rows = 0
```

## Manual Review Fill Workflow And Bright-Dot Finalization

Timestamp:

```text
2026-06-30 Asia/Shanghai
```

Filled the remaining blank `bright_dot_or_marker` cells as `none` using:

```text
gradcam_mask_overlap/tools/finalize_bright_dot_marker_none.py
```

The manual review table was built and filled in stages:

```text
build_manual_mask_review_table.py
  Creates manual_mask_review_template.csv from the image-level master table,
  mask manifest, and previous manual-label references. Adds image/mask paths
  for review. Does not make final new manual decisions.

seed_manual_mask_review_from_notes.py
  Applies the latest reviewer morphology and mask-QC notes. Fills:
    manual_mask_type
    mask_qc_type
    notes
  Leaves bright_dot_or_marker and usable_for_overlap unchanged.

update_bright_dot_marker_review.py
  Applies the explicitly listed image-level bright-dot / marker annotations.
  Fills:
    bright_dot_or_marker
  only for listed cases.

finalize_bright_dot_marker_none.py
  Converts all remaining blank bright_dot_or_marker cells to:
    none
```

Current manual review table status:

```text
review rows = 290

manual_mask_type:
  localized_malignant_annotation = 119
  localized_mass_like = 84
  calcification_region = 49
  uncertain_annotation = 20
  architectural_distortion_like = 18

mask_qc_type:
  tight_or_reasonable = 246
  diffuse_large = 20
  cropped_or_truncated = 9
  border_touching = 7
  overinclusive_annotation = 6
  foreground_like_or_contour = 2

bright_dot_or_marker:
  none = 265
  outside_mask = 13
  inside_mask = 9
  on_border = 2
  uncertain = 1

usable_for_overlap blank rows = 290
bright_dot_or_marker blank rows = 0
manual_mask_type / mask_qc_type blank rows = 0
```

Important downstream note:

```text
manual_mask_review_template.csv is now the active manual-review working table,
but its new columns have not yet been propagated into the analysis tables.
Do not use the old manual_annotation_type columns as final labels once this
review table is completed.
```

## Bright-Dot / Marker Limitation Analysis

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Generated a limitation-analysis folder for all images where:

```text
bright_dot_or_marker != none
```

Script:

```text
gradcam_mask_overlap/tools/analyze_bright_dot_marker_cases.py
```

Output folder:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/
```

Files:

```text
README.md
bright_dot_cases_confusion_wide.csv
bright_dot_cases_by_model_long.csv
bright_dot_confusion_summary_by_model.csv
bright_dot_confusion_counts_simple.csv
bright_dot_image_figure_index.csv
figures_top20/
```

The wide table is one row per marker case and lists the four-model confusion
matrix group (`TP`, `FN`, `FP`, `TN`), malignant probability, and top-20%
overlap metrics. The long table is one row per image-model pair. The summary
table contains TP/FN/FP/TN counts by model and marker location.

Follow-up clarification:

```text
bright_dot_confusion_counts_simple.csv
  Simplest table for the paper-limitation question:
  among bright-dot/marker images, how many TP/FN/FP/TN each model has.

bright_dot_image_figure_index.csv
  One row per bright-dot/marker image, with four model prediction groups and
  direct path to the rendered top-20% four-model QC figure.
```

Figure-title clarification:

```text
The top-20% QC figures were regenerated so each title includes all four model
confusion groups:

marker=<location> | R=<ResNet50> V=<ViT_B16> C=<ConvNeXt_Base> S=<Swin_Base> | <image_id>
```

## Bright-Dot Attention Review Template

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Created an empty manual-review table for whether each model visually attends to
the bright dot / marker in the rendered top-20% QC panel.

Script:

```text
gradcam_mask_overlap/tools/build_bright_dot_attention_review_table.py
```

Output:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_attention_review.csv
```

Rows:

```text
25 bright-dot / marker images
```

Columns to fill:

```text
ResNet50_attends_marker
ViT_B16_attends_marker
ConvNeXt_Base_attends_marker
Swin_Base_attends_marker
```

Allowed values:

```text
yes
no
uncertain
```

Optional note columns:

```text
ResNet50_attention_note
ViT_B16_attention_note
ConvNeXt_Base_attention_note
Swin_Base_attention_note
```

Validation:

```text
attention-review rows = 25
all four attends_marker columns blank for 25 rows
all four attention_note columns blank for 25 rows
```

Rendered figures:

```text
25 top-20% four-model QC panels
```

Each figure contains:

```text
ROI image
lesion mask
raw heatmap
heatmap overlay
top-20% saliency
mask + saliency
```

Marker-location counts:

```text
inside_mask = 9
on_border = 2
outside_mask = 13
uncertain = 1
```

Overall confusion counts among the 25 marker cases:

| Model | N | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet50` | 25 | 16 | 3 | 3 | 3 |
| `vit_b16` | 25 | 16 | 3 | 2 | 4 |
| `convnext_base` | 25 | 17 | 2 | 5 | 1 |
| `swin_base` | 25 | 17 | 2 | 2 | 4 |

Interpretation note:

```text
These cases are collected for limitation and sensitivity review. Marker
location does not by itself prove shortcut learning; it flags images where a
small bright dot or marker-like artifact could complicate saliency/overlap
interpretation.
```

## Bright-Dot Attention Manual Review Results

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Purpose:

```text
Use manual review of the rendered top-20% saliency panels to record whether each
model visually attends to the bright dot / marker in the 25 marker-positive
images.
```

Scripts:

```text
gradcam_mask_overlap/tools/build_bright_dot_attention_review_table.py
gradcam_mask_overlap/tools/apply_bright_dot_attention_codes.py
```

Updated / generated outputs:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_attention_review.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_attention_summary_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_attention_by_confusion.csv
```

Coding standard:

```text
1 = the model's top-20% saliency visibly attends to the bright dot / marker
0 = the model's top-20% saliency does not visibly attend to the bright dot / marker
```

Implementation note:

```text
The unused *_attention_note columns were removed from
bright_dot_attention_review.csv and from the table-building script.
```

Model-level attention summary:

| Model | N | Attends marker | Does not attend | Attention rate | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResNet50` | 25 | 12 | 13 | 48.0% | 16 | 3 | 3 | 3 |
| `ViT_B16` | 25 | 21 | 4 | 84.0% | 16 | 3 | 2 | 4 |
| `ConvNeXt_Base` | 25 | 23 | 2 | 92.0% | 17 | 2 | 5 | 1 |
| `Swin_Base` | 25 | 13 | 12 | 52.0% | 17 | 2 | 2 | 4 |

Attention by confusion group:

| Model | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| `ResNet50` | 5/16 | 2/3 | 3/3 | 2/3 |
| `ViT_B16` | 13/16 | 3/3 | 2/2 | 3/4 |
| `ConvNeXt_Base` | 15/17 | 2/2 | 5/5 | 1/1 |
| `Swin_Base` | 7/17 | 1/2 | 2/2 | 3/4 |

Interpretation:

```text
ConvNeXt_Base showed the strongest marker-attention signal: 23/25 marker cases
(92.0%) and 5/5 benign marker false-positive cases attended to the marker. This
supports describing ConvNeXt as the strongest shortcut/artifact-sensitivity
concern in this marker subset.

ViT_B16 also showed frequent marker attention: 21/25 cases (84.0%). However, its
benign marker false-positive count was lower than ConvNeXt (2 vs. 5), suggesting
that marker attention was common but did not translate into the same level of
benign false-positive behavior.

ResNet50 and Swin_Base showed lower overall marker attention rates (48.0% and
52.0%). Still, when benign marker cases became false positives, all available FP
cases attended to the marker for each model, so marker-related saliency remains
a relevant limitation rather than a ConvNeXt-only issue.

These results should be interpreted as a limitation/sanity-check analysis, not
as proof that the marker caused the predictions.
```

## Master Table Manual-QC / Bright-Dot Integration

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Purpose:

```text
Update the image-level master table so it uses the current manual mask-review
schema and carries bright-dot / marker-review information needed for downstream
subgroup and limitation analyses.
```

Script updated:

```text
gradcam_mask_overlap/tools/build_test_image_master_table.py
```

Output regenerated:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Rows / columns:

```text
290 image-level rows
69 columns
```

Manual-QC fields now included in the master table:

```text
manual_mask_type
mask_qc_type
bright_dot_or_marker
usable_for_overlap
manual_review_notes
```

Bright-dot attention fields now included in the master table:

```text
ResNet50_attends_marker
ViT_B16_attends_marker
ConvNeXt_Base_attends_marker
Swin_Base_attends_marker
```

Coding convention:

```text
For bright-dot / marker-positive images:
  1 = model visibly attends to the marker in the top-20% saliency panel
  0 = model does not visibly attend to the marker

For marker-negative images:
  blank = not applicable
```

Validation:

```text
bright_dot_or_marker counts:
  none = 265
  outside_mask = 13
  inside_mask = 9
  on_border = 2
  uncertain = 1

ResNet50_attends_marker:
  blank = 265
  1 = 12
  0 = 13

ViT_B16_attends_marker:
  blank = 265
  1 = 21
  0 = 4

ConvNeXt_Base_attends_marker:
  blank = 265
  1 = 23
  0 = 2

Swin_Base_attends_marker:
  blank = 265
  1 = 13
  0 = 12

bright-dot rows with all attention fields available = 25
marker-negative rows with any attention field filled = 0
```

Manual-field cleanup decision:

```text
The master table now keeps the current analysis-ready manual-QC fields rather
than the older first-pass manual annotation fields. The older/source manual
fields were not deleted from the project; they remain preserved in
manual_mask_review_template.csv for auditability and provenance.
```

Final inclusion-status note:

```text
usable_for_overlap has now been filled by mask-quality rule:

  yes = 183
  cautious = 96
  no = 11

included_in_main_overlap_analysis is True only for usable_for_overlap = yes.
Rows marked cautious are retained for sensitivity analysis only, with
missing_reason = cautious_sensitivity_only. Rows marked no are excluded from
overlap analysis, with missing_reason = not_usable_for_overlap.
```

## Analysis Progress Plan / Roadmap

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Created a standing progress-plan document to track the current overlap-analysis
strategy and remaining work:

```text
gradcam_mask_overlap/experiment_logs/overlap_analysis_progress_plan.md
```

Current high-level plan:

```text
Track 1: Mask annotation / mask-quality limitation
  Fields: mask_area_fraction, manual_mask_type, mask_qc_type, usable_for_overlap
  Role: decide whether the mask is suitable as an overlap reference.

Track 2: Bright-dot / marker artifact limitation
  Fields: bright_dot_or_marker and four model-specific *_attends_marker fields
  Role: assess image-artifact / shortcut-sensitivity effects separately from
  mask quality.
```

Key decision recorded:

```text
bright_dot_or_marker should not directly determine usable_for_overlap. It will
be handled as a separate artifact-sensitivity analysis.
```

## Bright-Dot Marker-Exclusion Sensitivity Analysis

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Purpose:

```text
Complete the bright-dot / marker limitation track by comparing all images,
marker-negative images, and marker-positive images for classification behavior
and overlap metrics.
```

Script:

```text
gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py
```

Inputs:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Outputs:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_marker_classification_sensitivity_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_marker_classification_performance_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_marker_classification_performance_delta_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_marker_overlap_sensitivity_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_marker_exclusion_delta_by_model.csv
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/bright_dot_marker_sensitivity_summary.md
```

Scope note:

```text
This version uses all 290 test images because usable_for_overlap is still
pending. After mask-quality usability is finalized, rerun the same analysis on
the final usable_for_overlap subset.
```

Dataset split:

```text
all_images = 290
marker_negative = 265
marker_positive = 25
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

Selected overlap deltas after excluding marker-positive images:

```text
Delta = marker_negative mean - all_images mean
```

| Model | Mean iou_top20 delta | Mean dice_top20 delta | Mean normalized_energy delta | Mean peak_in_mask delta |
| --- | ---: | ---: | ---: | ---: |
| `ResNet50` | -0.002897 | -0.004294 | -0.009700 | -0.005010 |
| `ViT_B16` | -0.001460 | -0.002440 | -0.000771 | -0.006831 |
| `ConvNeXt_Base` | +0.002877 | +0.003312 | +0.015070 | +0.017567 |
| `Swin_Base` | -0.005598 | -0.008163 | -0.013900 | -0.010279 |

Interpretation:

```text
The marker-positive subset is small and label-imbalanced, so these results are
a sensitivity / limitation check rather than causal proof.

The strongest signal is classification-related rather than global-overlap
related. Removing marker-positive images changes mean top-20 IoU/Dice only
slightly, but marker-positive benign cases have elevated FP rates.

ConvNeXt_Base remains the strongest artifact-sensitivity concern: marker
attention was frequent, benign marker-positive FP rate was highest, and
specificity improved the most after marker-positive images were excluded.

ViT_B16 also frequently attended to markers, but its marker-positive benign FP
rate was lower than ConvNeXt_Base. ResNet50 and Swin_Base showed lower overall
marker-attention rates, although marker-related FP cases still occurred.
```

Current conclusion and remaining bright-dot work:

```text
The bright-dot / marker track is complete for the all-image test set. No
additional manual bright-dot labeling is needed for the current dataset.

The key result should be interpreted as artifact sensitivity, not as a global
performance improvement experiment. Marker-positive cases are enriched for
malignant images, so marker-positive AUC/recall/F1 can look higher. The more
meaningful concern is the elevated benign marker-positive FP rate, especially
for ConvNeXt_Base.

Global overlap metrics changed only slightly after marker-positive images were
excluded, suggesting that bright-dot / marker artifacts mainly affect
classification behavior and saliency interpretation rather than driving average
overlap metrics.

After usable_for_overlap is finalized in the mask-quality track, rerun
analyze_bright_dot_marker_sensitivity.py on the final mask-usable subset to
produce the publication-ready marker-exclusion sensitivity analysis.
```

Evidence chain for the bright-dot interpretation:

```text
The statement that bright-dot / marker artifacts mainly affect classification
or shortcut interpretation, rather than average overlap metrics, comes from
three linked observations.

1. Average top-20 overlap metric changes after marker exclusion were small:

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

   These small absolute deltas do not support the stronger claim that markers
   materially shift global average overlap performance.

2. Benign false-positive rates differed more clearly between marker-negative
   and marker-positive cases:

   ResNet50 = 0.350365 vs 0.500000
   ViT_B16 = 0.116788 vs 0.333333
   ConvNeXt_Base = 0.218978 vs 0.833333
   Swin_Base = 0.072993 vs 0.333333

   This is most pronounced for ConvNeXt_Base, where 5/6 benign marker-positive
   cases were false positives.

3. Manual saliency review showed that the models often attended to markers:

   ResNet50 = 12/25 = 48.0%
   ViT_B16 = 21/25 = 84.0%
   ConvNeXt_Base = 23/25 = 92.0%
   Swin_Base = 13/25 = 52.0%

   Among marker-positive benign false-positive cases, all available FP cases
   attended to the marker:
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

## Usable For Overlap Rule Fill

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Purpose:

```text
Fill usable_for_overlap from the mask annotation / mask-quality track so that
the main overlap analysis set and sensitivity-only set are explicitly defined.
```

Script:

```text
gradcam_mask_overlap/tools/fill_usable_for_overlap_from_mask_qc.py
```

Inputs:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

Updated / generated outputs:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
gradcam_mask_overlap/metadata/test_image_master_table.csv
gradcam_mask_overlap/metadata/usable_for_overlap_rule_summary.json
gradcam_mask_overlap/metadata/usable_for_overlap_review_candidates.csv
```

Rule:

```text
yes:
  manual_mask_type in {localized_mass_like, localized_malignant_annotation}
  and mask_qc_type = tight_or_reasonable

cautious:
  manual_mask_type in {calcification_region, architectural_distortion_like,
  uncertain_annotation}
  or mask_qc_type in {overinclusive_annotation, border_touching, diffuse_large}

no:
  mask_qc_type in {foreground_like_or_contour, cropped_or_truncated,
  bad_or_mismatched}
```

Important:

```text
bright_dot_or_marker was intentionally not used to decide usable_for_overlap.
Bright-dot / marker artifacts remain a separate artifact-sensitivity track.
```

Validation:

```text
manual_mask_review_template.csv:
  yes = 183
  cautious = 96
  no = 11

test_image_master_table.csv:
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

Focused review file:

```text
gradcam_mask_overlap/metadata/usable_for_overlap_review_candidates.csv
```

This file contains only the 107 `cautious` or `no` rows and should be used for
targeted manual review before publication-level overlap analysis.

## Diffuse-Large QC Review Correction

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Context:

```text
After rendering diffuse_large QC panels, manual review found that patient 0019
had been over-broadly labeled: LEFT_CC and LEFT_MLO were localized_malignant_annotation / not
diffuse_large, while RIGHT_CC and RIGHT_MLO were the true problematic
diffuse_large side.
```

Updated rows:

```text
0019_C_0019_1.LEFT_CC:
  manual_mask_type: uncertain_annotation -> localized_malignant_annotation
  mask_qc_type: diffuse_large -> tight_or_reasonable
  usable_for_overlap: no -> yes

0019_C_0019_1.LEFT_MLO:
  manual_mask_type: uncertain_annotation -> localized_malignant_annotation
  mask_qc_type: diffuse_large -> tight_or_reasonable
  usable_for_overlap: no -> yes

0019_C_0019_1.RIGHT_CC:
  mask_qc_type: diffuse_large retained
  usable_for_overlap: no -> cautious

0019_C_0019_1.RIGHT_MLO:
  mask_qc_type: diffuse_large retained
  usable_for_overlap: no -> cautious
```

Scripts / outputs:

```text
Updated manual override logic in:
  gradcam_mask_overlap/tools/fill_usable_for_overlap_from_mask_qc.py

Regenerated:
  gradcam_mask_overlap/metadata/manual_mask_review_template.csv
  gradcam_mask_overlap/metadata/test_image_master_table.csv
  gradcam_mask_overlap/metadata/usable_for_overlap_rule_summary.json
  gradcam_mask_overlap/metadata/usable_for_overlap_review_candidates.csv
```

Updated counts:

```text
mask_qc_type:
  tight_or_reasonable = 248
  diffuse_large = 18
  cropped_or_truncated = 9
  border_touching = 7
  overinclusive_annotation = 6
  foreground_like_or_contour = 2

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

QC figures:

```text
The original diffuse_large QC folder contains pre-correction 0019 LEFT panels.
To avoid stale interpretation, a corrected folder was generated:

gradcam_mask_overlap/qc_figures/mask_qc_diffuse_large_top20_after_0019_correction/
```

## Diffuse-Large Usability Rule Update

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Rule correction:

```text
diffuse_large should be assigned to usable_for_overlap = cautious, not no.
```

Implementation:

```text
Updated gradcam_mask_overlap/tools/fill_usable_for_overlap_from_mask_qc.py:
  diffuse_large moved from NO_QC_TYPES to CAUTIOUS_QC_TYPES

Regenerated:
  gradcam_mask_overlap/metadata/manual_mask_review_template.csv
  gradcam_mask_overlap/metadata/test_image_master_table.csv
  gradcam_mask_overlap/metadata/usable_for_overlap_rule_summary.json
  gradcam_mask_overlap/metadata/usable_for_overlap_review_candidates.csv
```

Updated counts:

```text
usable_for_overlap:
  yes = 183
  cautious = 96
  no = 11

diffuse_large rows:
  total = 18
  usable_for_overlap = cautious for all 18

included_in_main_overlap_analysis:
  True = 183
  False = 107

missing_reason:
  blank = 183
  cautious_sensitivity_only = 96
  not_usable_for_overlap = 11
```

## Current Master-Table Main / Sensitivity Overlap Analysis

Timestamp:

```text
2026-07-01 Asia/Shanghai
```

Purpose:

```text
Use the current finalized image-level master table to run the main lesion-mask
overlap analysis and the mask-usability sensitivity analysis.

This analysis intentionally keeps two tracks separate:
  1. lesion-mask usability for overlap analysis
  2. bright-dot / marker artifact sensitivity

bright_dot_or_marker is not used to decide usable_for_overlap.
```

Input table:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Input-table status at this run:

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

### Step 1: Generate Scope-Level Overlap Tables

Script added:

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
overlap_usable_scope_summary.json
overlap_usable_scope_summary.md
```

Scope definitions:

```text
main_yes:
  usable_for_overlap == yes

sensitivity_yes_cautious:
  usable_for_overlap in {yes, cautious}

excluded_no:
  usable_for_overlap == no

all_images:
  all 290 images; context only, not the primary reporting set
```

Metrics summarized from the master table:

```text
energy_inside_mask
normalized_energy
peak_in_mask
peak_distance_px
mask_coverage_top20
iou_top20
dice_top20
```

### How To Read The Generated Overlap CSV Tables

`overlap_usable_scope_counts.csv`:

```text
One row per analysis scope.
Use this first to confirm the denominator for each result.

Important columns:
  n_images
  n_malignant
  n_benign
  usable_yes
  usable_cautious
  usable_no
  marker_positive
  marker_negative
```

`overlap_usable_scope_model_metric_summary.csv`:

```text
One row per scope x model x metric.

Important columns:
  scope = main_yes / sensitivity_yes_cautious / excluded_no / all_images
  model = ResNet50 / ViT_B16 / ConvNeXt_Base / Swin_Base
  metric = one of the seven overlap/localization metrics
  n_valid = non-missing values used in the summary
  mean / median / sd / min / max = descriptive statistics

Use this table for plotting or manuscript summary statistics.
```

`overlap_usable_scope_model_metric_ranks.csv`:

```text
One row per scope x metric x model rank.

Important columns:
  rank = rank by mean value within the same scope and metric
  direction = higher_is_better or lower_is_better

For all metrics higher is better except:
  peak_distance_px = lower_is_better
```

For `peak_distance_px`, `n_valid` can be lower for ResNet50 because zero-saliency
maps have no valid peak coordinate and therefore no valid peak-to-mask distance.

### Scope Counts

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
The primary overlap analysis uses 183 localized / tight-or-reasonable masks.
The sensitivity analysis adds 96 cautious masks, giving 279 mask-usable or
partially usable cases. Only 11 images are excluded as not usable for overlap.

Marker-positive images are present in both the main and sensitivity sets, but
they remain a separate artifact-sensitivity track rather than an exclusion rule.
```

### Main Analysis Results: usable_for_overlap = yes

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
```

Ranking uses the mean metric value. Higher is better except `peak_distance_px`.

| Metric | Rank 1 | Rank 2 | Main interpretation |
| --- | --- | --- | --- |
| `energy_inside_mask` | `ConvNeXt_Base` 0.754565 | `ResNet50` 0.671732 | ConvNeXt concentrates more saliency energy inside the reference mask. |
| `normalized_energy` | `ConvNeXt_Base` 1.40647 | `ResNet50` 1.26175 | ConvNeXt has the strongest mask-normalized enrichment. |
| `peak_in_mask` | `ConvNeXt_Base` 0.79235 | `ResNet50` 0.743169 | ConvNeXt most often places the peak inside the mask. |
| `peak_distance_px` | `ResNet50` 3.31036 | `ConvNeXt_Base` 3.35891 | ResNet50 is fractionally better, but the two are very close. |
| `mask_coverage_top20` | `ConvNeXt_Base` 0.290899 | `ResNet50` 0.278347 | ConvNeXt top-20% saliency covers slightly more of the mask. |
| `iou_top20` | `ConvNeXt_Base` 0.272865 | `ResNet50` 0.263546 | ConvNeXt has the highest top-20% IoU. |
| `dice_top20` | `ConvNeXt_Base` 0.421089 | `ResNet50` 0.401878 | ConvNeXt has the highest top-20% Dice. |

Complete main-set ordering:

```text
Most metrics:
  ConvNeXt_Base > ResNet50 > Swin_Base > ViT_B16

peak_distance_px:
  ResNet50 ~= ConvNeXt_Base << Swin_Base < ViT_B16
```

### Sensitivity Analysis Results: usable_for_overlap in {yes, cautious}

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
```

| Metric | Rank 1 | Rank 2 | Sensitivity interpretation |
| --- | --- | --- | --- |
| `energy_inside_mask` | `ConvNeXt_Base` 0.744003 | `ResNet50` 0.636575 | ConvNeXt remains highest after adding cautious masks. |
| `normalized_energy` | `ConvNeXt_Base` 1.41026 | `ResNet50` 1.21138 | ConvNeXt remains highest. |
| `peak_in_mask` | `ConvNeXt_Base` 0.792115 | `ResNet50` 0.702509 | ConvNeXt remains highest. |
| `peak_distance_px` | `ConvNeXt_Base` 3.21054 | `ResNet50` 3.96829 | ConvNeXt becomes best when cautious masks are included. |
| `mask_coverage_top20` | `ConvNeXt_Base` 0.290894 | `ResNet50` 0.26215 | ConvNeXt remains highest. |
| `iou_top20` | `ConvNeXt_Base` 0.271164 | `ResNet50` 0.245369 | ConvNeXt remains highest. |
| `dice_top20` | `ConvNeXt_Base` 0.418542 | `ResNet50` 0.376964 | ConvNeXt remains highest. |

Sensitivity interpretation:

```text
Adding cautious masks does not reverse the main model ranking. ConvNeXt_Base
remains the strongest localization model across all seven metrics in the
yes+cautious analysis.

This supports the robustness of the main overlap result: the conclusion is not
an artifact of using only the strictest 183-mask subset.
```

### Excluded / Limitation Set

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
```

The excluded set contains only 11 images:

```text
usable_for_overlap == no
n = 11
```

Interpretation:

```text
The excluded set is not suitable for primary model comparison. Some numerical
overlap values can appear high because very broad, cropped, or contour-like
masks can inflate overlap. These rows should be discussed as mask-quality
limitations rather than used to rank models.
```

### Step 2: Rerun Bright-Dot / Marker Sensitivity On Mask-Usable Sets

Updated script:

```text
gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py
```

New option:

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

Generated output folders:

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

### How To Read The Bright-Dot Sensitivity CSV Tables

`bright_dot_marker_classification_sensitivity_by_model.csv`:

```text
Within each output folder, all_images means all rows after the usable-scope
prefilter, not necessarily all 290 test images.

Scopes:
  all_images = all rows in that usable-scope run
  marker_negative = bright_dot_or_marker == none
  marker_positive = bright_dot_or_marker != none
  marker_positive_benign = marker-positive benign rows only
  marker_positive_malignant = marker-positive malignant rows only

Use this table to inspect TP/FN/FP/TN and benign FP rate by model.
```

`bright_dot_marker_exclusion_delta_by_model.csv`:

```text
comparison = marker_negative_minus_all_images

delta_after_excluding_marker:
  marker_negative - all_images

marker_positive_minus_marker_negative:
  marker_positive - marker_negative

Use this table to determine whether excluding marker-positive images changes
classification behavior or overlap summaries.
```

`bright_dot_marker_overlap_sensitivity_by_model.csv`:

```text
One row per scope x model x overlap metric.
Use this table to compare average overlap metrics between marker-negative and
marker-positive images.
```

### Bright-Dot Counts After Usable-Scope Filtering

| Usable scope | Analyzed rows | Marker-positive | Marker-negative |
| --- | ---: | ---: | ---: |
| `main_yes` | 183 | 18 | 165 |
| `yes_cautious` | 279 | 22 | 257 |
| full table context | 290 | 25 | 265 |

Interpretation:

```text
Three marker-positive rows are in usable_for_overlap = no and are therefore
excluded from the yes+cautious mask-usable rerun.
```

### Bright-Dot Classification Results: main_yes

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

### Bright-Dot Classification Results: yes_cautious

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

### Bright-Dot Exclusion Delta

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
Marker exclusion improves benign FP rate / specificity slightly, especially for
ConvNeXt_Base, but overlap metric deltas are very small.

For ConvNeXt_Base, excluding marker-positive images changes mean IoU by only:
  main_yes: +0.002889
  yes_cautious: +0.002233

and mean Dice by only:
  main_yes: +0.003267
  yes_cautious: +0.002490

These changes are too small to explain the overall model ranking.
```

### Experimental Analysis

Primary overlap result:

```text
ConvNeXt_Base is the strongest model for lesion-mask localization in the
current master-table analysis. It ranks first for six of seven main-set metrics
and all seven yes+cautious sensitivity metrics.

ResNet50 is consistently second on most metrics and is nearly tied with
ConvNeXt_Base for peak-to-mask distance in the main set.

Swin_Base is generally third.

ViT_B16 attention rollout is consistently weakest for lesion-mask localization,
especially for peak_in_mask and top-20% overlap metrics.
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
  main_yes: 18/183
  yes_cautious: 22/279

Excluding marker-positive rows changes mean overlap metrics only minimally and
does not change the overall localization ranking.
```

Mask-quality limitation interpretation:

```text
The 11 usable_for_overlap = no rows should remain excluded from primary overlap
claims. Their masks are not reliable lesion-reference regions, so numerical
overlap can be misleading even when values look high.

The 96 cautious rows are appropriate for sensitivity analysis, not for the
strict primary overlap denominator.
```

### Conclusion For Current Reporting

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
  Bright-dot / marker artifacts are not the main driver of the average overlap
  results, but they are a classification/saliency limitation. ConvNeXt_Base is
  the strongest artifact-sensitivity concern because marker-positive benign
  cases are frequently false positives and manual marker-attention review also
  showed high ConvNeXt marker attention.

Mask-QC limitation:
  Excluded masks should be discussed separately as mask-quality limitations.
  Cautious masks support the same model-ranking conclusion when included in a
  sensitivity analysis.
```

Verification:

```text
python3 -m py_compile \
  gradcam_mask_overlap/tools/analyze_master_overlap_by_usable_set.py \
  gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py

Result: passed
```
