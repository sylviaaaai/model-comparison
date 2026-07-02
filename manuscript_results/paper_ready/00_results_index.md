# Paper Results Index: Grad-CAM / Mask Overlap Analysis

Date started: 2026-07-01
Project root: `/mnt/e/SW整合/gradcam_mask_overlap`

## Purpose

This file is the paper-facing index for the current overlap-analysis results.
It does not replace the raw CSV outputs or experiment logs. Instead, it records
which files should be treated as current, what denominator each result uses,
and how each result should be framed in the manuscript.

Use this file as the first stop before writing:

```text
Methods
Results
Tables
Figures
Limitations
Supplementary material
```

## Current Source of Truth

Main analysis table:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Manual-review source table:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

Latest standalone experiment log:

```text
gradcam_mask_overlap/experiment_logs/2026-07-01_master_table_overlap_and_marker_sensitivity_log.md
```

Progress checklist:

```text
gradcam_mask_overlap/experiment_logs/overlap_analysis_progress_plan.md
```

Closeout note:

```text
gradcam_mask_overlap/paper_draft_materials/01_experiment_closeout.md
```

## Analysis Populations

Read from:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_counts.csv
```

| Scope | Definition | N | Malignant | Benign | Marker-positive | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `main_yes` | `usable_for_overlap == yes` | 183 | 107 | 76 | 18 | Main overlap analysis |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | 141 | 138 | 22 | Mask-QC sensitivity analysis |
| `excluded_no` | `usable_for_overlap == no` | 11 | 6 | 5 | 3 | Limitation / excluded cases only |
| `all_images` | all test images | 290 | 147 | 143 | 25 | Context only |

Primary reporting denominator:

```text
main_yes, n = 183
```

Sensitivity denominator:

```text
yes + cautious, n = 279
```

Do not use:

```text
all_images as the main overlap denominator
excluded_no as a model-ranking set
```

## Main Tables for Paper Writing

### Cohort and Usability Counts

Current table:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_counts.csv
```

Use for:

```text
cohort flow
main-vs-sensitivity denominators
mask-usability reporting
marker-positive counts inside each denominator
```

### Main Overlap Metrics

Current table:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_summary.csv
```

Use for:

```text
mean / median / SD summaries
plots
supplementary metric table
```

Metrics:

```text
energy_inside_mask
normalized_energy
peak_in_mask
peak_distance_px
mask_coverage_top20
iou_top20
dice_top20
```

### Model Rankings

Current table:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
```

Use for:

```text
quick model-order checks
Results narrative
ranking robustness between main and sensitivity analyses
```

Important:

```text
higher is better for all metrics except peak_distance_px.
peak_distance_px is lower-is-better.
```

### Paired Model Comparisons

Current folder:

```text
gradcam_mask_overlap/overlap_analysis/paired_model_comparisons/
```

Latest log:

```text
gradcam_mask_overlap/experiment_logs/2026-07-01_paired_overlap_model_comparison_log.md
```

Main paper-facing summary:

```text
paired_overlap_convnext_vs_others_summary.csv
```

Records:

```text
scope x metric x comparator_model

ConvNeXt_Base is always model_a / reference_model.
Positive effect_delta always favors ConvNeXt_Base.
Bootstrap 95% CI uses 5000 resamples, seed = 20260701.
```

Supplementary all-pair summary:

```text
paired_overlap_all_model_pairs_summary.csv
```

Records:

```text
scope x metric x unordered model pair

Positive effect_delta favors model_a.
Use this for supplementary all-model comparisons and audit checks.
```

Per-image paired-delta table:

```text
paired_overlap_convnext_vs_others_long.csv
```

Records:

```text
scope x image_id x metric x comparator_model

Use this for paired-delta plots or case-level inspection.
```

Human-readable summary:

```text
paired_overlap_model_comparison_summary.md
```

Key paired-analysis interpretation:

```text
ConvNeXt_Base vs ViT_B16:
  large and stable positive paired differences across main and sensitivity sets.

ConvNeXt_Base vs Swin_Base:
  stable positive paired differences across most/all key metrics.

ConvNeXt_Base vs ResNet50:
  main_yes:
    clear paired advantage for energy_inside_mask and normalized_energy.
    top-20% IoU/Dice differences are small and bootstrap CIs cross 0.

  yes_cautious:
    positive paired differences for energy, normalized energy, peak_in_mask,
    mask_coverage_top20, iou_top20, and dice_top20.

  peak_distance_px:
    interpret conservatively because ResNet50 has invalid zero-saliency rows
    and ConvNeXt-vs-ResNet50 CIs cross 0.
```

### Exploratory Subgroup Analysis

Current folder:

```text
gradcam_mask_overlap/overlap_analysis/subgroup_analyses/
```

Latest log:

```text
gradcam_mask_overlap/experiment_logs/2026-07-02_overlap_subgroup_analysis_log.md
```

Tables:

```text
overlap_by_true_label_summary.csv
overlap_by_confusion_group_summary.csv
overlap_by_true_label_and_confusion_group_summary.csv
overlap_subgroup_counts.csv
overlap_subgroup_analysis_summary.md
```

Paper-ready compact tables:

```text
gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_true_label_key_metrics.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_confusion_group_dice.csv
```

Records:

```text
overlap_by_true_label_summary.csv:
  scope x true_label x model x metric

overlap_by_confusion_group_summary.csv:
  scope x model-specific TP/FN/FP/TN x model x metric

overlap_by_true_label_and_confusion_group_summary.csv:
  scope x true_label x model-specific TP/FN/FP/TN x model x metric

overlap_subgroup_counts.csv:
  denominators for every subgroup
```

Use for:

```text
explaining whether overlap patterns differ by malignant vs benign images
checking TP vs FN localization
checking whether FP cases localize to benign masks
supplementary exploratory analysis
```

Do not use for:

```text
primary model ranking
formal claims from small FP/FN subgroup denominators
```

Key exploratory interpretation:

```text
ConvNeXt_Base remains high in both malignant and benign strata.
TP cases generally show higher overlap than FN cases.
FP cases often show substantial overlap with benign annotations, suggesting
that false-positive malignant predictions may still localize to annotated
benign findings.
```

### Classification Threshold / Paired AUC Analysis

Current folder:

```text
gradcam_mask_overlap/classification_analysis/threshold_auc/
```

Latest log:

```text
gradcam_mask_overlap/experiment_logs/2026-07-02_classification_threshold_auc_analysis_log.md
```

Tables:

```text
classification_validation_selected_thresholds.csv
classification_test_locked_threshold_performance.csv
classification_test_fixed_0p5_performance.csv
classification_test_auc_by_model.csv
classification_test_auc_pairwise_bootstrap.csv
classification_threshold_auc_summary.md
```

Paper-readable table versions:

```text
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_readable_tables.md
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_validation_thresholds_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_test_locked_thresholds_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_test_fixed_0p5_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_paired_auc_readable.csv
```

Records:

```text
classification_validation_selected_thresholds.csv:
  one row per model; thresholds selected on validation only

classification_test_locked_threshold_performance.csv:
  one row per model; validation-selected threshold applied to test

classification_test_auc_by_model.csv:
  one row per model; threshold-free test AUC with patient-level bootstrap CI

classification_test_auc_pairwise_bootstrap.csv:
  one row per model pair; paired patient-level bootstrap AUC difference
```

Use for:

```text
classification-performance companion table
showing threshold selection was validation-only
test-set paired AUC comparison
separating classification ranking from localization ranking
```

Key classification interpretation:

```text
ViT_B16 has the highest numerical test AUC.
ConvNeXt_Base and Swin_Base are close behind.
ResNet50 is lower than the other three under paired bootstrap.

Paired AUC differences among ViT_B16, ConvNeXt_Base, and Swin_Base cross 0,
so their ordering should be described conservatively.

This does not change the overlap conclusion: ConvNeXt_Base remains the
strongest lesion-localization / mask-overlap model.
```

### Peak-Distance Validity

Current table:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_peak_distance_validity_by_scope.csv
```

Use for:

```text
denominator explanation for peak_distance_px
reviewer-facing note about ResNet50 zero-saliency cases
supplementary methods or table footnote
```

Key values:

| Model | Scope | N images | Valid peak distance | Invalid | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| `ResNet50` | `main_yes` | 183 | 167 | 16 | Peak distance was computed only for valid non-zero saliency maps. |
| `ResNet50` | `sensitivity_yes_cautious` | 279 | 250 | 29 | Peak distance was computed only for valid non-zero saliency maps. |
| `ResNet50` | `all_images` | 290 | 261 | 29 | Peak distance was computed only for valid non-zero saliency maps. |

Writing rule:

```text
Do not claim that ResNet50 is clearly best for peak_distance_px.
Use conservative wording: ResNet50 and ConvNeXt_Base had similarly small
peak-to-mask distances among images with valid non-zero saliency maps.
```

### Bright-Dot / Marker Sensitivity

Main-set rerun:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes/
```

Sensitivity-set rerun:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious/
```

Most important files:

```text
bright_dot_marker_classification_sensitivity_by_model.csv
bright_dot_marker_exclusion_delta_by_model.csv
bright_dot_marker_overlap_sensitivity_by_model.csv
bright_dot_marker_sensitivity_summary.md
```

Use for:

```text
artifact limitation
marker-positive benign FP rate
checking whether marker exclusion changes overlap metrics
```

## Main Results Narrative

Current high-level result:

```text
ConvNeXt_Base shows the strongest lesion localization by energy concentration,
peak-in-mask rate, and top-20% IoU/Dice.

ResNet50 is the closest comparator.

Swin_Base is weaker.

ViT_B16 attention rollout is weakest for this lesion-mask overlap task.
```

Main-set result:

```text
In usable_for_overlap = yes cases (n = 183), ConvNeXt_Base ranks first for
six of seven localization metrics. ResNet50 is fractionally better for
peak_distance_px, but that metric has fewer valid ResNet50 rows because of
zero-saliency maps and should be interpreted conservatively.

Paired comparison adds nuance: ConvNeXt_Base is clearly paired-higher than
ResNet50 for energy_inside_mask and normalized_energy, but main-set top-20%
IoU/Dice paired differences are small and have bootstrap CIs crossing 0.
```

Sensitivity-set result:

```text
In usable_for_overlap in {yes, cautious} cases (n = 279), ConvNeXt_Base ranks
first for all seven metrics. This supports the robustness of the main result to
mask-QC inclusion criteria.

Paired comparison also supports ConvNeXt_Base over ResNet50 for top-20%
mask coverage, IoU, and Dice in the yes+cautious set.
```

## Bright-Dot / Marker Limitation Narrative

Current conclusion:

```text
Bright-dot / marker artifacts are a real saliency/classification limitation,
especially for ConvNeXt_Base, but they do not drive the average overlap result
or change the main localization ranking.
```

Key values:

```text
Full table marker-positive images = 25
main_yes marker-positive images = 18
yes_cautious marker-positive images = 22
```

Marker-positive benign FP counts in `yes_cautious`:

```text
ResNet50 = 3
ViT_B16 = 2
ConvNeXt_Base = 5
Swin_Base = 2
```

Marker-exclusion effect:

```text
Marker exclusion slightly improves benign FP rate / specificity, especially
for ConvNeXt_Base, but overlap metric deltas are very small.
```

ConvNeXt_Base overlap deltas after marker exclusion:

```text
main_yes:
  mean IoU delta = +0.002889
  mean Dice delta = +0.003267

yes_cautious:
  mean IoU delta = +0.002233
  mean Dice delta = +0.002490
```

## Mask-Quality Limitation Narrative

Current conclusion:

```text
The 11 usable_for_overlap = no images should remain excluded from primary
overlap claims. These masks are not reliable lesion-reference regions.
```

Use:

```text
main_yes as primary
yes_cautious as sensitivity
excluded_no as limitation discussion
```

Do not:

```text
rank models using excluded_no
interpret high overlap values in excluded_no as reliable localization
```

## Candidate Main-Text Tables

Table 1: Cohort and mask usability

```text
Source: overlap_usable_scope_counts.csv
Rows: main_yes, sensitivity_yes_cautious, excluded_no, all_images
```

Table 2: Main overlap performance by model

```text
Source: overlap_usable_scope_model_metric_summary.csv
Scope: main_yes
Suggested metrics:
  normalized_energy
  peak_in_mask
  peak_distance_px with validity footnote
  iou_top20
  dice_top20
```

Table 3: Paired ConvNeXt comparison

```text
Source: paired_overlap_convnext_vs_others_summary.csv
Scopes: main_yes and sensitivity_yes_cautious
Suggested metrics:
  normalized_energy
  iou_top20
  dice_top20
  peak_distance_px with validity footnote
```

Table 4: Sensitivity overlap performance

```text
Source: overlap_usable_scope_model_metric_summary.csv
Scope: sensitivity_yes_cautious
Could be supplementary if main text space is limited.
```

Table 5: Exploratory subgroup analysis

```text
Source:
  overlap_by_true_label_summary.csv
  overlap_by_confusion_group_summary.csv
  overlap_subgroup_counts.csv

Suggested location:
  supplement

Suggested focus:
  true-label subgroup Dice/IoU
  TP vs FN and FP vs TN descriptive patterns
```

Table 6: Bright-dot / marker limitation

```text
Source:
  bright_dot_marker_classification_sensitivity_by_model.csv
  bright_dot_marker_exclusion_delta_by_model.csv

Suggested focus:
  marker-positive benign FP rate
  specificity
  overlap deltas after marker exclusion
```

Table 7: Classification threshold and paired AUC companion

```text
Source:
  classification_test_locked_threshold_performance.csv
  classification_test_auc_by_model.csv
  classification_test_auc_pairwise_bootstrap.csv

Suggested location:
  main text if classification performance is central
  supplement if the paper focuses primarily on localization/interpretability

Suggested focus:
  validation-selected locked thresholds
  test AUC with patient-level bootstrap CI
  paired AUC differences
```

## Candidate Figures

Figure 1: Study / analysis flow

```text
290 test images
183 main overlap images
96 cautious sensitivity-only images
11 excluded no-overlap images
25 marker-positive images
```

Figure 2: Main overlap metrics by model

```text
Source: overlap_usable_scope_model_metric_summary.csv
Scope: main_yes
Metrics: normalized_energy, iou_top20, dice_top20, peak_in_mask
```

Figure 3: Main vs sensitivity ranking stability

```text
Source: overlap_usable_scope_model_metric_ranks.csv
Compare main_yes vs sensitivity_yes_cautious.
```

Figure 4: Paired ConvNeXt deltas

```text
Source: paired_overlap_convnext_vs_others_long.csv
Candidate plot:
  paired-delta boxplots by comparator and metric
```

Figure 5: Exploratory subgroup plot

```text
Source:
  overlap_by_true_label_summary.csv
  overlap_by_confusion_group_summary.csv

Candidate plot:
  Dice or normalized_energy by true_label and confusion_group

Suggested location:
  supplement unless it becomes central to the Results narrative.
```

Figure 6: Bright-dot / marker limitation

```text
Source:
  bright_dot_marker_classification_sensitivity_by_model.csv
  bright_dot_marker_exclusion_delta_by_model.csv

Show ConvNeXt marker-positive benign FP signal and small overlap deltas.
```

Figure 7: Classification companion plot

```text
Source:
  classification_test_auc_by_model.csv
  classification_test_auc_pairwise_bootstrap.csv

Candidate plot:
  test AUC with 95% CI by model
  optional paired AUC delta plot

Suggested location:
  supplement unless classification performance becomes a main-text pillar.
```

## Files to Treat Carefully

Old diffuse-large QC folder:

```text
gradcam_mask_overlap/qc_figures/mask_qc_diffuse_large_top20/
```

Reason:

```text
Contains pre-correction 0019 LEFT panels.
Do not use for final interpretation.
```

Corrected diffuse-large QC folder:

```text
gradcam_mask_overlap/qc_figures/mask_qc_diffuse_large_top20_after_0019_correction/
```

All-image bright-dot analysis:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/
```

Use as:

```text
context / full-sample artifact analysis
```

Do not use as:

```text
the only final mask-usable marker sensitivity result
```

Preferred marker sensitivity folders:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/main_yes/
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/usable_yes_cautious/
```

## Methods Provenance

Current scripts:

```text
gradcam_mask_overlap/tools/build_test_image_master_table.py
gradcam_mask_overlap/tools/fill_usable_for_overlap_from_mask_qc.py
gradcam_mask_overlap/tools/analyze_master_overlap_by_usable_set.py
gradcam_mask_overlap/tools/analyze_bright_dot_marker_sensitivity.py
gradcam_mask_overlap/tools/analyze_paired_overlap_model_comparison.py
gradcam_mask_overlap/tools/analyze_overlap_subgroups.py
gradcam_mask_overlap/tools/analyze_classification_thresholds_and_auc.py
gradcam_mask_overlap/tools/build_paper_ready_classification_tables.py
```

Manual source:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

Final analysis source:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

## Open Items Before Manuscript Draft

```text
[ ] Decide which metrics go into the main-text table.
[ ] Decide which metrics are supplementary only.
[ ] Draft exact wording for peak_distance_px denominator footnote.
[ ] Decide whether paired comparison table goes in main text or supplement.
[ ] Decide whether subgroup analysis stays supplementary.
[ ] Decide whether classification threshold / paired AUC analysis goes in main text or supplement.
[ ] Draft exact wording for bright-dot / marker limitation.
[ ] Decide whether to include representative QC images in main text or supplement.
[ ] Confirm no manual_mask_review_template.csv edits are pending.
[ ] Regenerate master table and paper-ready summaries if manual labels change.
```
