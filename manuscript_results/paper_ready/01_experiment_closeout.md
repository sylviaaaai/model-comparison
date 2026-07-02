# Experiment Closeout: Grad-CAM Mask Overlap and Classification Companion

Date: 2026-07-02
Workspace: `/mnt/e/SW整合`

## Closeout Status

The experiment is ready to move from analysis into manuscript drafting.

Completed analysis lines:

```text
1. Master-table mask usability and primary overlap analysis
2. Main/sensitivity overlap comparison
3. Bright-dot / marker artifact sensitivity
4. Paired overlap model comparisons
5. Exploratory overlap subgroup analysis
6. Validation-selected classification threshold analysis
7. Patient-level paired test AUC comparison
8. Paper-readable compact tables
```

Current paper-facing index:

```text
gradcam_mask_overlap/paper_draft_materials/00_results_index.md
```

Use that file as the entry point before writing Results, Methods, tables, and
supplementary material.

## Source Of Truth

Primary overlap source:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Manual mask-review source:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

Important rule:

```text
If manual mask labels change, edit manual_mask_review_template.csv and then
regenerate test_image_master_table.csv before rerunning final analyses.
```

## Final Analysis Populations

```text
All final test images: 290

main_yes:
  usable_for_overlap == yes
  n = 183
  use as primary overlap denominator

sensitivity_yes_cautious:
  usable_for_overlap in {yes, cautious}
  n = 279
  use as mask-QC sensitivity denominator

excluded_no:
  usable_for_overlap == no
  n = 11
  use only as excluded/limitation cases
```

Classification analysis denominator:

```text
Final test set, n = 290 images, 135 patients.
```

Classification and overlap denominators are intentionally different. Do not
force the classification results onto the 183-image main overlap set unless a
new supplementary analysis is explicitly planned.

## Main Results To Carry Forward

Overlap / localization:

```text
ConvNeXt_Base is the strongest model for lesion-mask overlap/localization.
This is clearest for energy concentration, peak-in-mask, mask coverage, IoU,
and Dice.

ResNet50 is the closest overlap comparator but peak_distance_px must be
interpreted conservatively because ResNet50 has invalid zero-saliency rows.

ViT_B16 has weaker attention-rollout overlap despite strong classification
AUC.
```

Bright-dot / marker limitation:

```text
Bright-dot / marker artifacts are a real shortcut/artifact sensitivity issue,
especially for ConvNeXt_Base benign false positives.

They do not drive the average overlap ranking.
```

Classification:

```text
ViT_B16 has the highest numerical test AUC.
ConvNeXt_Base and Swin_Base are close behind.
ResNet50 is lower than the other three under paired patient-level bootstrap.

Paired AUC differences among ViT_B16, ConvNeXt_Base, and Swin_Base cross 0,
so their ordering should be stated conservatively.
```

Manuscript framing:

```text
Separate the classification claim from the localization claim:

  Classification AUC: ViT_B16 numerically highest.
  Localization / overlap: ConvNeXt_Base strongest.
```

## Paper-Readable Tables

Overlap and subgroup tables:

```text
gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_true_label_key_metrics.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_confusion_group_dice.csv
```

Classification readable tables:

```text
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_readable_tables.md
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_validation_thresholds_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_test_locked_thresholds_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_test_fixed_0p5_readable.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_classification_paired_auc_readable.csv
```

Primary overlap tables:

```text
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_counts.csv
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_summary.csv
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_model_metric_ranks.csv
gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_peak_distance_validity_by_scope.csv
```

Paired overlap tables:

```text
gradcam_mask_overlap/overlap_analysis/paired_model_comparisons/paired_overlap_convnext_vs_others_summary.csv
gradcam_mask_overlap/overlap_analysis/paired_model_comparisons/paired_overlap_all_model_pairs_summary.csv
gradcam_mask_overlap/overlap_analysis/paired_model_comparisons/paired_overlap_model_comparison_summary.md
```

## Final Logs

```text
gradcam_mask_overlap/experiment_logs/2026-07-01_master_table_overlap_and_marker_sensitivity_log.md
gradcam_mask_overlap/experiment_logs/2026-07-01_paired_overlap_model_comparison_log.md
gradcam_mask_overlap/experiment_logs/2026-07-02_overlap_subgroup_analysis_log.md
gradcam_mask_overlap/experiment_logs/2026-07-02_classification_threshold_auc_analysis_log.md
gradcam_mask_overlap/experiment_logs/overlap_analysis_progress_plan.md
```

## Do Not Use As Final Evidence

Old diffuse-large QC folder:

```text
gradcam_mask_overlap/qc_figures/mask_qc_diffuse_large_top20/
```

Reason:

```text
Contains pre-correction 0019 LEFT panels.
```

Do not use `excluded_no` for model ranking. These masks are not reliable
reference regions.

Do not describe ResNet50 as clearly best for `peak_distance_px`; use the
valid non-zero saliency-map denominator footnote.

## Remaining Manuscript Decisions

```text
[ ] Decide which overlap metrics are main-text table columns.
[ ] Decide whether paired overlap comparison is main text or supplement.
[ ] Decide whether classification companion analysis is main text or supplement.
[ ] Decide whether subgroup analysis stays supplementary.
[ ] Draft peak_distance_px denominator footnote.
[ ] Draft bright-dot / marker artifact limitation paragraph.
[ ] Choose representative QC/example figures.
```

## GitHub Status

Current blocker:

```text
gradcam_mask_overlap/ is not inside a valid local git repository.
The root .git directory at /mnt/e/SW整合/.git is empty.
GitHub CLI gh is not installed in this environment.
```

There is a separate valid repository:

```text
/mnt/e/SW整合/model-comparison-repo
origin: https://github.com/sylviaaaai/model-comparison.git
```

But it already has many unrelated modified/deleted/generated Grad-CAM files.
Do not stage that repository wholesale for this closeout.

Recommended GitHub options:

```text
Option A:
  Put gradcam_mask_overlap under its own GitHub repository and commit the
  analysis logs, scripts, metadata, and paper_draft_materials.

Option B:
  Intentionally copy selected final paper-facing outputs into
  model-comparison-repo under a new clean folder, then commit only that folder.

Option C:
  Install gh, initialize a valid repository for /mnt/e/SW整合 or
  gradcam_mask_overlap, add a remote, and push a dedicated closeout branch.
```

Do not update GitHub until the target repository and intended file scope are
confirmed.
