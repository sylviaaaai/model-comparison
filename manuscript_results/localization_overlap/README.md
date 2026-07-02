# Localization / Mask-Overlap Results

This folder collects the Grad-CAM / attention localization results for
manuscript writing.

## What This Covers

```text
1. Mask-usability denominators
2. Primary overlap analysis on usable_for_overlap == yes
3. Sensitivity analysis on usable_for_overlap in {yes, cautious}
4. Paired model comparisons
5. Peak-distance validity audit
6. Exploratory subgroup tables
```

## First Tables To Read

```text
tables/overlap_usable_scope_summary.md
tables/overlap_usable_scope_counts.csv
tables/overlap_usable_scope_model_metric_summary.csv
tables/paired_overlap_model_comparison_summary.md
```

## Main Interpretation

```text
ConvNeXt-Base has the strongest lesion-mask overlap/localization behavior.
The result is robust in the yes+cautious sensitivity set.
ResNet50 is the closest overlap comparator.
ViT-B/16 has strong classification AUC but weaker attention-rollout overlap.
```

## Table Guide

```text
overlap_usable_scope_counts.csv
  Denominators for main, sensitivity, excluded, and all-image scopes.

overlap_usable_scope_model_metric_summary.csv
  Model x metric summaries for each usable scope.

overlap_usable_scope_model_metric_ranks.csv
  Quick model ranking by metric and scope.

overlap_peak_distance_validity_by_scope.csv
  Reviewer-facing audit table for valid/invalid peak_distance_px cases.

paired_overlap_convnext_vs_others_summary.csv
  ConvNeXt-vs-comparator paired bootstrap summary.

paired_overlap_all_model_pairs_summary.csv
  Supplementary all-pair paired comparison table.

tableS_subgroup_true_label_key_metrics.csv
  Compact supplementary true-label subgroup table.

tableS_subgroup_confusion_group_dice.csv
  Compact supplementary TP/FN/FP/TN Dice table.
```

## Writing Notes

Use `main_yes` as the primary overlap denominator:

```text
usable_for_overlap == yes
n = 183
```

Use `sensitivity_yes_cautious` as the mask-quality sensitivity denominator:

```text
usable_for_overlap in {yes, cautious}
n = 279
```

Do not use `excluded_no` for model ranking.

Use conservative wording for `peak_distance_px` because ResNet50 has invalid
zero-saliency peak-distance cases.

