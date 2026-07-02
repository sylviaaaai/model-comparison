# Experiment Log: Paired Overlap Model Comparison

Date: 2026-07-01
Workspace: `/mnt/e/SW整合`

## Purpose

The previous master-table analysis ranked models by descriptive means. This
run adds paired model comparisons on the same image sets, so model differences
can be interpreted as within-image deltas rather than independent group means.

Main question:

```text
Is ConvNeXt_Base's higher overlap/localization score stable when compared
image-by-image against the other models?
```

## Input

Source table:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Scopes:

```text
main_yes:
  usable_for_overlap == yes
  n = 183

sensitivity_yes_cautious:
  usable_for_overlap in {yes, cautious}
  n = 279
```

Reference model:

```text
ConvNeXt_Base
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

## Script and Command

Script:

```text
gradcam_mask_overlap/tools/analyze_paired_overlap_model_comparison.py
```

Command:

```text
python3 gradcam_mask_overlap/tools/analyze_paired_overlap_model_comparison.py
```

Bootstrap settings:

```text
bootstrap_iterations = 5000
seed = 20260701
```

## Outputs

Output folder:

```text
gradcam_mask_overlap/overlap_analysis/paired_model_comparisons/
```

Generated files:

```text
paired_overlap_convnext_vs_others_summary.csv
paired_overlap_all_model_pairs_summary.csv
paired_overlap_convnext_vs_others_long.csv
paired_overlap_model_comparison_summary.md
paired_overlap_model_comparison_summary.json
```

## How to Read the Tables

### `paired_overlap_convnext_vs_others_summary.csv`

This is the main paper-facing paired comparison table.

One row per:

```text
scope x metric x comparator_model
```

Reference model is always:

```text
ConvNeXt_Base
```

Important columns:

```text
scope
metric
direction
model_a = ConvNeXt_Base
model_b = comparator
n_scope_images
n_pairs
n_invalid_pairs
model_a_mean
model_b_mean
raw_mean_delta_model_a_minus_model_b
effect_mean_delta_positive_favors_model_a
effect_bootstrap_ci_low
effect_bootstrap_ci_high
model_a_better_rate
pair_validity_note
```

Effect-delta convention:

```text
For higher-is-better metrics:
  effect_delta = ConvNeXt_Base - comparator

For peak_distance_px:
  effect_delta = comparator - ConvNeXt_Base

Therefore:
  positive effect_delta always favors ConvNeXt_Base
```

Use this table for:

```text
Results narrative
paired mean deltas
bootstrap 95% confidence intervals
ConvNeXt-vs-each-model consistency checks
```

### `paired_overlap_all_model_pairs_summary.csv`

This is the supplementary / audit table.

One row per:

```text
scope x metric x unordered model pair
```

It includes all model pairs:

```text
ResNet50 vs ViT_B16
ResNet50 vs ConvNeXt_Base
ResNet50 vs Swin_Base
ViT_B16 vs ConvNeXt_Base
ViT_B16 vs Swin_Base
ConvNeXt_Base vs Swin_Base
```

Use this table for:

```text
supplementary all-pair comparison
checking non-ConvNeXt comparisons
audit trail
```

Important note:

```text
In this table, positive effect_delta favors model_a, not necessarily
ConvNeXt_Base.
```

### `paired_overlap_convnext_vs_others_long.csv`

This is the per-image paired-delta table.

One row per:

```text
scope x image_id x metric x comparator_model
```

Important columns:

```text
image_id
metric
reference_model
comparator_model
reference_value
comparator_value
raw_delta_reference_minus_comparator
effect_delta_positive_favors_reference
valid_pair
```

Use this table for:

```text
paired scatter plots
paired-delta boxplots
case-level inspection
checking which images drive a model difference
```

### `paired_overlap_model_comparison_summary.md`

This is the human-readable summary of ConvNeXt-vs-other comparisons.

Use this table for:

```text
quick reading
drafting Results text
checking CI direction
```

### `paired_overlap_model_comparison_summary.json`

This records reproducibility metadata:

```text
source master table
scope sizes
reference model
bootstrap iterations
seed
output file paths
```

## Key Results: main_yes

Scope:

```text
usable_for_overlap == yes
n = 183
```

ConvNeXt_Base vs ResNet50:

| Metric | N pairs | Mean effect delta | 95% bootstrap CI | Interpretation |
| --- | ---: | ---: | --- | --- |
| `energy_inside_mask` | 183 | 0.0828325 | 0.0399988 to 0.125735 | ConvNeXt higher. |
| `normalized_energy` | 183 | 0.144722 | 0.068457 to 0.2233 | ConvNeXt higher. |
| `peak_in_mask` | 183 | 0.0491803 | -0.0327869 to 0.125683 | Direction favors ConvNeXt, CI crosses 0. |
| `peak_distance_px` | 167 | 0.126152 | -1.53271 to 1.78387 | Very similar; valid-pair denominator only. |
| `mask_coverage_top20` | 183 | 0.0125526 | -0.00308359 to 0.029037 | Small difference; CI crosses 0. |
| `iou_top20` | 183 | 0.00931918 | -0.0073673 to 0.0262583 | Small difference; CI crosses 0. |
| `dice_top20` | 183 | 0.0192116 | -0.00345712 to 0.0428307 | Small difference; CI crosses 0. |

Interpretation:

```text
In the strict main set, ConvNeXt_Base is clearly higher than ResNet50 for
energy-based metrics. For top-20% IoU/Dice and peak metrics, ConvNeXt and
ResNet50 are close, and confidence intervals cross 0.
```

ConvNeXt_Base vs Swin_Base:

```text
ConvNeXt_Base has positive paired mean deltas for all main metrics.
The bootstrap CI is positive for energy, normalized energy, peak_in_mask,
top-20% coverage, IoU, and Dice.
```

ConvNeXt_Base vs ViT_B16:

```text
ConvNeXt_Base has large positive paired deltas across all main metrics.
This is the most stable pairwise separation.
```

## Key Results: sensitivity_yes_cautious

Scope:

```text
usable_for_overlap in {yes, cautious}
n = 279
```

ConvNeXt_Base vs ResNet50:

| Metric | N pairs | Mean effect delta | 95% bootstrap CI | Interpretation |
| --- | ---: | ---: | --- | --- |
| `energy_inside_mask` | 279 | 0.107427 | 0.0726008 to 0.14349 | ConvNeXt higher. |
| `normalized_energy` | 279 | 0.198885 | 0.129077 to 0.268386 | ConvNeXt higher. |
| `peak_in_mask` | 279 | 0.0896057 | 0.0250896 to 0.154122 | ConvNeXt higher. |
| `peak_distance_px` | 250 | 0.827168 | -0.685047 to 2.33957 | Direction favors ConvNeXt, CI crosses 0. |
| `mask_coverage_top20` | 279 | 0.0287442 | 0.0147894 to 0.0432344 | ConvNeXt higher. |
| `iou_top20` | 279 | 0.0257954 | 0.0115716 to 0.0403075 | ConvNeXt higher. |
| `dice_top20` | 279 | 0.0415775 | 0.021553 to 0.061808 | ConvNeXt higher. |

Interpretation:

```text
In the broader yes+cautious set, ConvNeXt_Base is paired-higher than ResNet50
for all non-distance metrics, with positive bootstrap CIs. peak_distance_px
still requires conservative interpretation because it uses only valid
non-zero-saliency pairs and its CI crosses 0.
```

ConvNeXt_Base vs Swin_Base:

```text
ConvNeXt_Base has positive paired mean deltas for all sensitivity-set metrics.
The CI is positive for all listed metrics, including top-20% IoU/Dice.
```

ConvNeXt_Base vs ViT_B16:

```text
ConvNeXt_Base remains strongly higher than ViT_B16 across all paired metrics.
```

## Statistical / Reporting Interpretation

Use cautious wording:

```text
ConvNeXt_Base had the highest model-level mean overlap metrics and showed
positive paired differences against Swin_Base and ViT_B16 across the main and
sensitivity analyses.

Compared with ResNet50, ConvNeXt_Base showed clear paired advantages for
energy-based metrics in the strict main set, while top-20% IoU/Dice differences
were small and had bootstrap CIs crossing 0. In the broader yes+cautious
sensitivity set, ConvNeXt_Base showed positive paired differences over ResNet50
for top-20% IoU/Dice as well.
```

For peak distance:

```text
Do not claim a clear ConvNeXt-vs-ResNet50 peak_distance_px advantage.
Both models have small distances among valid pairs, and ResNet50 has invalid
peak-distance rows due to zero-saliency maps.
```

## Verification

Syntax check:

```text
python3 -m py_compile gradcam_mask_overlap/tools/analyze_paired_overlap_model_comparison.py
```

Result:

```text
passed
```
