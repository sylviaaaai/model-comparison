# Experiment Log: Overlap Subgroup Analysis

Date: 2026-07-02
Workspace: `/mnt/e/SW整合`

## Purpose

This exploratory analysis summarizes overlap/localization metrics by:

```text
1. true_label
2. model-specific confusion group
3. true_label x model-specific confusion group
```

This analysis explains where the main overlap signal is coming from. It does
not replace the primary model-level analysis or paired model-comparison
analysis.

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

Excluded set:

```text
usable_for_overlap == no
```

was not included because these masks are not reliable overlap reference regions.

## Script and Command

Script:

```text
gradcam_mask_overlap/tools/analyze_overlap_subgroups.py
```

Command:

```text
python3 gradcam_mask_overlap/tools/analyze_overlap_subgroups.py
```

## Outputs

Output folder:

```text
gradcam_mask_overlap/overlap_analysis/subgroup_analyses/
```

Generated files:

```text
overlap_by_true_label_summary.csv
overlap_by_confusion_group_summary.csv
overlap_by_true_label_and_confusion_group_summary.csv
overlap_subgroup_counts.csv
overlap_subgroup_analysis_summary.md
overlap_subgroup_analysis_summary.json
```

Paper-ready compact tables were also generated from the full subgroup tables:

```text
gradcam_mask_overlap/tools/build_paper_ready_subgroup_tables.py

gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_true_label_key_metrics.csv
gradcam_mask_overlap/paper_draft_materials/tables/tableS_subgroup_confusion_group_dice.csv
gradcam_mask_overlap/paper_draft_materials/tables/paper_ready_subgroup_tables_summary.json
```

These compact tables are intended for human reading and supplementary-table
drafting. The full subgroup CSVs remain the audit tables.

## How to Read the Tables

### `overlap_by_true_label_summary.csv`

One row per:

```text
scope x true_label x model x metric
```

Use this table to ask:

```text
Do overlap metrics differ between malignant and benign images?
Is the model ranking similar within each true-label stratum?
```

Important columns:

```text
scope
true_label
model
metric
n_images
n_valid
mean
median
sd
marker_positive
zero_saliency_any_model
```

### `overlap_by_confusion_group_summary.csv`

One row per:

```text
scope x model-specific confusion_group x model x metric
```

Important:

```text
The confusion group is model-specific.
For example, ResNet50 TP/FN/FP/TN are not the same rows as ConvNeXt_Base
TP/FN/FP/TN.
```

Use this table to ask:

```text
Are TP cases more localized than FN cases?
Do FP cases show high overlap with benign masks?
Are TN/FN groups affected by zero-saliency maps?
```

### `overlap_by_true_label_and_confusion_group_summary.csv`

One row per:

```text
scope x true_label x model-specific confusion_group x model x metric
```

This table is mostly an audit / supplement table. It makes the structural
relationship explicit:

```text
TP/FN -> malignant rows
FP/TN -> benign rows
```

### `overlap_subgroup_counts.csv`

This is the denominator table. Use it before interpreting subgroup means.

Important columns:

```text
stratifier
scope
model
true_label
confusion_group
n_images
n_malignant
n_benign
TP / FN / FP / TN
marker_positive
zero_saliency_any_model
```

## Key Denominators

Main set:

```text
main_yes:
  malignant = 107
  benign = 76
```

Sensitivity set:

```text
sensitivity_yes_cautious:
  malignant = 141
  benign = 138
```

Model-specific main-set confusion groups:

| Model | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| `ResNet50` | 70 | 37 | 32 | 44 |
| `ViT_B16` | 76 | 31 | 11 | 65 |
| `ConvNeXt_Base` | 74 | 33 | 18 | 58 |
| `Swin_Base` | 65 | 42 | 6 | 70 |

Model-specific yes+cautious confusion groups:

| Model | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| `ResNet50` | 83 | 58 | 50 | 88 |
| `ViT_B16` | 95 | 46 | 18 | 120 |
| `ConvNeXt_Base` | 98 | 43 | 35 | 103 |
| `Swin_Base` | 85 | 56 | 10 | 128 |

## True-Label Findings

Mean top-20% Dice by true label:

| Scope | True label | ResNet50 | ViT_B16 | ConvNeXt_Base | Swin_Base |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `malignant` | 0.427275 | 0.122827 | 0.444076 | 0.358865 |
| `main_yes` | `benign` | 0.366122 | 0.106356 | 0.388726 | 0.192647 |
| `sensitivity_yes_cautious` | `malignant` | 0.410223 | 0.113650 | 0.437833 | 0.355718 |
| `sensitivity_yes_cautious` | `benign` | 0.342983 | 0.100072 | 0.398832 | 0.198072 |

Interpretation:

```text
ConvNeXt_Base remains the highest mean Dice model in both malignant and benign
subgroups.

Swin_Base shows a larger malignant-vs-benign separation than ConvNeXt_Base or
ResNet50.

ViT_B16 remains low in both true-label strata.
```

## Confusion-Group Findings

Mean top-20% Dice by model-specific confusion group:

| Scope | Model | TP | FN | FP | TN |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `ResNet50` | 0.497607 | 0.294213 | 0.475056 | 0.286897 |
| `main_yes` | `ViT_B16` | 0.137697 | 0.086369 | 0.126475 | 0.102951 |
| `main_yes` | `ConvNeXt_Base` | 0.484293 | 0.353893 | 0.455702 | 0.367940 |
| `main_yes` | `Swin_Base` | 0.464191 | 0.195862 | 0.422254 | 0.172966 |
| `sensitivity_yes_cautious` | `ResNet50` | 0.485913 | 0.301908 | 0.483169 | 0.263332 |
| `sensitivity_yes_cautious` | `ViT_B16` | 0.128551 | 0.082876 | 0.129310 | 0.095687 |
| `sensitivity_yes_cautious` | `ConvNeXt_Base` | 0.472061 | 0.359822 | 0.457968 | 0.378737 |
| `sensitivity_yes_cautious` | `Swin_Base` | 0.454501 | 0.205778 | 0.436227 | 0.179466 |

Interpretation:

```text
TP cases generally show higher overlap than FN cases, especially for ResNet50
and Swin_Base. This supports the interpretation that successful malignant
classification is associated with stronger lesion localization.

FP cases also show high overlap for several models. This means false-positive
malignant predictions often focus on the benign annotated region rather than
being purely diffuse or random.

TN cases tend to have lower overlap, and ResNet50 TN/FN groups include
zero-saliency cases. This reinforces the need to interpret ResNet50 peak
distance and overlap in TN/FN groups carefully.
```

## Reporting Guidance

Use this analysis as:

```text
exploratory subgroup analysis
supplementary table material
mechanistic explanation for TP/FN/FP behavior
```

Recommended manuscript placement:

```text
Supplementary Results / Supplementary Tables
```

Suggested main-text use:

```text
Mention only briefly in the Results or Discussion as a descriptive exploratory
finding. Do not present this as the primary model-comparison result.
```

Do not use it as:

```text
the primary model-ranking result
a replacement for main_yes / yes_cautious paired comparison
a formal hypothesis-testing table for small FP/FN subgroups
```

## Supplementary Conclusion

Status:

```text
Supplementary exploratory analysis
```

Conclusion:

```text
The subgroup analysis supports, but does not replace, the primary overlap
findings. ConvNeXt_Base retained high lesion-mask overlap in both malignant
and benign strata, indicating that its overall localization performance is not
driven by only one true-label subgroup.

Across models, TP malignant cases generally showed higher overlap than FN
malignant cases, suggesting that successful malignant classification tends to
coincide with stronger lesion localization.

FP benign cases also showed substantial overlap for several models, especially
ResNet50, ConvNeXt_Base, and Swin_Base. This suggests that false-positive
malignant predictions often localize to annotated benign findings rather than
being purely diffuse or random. This result helps contextualize false-positive
behavior and the bright-dot / marker artifact limitation.
```

Supplementary interpretation:

```text
These findings are descriptive and subgroup denominators are smaller than the
primary analysis populations. They should be reported as supplementary
mechanistic evidence, not as a formal subgroup hypothesis test.
```

Recommended paper wording:

```text
Exploratory subgroup analyses stratified by true label and model-specific
confusion group showed that ConvNeXt_Base retained high overlap in both
malignant and benign strata. TP cases generally had higher lesion-mask overlap
than FN cases, while FP cases often showed substantial overlap with benign
annotations, suggesting that false-positive malignant predictions may still be
localized to annotated benign findings.
```

Important caution:

```text
Subgroup denominators, especially FP/FN counts, are smaller than the primary
analysis populations. These results should be interpreted descriptively.
```

## Verification

Syntax check:

```text
python3 -m py_compile gradcam_mask_overlap/tools/analyze_overlap_subgroups.py
```

Result:

```text
passed
```
