# Manuscript Results Package

This folder is the paper-facing results package for the four-model DDSM ROI
classification and Grad-CAM / attention localization study.

It consolidates the key tables that were previously spread across:

```text
summary/four_model_comparison_tables/
gradcam_mask_overlap/
models/*/final_results/
```

The original folders remain the source locations for full experiment records.
This folder is a curated reading and manuscript-writing entry point.

## Start Here

Recommended read order:

```text
1. classification/
2. localization_overlap/
3. artifact_sensitivity/
4. paper_ready/
```

The two most useful overview files are:

```text
paper_ready/00_results_index.md
paper_ready/01_experiment_closeout.md
```

## Folder Structure

```text
manuscript_results/
├── README.md
├── classification/
│   ├── README.md
│   └── tables/
├── localization_overlap/
│   ├── README.md
│   └── tables/
├── artifact_sensitivity/
│   ├── README.md
│   └── tables/
│       ├── all_images/
│       ├── main_yes/
│       └── usable_yes_cautious/
└── paper_ready/
    ├── README.md
    ├── 00_results_index.md
    └── 01_experiment_closeout.md
```

## Analysis Lines

### Classification Performance

Question:

```text
Which model predicts malignant vs benign labels best?
```

Main source folder:

```text
classification/
```

Includes:

```text
final model comparison
test AUC with patient-level bootstrap CI
validation-selected threshold analysis
locked-threshold test performance
paired test AUC comparison
resource comparison
```

Key interpretation:

```text
ViT-B/16 has the highest numerical final test AUC.
ConvNeXt-Base and Swin-Base are close behind.
ResNet50 is lower than the other three under paired patient-level bootstrap.

Paired AUC differences among ViT-B/16, ConvNeXt-Base, and Swin-Base cross 0,
so their ordering should be written conservatively.
```

### Localization / Mask Overlap

Question:

```text
When a model makes a prediction, does its Grad-CAM / attention signal align
with the manually reviewed lesion mask?
```

Main source folder:

```text
localization_overlap/
```

Includes:

```text
usable mask denominators
main overlap summaries
main vs sensitivity scope results
paired overlap model comparisons
peak-distance validity table
supplementary subgroup tables
```

Key interpretation:

```text
ConvNeXt-Base shows the strongest lesion-mask overlap/localization behavior.
ResNet50 is the closest overlap comparator.
ViT-B/16 has strong classification performance but weaker attention-rollout
overlap.
```

### Artifact / Bright-Dot Sensitivity

Question:

```text
Do bright-dot / marker artifacts act as shortcut signals or affect the overlap
conclusion?
```

Main source folder:

```text
artifact_sensitivity/
```

Includes:

```text
all_images/
main_yes/
usable_yes_cautious/
```

Key interpretation:

```text
Bright-dot / marker artifacts are a real limitation, especially for ConvNeXt
benign false positives, but they do not drive the average overlap ranking.
```

## How To Use This In A Manuscript

Recommended Results organization:

```text
1. Classification performance
   - final model AUC and patient-level bootstrap CI
   - validation-selected operating thresholds as an operating-point companion

2. Localization performance
   - main overlap denominator: usable_for_overlap == yes
   - sensitivity denominator: usable_for_overlap in {yes, cautious}
   - paired overlap comparisons

3. Classification-localization relationship
   - classification strength and lesion-mask localization are related but not
     identical
   - ViT-B/16 is numerically strongest for AUC, while ConvNeXt-Base is
     strongest for localization

4. Limitations and sensitivity analyses
   - peak-distance validity for ResNet50 zero-saliency cases
   - bright-dot / marker artifact sensitivity
   - exploratory subgroup analysis
```

## Denominators

Classification:

```text
Final test set: 290 images, 135 patients.
```

Primary overlap:

```text
main_yes:
  usable_for_overlap == yes
  n = 183 images
```

Sensitivity overlap:

```text
sensitivity_yes_cautious:
  usable_for_overlap in {yes, cautious}
  n = 279 images
```

Excluded overlap limitation:

```text
excluded_no:
  usable_for_overlap == no
  n = 11 images
```

Do not force the classification and overlap analyses to share the same
denominator unless a new supplementary analysis is explicitly planned.

## Important Writing Rules

```text
Do not present classification AUC and localization overlap as one combined
score.

Do not claim ResNet50 is clearly best for peak_distance_px because ResNet50
has invalid zero-saliency peak-distance cases.

Do not use excluded_no images for model ranking.

Do not use old diffuse_large QC figures from the pre-0019-correction folder.
```

## Original Source Locations

Classification source:

```text
summary/four_model_comparison_tables/
gradcam_mask_overlap/classification_analysis/threshold_auc/
```

Overlap source:

```text
gradcam_mask_overlap/overlap_analysis/
gradcam_mask_overlap/metadata/test_image_master_table.csv
```

Artifact source:

```text
gradcam_mask_overlap/limitation_analysis/bright_dot_marker/
```

Paper-planning source:

```text
gradcam_mask_overlap/paper_draft_materials/
```

