# Mammography Four-Model Classification and Saliency-Mask Overlap Analysis

This repository contains the cleaned final-analysis package for a binary mammography ROI classification study and its follow-up quantitative explainability analysis. The classification task is **Benign vs Cancer**, with **Cancer treated as the positive class**.

The study compares four representative deep learning backbones under the same patient-level data split and then evaluates whether their saliency or attention maps spatially align with lesion annotation masks.

Rather than serving as a chronological experiment log, this repository is organized around the main manuscript analysis sections: model comparison, threshold selection, paired statistical analysis, saliency-mask overlap, mask-quality sensitivity analysis, contextual saliency analysis, subgroup analysis, and marker-artifact sensitivity checks.

---

## Project Overview

This project addresses two connected questions:

1. **Classification performance**
   How do four final mammography ROI classifiers compare under the same patient-level split and locked final test set?

2. **Explainability and localization**
   Do the explanation maps from these models align with lesion annotation masks, and are those localization patterns robust to mask quality, subgroup structure, paired comparisons, contextual saliency, and marker-like artifacts?

The four final model families and explanation methods are:

| Model | Model family | Explanation method used for overlap analysis |
| --- | --- | --- |
| ResNet50 | Classical CNN | Grad-CAM |
| ViT-B/16 | Vision Transformer | Attention rollout |
| ConvNeXt-Base | Modern ConvNet | Grad-CAM |
| Swin-Base | Hierarchical Transformer | Grad-CAM |

---

## Key Findings

### Classification performance

The final classification comparison uses a patient-level split. Model selection was performed inside the training pool, and final evaluation was conducted on a locked test set.

Main classification findings:

* **ViT-B/16 achieved the highest test AUC point estimate** in the validation-selected threshold analysis.
* **ResNet50 showed lower test AUC** than ViT-B/16, ConvNeXt-Base, and Swin-Base in paired patient-level bootstrap comparisons.
* Differences among **ViT-B/16, ConvNeXt-Base, and Swin-Base** were smaller, and their confidence intervals overlapped or crossed zero in paired comparisons.
* The models showed different diagnostic trade-offs, so performance should not be interpreted by AUC alone.

### Saliency-mask overlap

The quantitative explainability analysis evaluates whether each model's saliency or attention map aligns with lesion annotation masks.

Main overlap findings:

* The strict primary overlap cohort is **`main_yes`**, containing **183 images** with usable masks.
* The broader mask-QC sensitivity cohort is **`sensitivity_yes_cautious`**, containing **279 images**.
* **ConvNeXt-Base showed the strongest overall lesion-mask alignment** across energy-based, peak-based, and thresholded overlap metrics.
* **ResNet50 was the closest overlap comparator to ConvNeXt-Base**, although some ResNet50 maps had zero or invalid saliency values, reducing valid counts for peak-distance metrics.
* **Swin-Base showed intermediate overlap performance**.
* **ViT-B/16 showed weaker lesion-mask localization under the attention-rollout overlap metrics**, suggesting that its attention-derived maps may reflect broader spatial or contextual information rather than localized lesion-centered activation.
* The overlap ranking remained stable after including cautious masks, supporting the robustness of the main localization finding.

Primary overlap denominators:

| Scope | Rule | N images | Role |
| --- | --- | ---: | --- |
| `main_yes` | `usable_for_overlap == yes` | 183 | Primary overlap result |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | Mask-QC sensitivity result |
| `excluded_no` | `usable_for_overlap == no` | 11 | Excluded / limitation cases only |
| `all_images` | all final test images | 290 | Context only, not a primary overlap denominator |

### Mask quality and marker-artifact sensitivity

Mask quality and marker-like artifacts were analyzed separately from the primary overlap analysis.

Main sensitivity findings:

* Some excluded masks produced high overlap values despite being unsuitable for localization interpretation, suggesting that over-inclusive or non-lesion-specific masks may artificially inflate overlap metrics.
* Bright-dot or marker-like artifacts appeared in a limited subset of test images.
* Marker-positive benign cases showed elevated false-positive behavior for some models.
* Excluding marker-positive cases did not materially change the global overlap ranking, so marker artifacts are treated as a limitation and sensitivity analysis rather than causal proof of shortcut learning.

---

## Repository Structure

```text
model-comparison-repo/
├── README.md
├── requirements.txt
├── .gitignore
├── 1_final_four_model_classification_comparison/
│   ├── README.md
│   ├── data_splits/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── final_80_10_10/
│   │   └── folds/
│   ├── classification_model_comparison/
│   │   ├── README.md
│   │   ├── models/
│   │   │   ├── resnet50/
│   │   │   ├── vit_b16/
│   │   │   ├── convnext_base/
│   │   │   └── swin_base/
│   │   └── results/
│   │       └── four_model_comparison_tables/
│   └── validation_selected_threshold_and_paired_auc_analysis/
│       ├── README.md
│       ├── validation_selected_threshold/
│       ├── paired_auc_analysis/
│       └── tools/
└── 2_main_saliency_mask_overlap_analysis/
    ├── README.md
    ├── analysis_cohort_and_input_alignment/
    │   ├── README.md
    │   ├── results/
    │   ├── tools/
    │   └── qualitative_saliency_examples_across_mask_types_and_artifacts/
    ├── primary_overlap_metrics/
    │   ├── README.md
    │   ├── results/
    │   ├── example_qc_panels/
    │   └── tools/
    ├── main_vs_sensitivity_overlap_result/
    │   ├── README.md
    │   ├── results/
    │   └── tools/
    ├── paired_overlap_model_comparison/
    │   ├── README.md
    │   ├── results/
    │   └── tools/
    ├── outside_context_partition_saliency_energy_analysis/
    │   ├── README.md
    │   ├── results/
    │   └── tools/
    ├── subgroup_analysis/
    │   ├── README.md
    │   ├── results/
    │   └── tools/
    └── bright_dot_marker_artifact_sensitivity/
        ├── README.md
        ├── results/
        └── tools/
```

---

## How to Read This Repository

Start with the two main section README files:

| Entry point | What it explains |
| --- | --- |
| `1_final_four_model_classification_comparison/README.md` | Patient-level split strategy, four-model classification comparison, validation-selected thresholds, locked test evaluation, and paired AUC analysis. |
| `2_main_saliency_mask_overlap_analysis/README.md` | Saliency-map alignment, mask usability, primary overlap metrics, sensitivity analyses, contextual saliency, subgroup analysis, and artifact checks. |

For the shortest path through the main results, read the following files in order:

1. `1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/best_final_models_comparison.md`
2. `1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/image_level_patient_bootstrap_ci_summary.md`
3. `1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/validation_selected_threshold/README.md`
4. `1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/paired_auc_analysis/README.md`
5. `2_main_saliency_mask_overlap_analysis/main_vs_sensitivity_overlap_result/README.md`
6. `2_main_saliency_mask_overlap_analysis/paired_overlap_model_comparison/README.md`
7. `2_main_saliency_mask_overlap_analysis/outside_context_partition_saliency_energy_analysis/README.md`
8. `2_main_saliency_mask_overlap_analysis/bright_dot_marker_artifact_sensitivity/README.md`

---

## Analysis Workflow

The repository follows the manuscript analysis order:

| Step | Folder | Purpose |
| ---: | --- | --- |
| 1 | `data_splits/` | Define patient-level train, validation, test, and 5-fold model-selection splits. |
| 2 | `classification_model_comparison/` | Train, select, and compare the final ResNet50, ViT-B/16, ConvNeXt-Base, and Swin-Base models. |
| 3 | `validation_selected_threshold_and_paired_auc_analysis/` | Select operating thresholds on validation predictions only, apply them once to the locked test set, and compare test AUCs using paired patient-level bootstrap. |
| 4 | `analysis_cohort_and_input_alignment/` | Align final test predictions, saliency maps, image IDs, ROI masks, and manual mask-usability labels. |
| 5 | `primary_overlap_metrics/` | Compute continuous and thresholded saliency-mask overlap metrics. |
| 6 | `main_vs_sensitivity_overlap_result/` | Compare the strict usable-mask cohort with the broader yes+cautious sensitivity cohort. |
| 7 | `paired_overlap_model_comparison/` | Compare models image-by-image using paired overlap deltas and bootstrap confidence intervals. |
| 8 | `outside_context_partition_saliency_energy_analysis/` | Partition saliency energy into lesion, perilesional ring, and outside regions. |
| 9 | `subgroup_analysis/` | Analyze overlap patterns by true label and model-specific confusion group. |
| 10 | `bright_dot_marker_artifact_sensitivity/` | Evaluate whether bright-dot or marker-like artifacts influence classification or saliency behavior. |

---

## Main Result Files

| Topic | Folder | File |
| --- | --- | --- |
| Four-model final comparison | `1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/` | `best_final_models_comparison.md` |
| Patient-level bootstrap CI | `1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/` | `image_level_patient_bootstrap_ci_summary.md` |
| Validation-selected thresholds | `1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/validation_selected_threshold/results/` | `classification_validation_selected_thresholds.csv` |
| Locked test threshold performance | `1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/validation_selected_threshold/results/` | `classification_test_locked_threshold_performance.csv` |
| Paired test AUC comparison | `1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/paired_auc_analysis/results/` | `classification_test_auc_pairwise_bootstrap.csv` |
| Primary overlap metrics | `2_main_saliency_mask_overlap_analysis/primary_overlap_metrics/results/` | `overlap_selected_model_summary.csv` |
| Main vs sensitivity overlap | `2_main_saliency_mask_overlap_analysis/main_vs_sensitivity_overlap_result/results/` | `overlap_usable_scope_model_metric_summary.csv` |
| Paired overlap comparison | `2_main_saliency_mask_overlap_analysis/paired_overlap_model_comparison/results/` | `00_convnext_paired_key_ci_table.csv` |
| Context-partition saliency energy | `2_main_saliency_mask_overlap_analysis/outside_context_partition_saliency_energy_analysis/results/` | `00_context_partition_key_summary.csv` |
| True-label subgroup analysis | `2_main_saliency_mask_overlap_analysis/subgroup_analysis/results/` | `00_true_label_key_metrics_readable.csv` |
| Marker-artifact sensitivity | `2_main_saliency_mask_overlap_analysis/bright_dot_marker_artifact_sensitivity/results/` | `00_bright_dot_marker_key_summary.csv` |

---

## Data, Checkpoint, and Path Policy

This GitHub-facing package is designed for manuscript traceability and result review.

The following files are **not included**:

* Raw mammography images
* Large trained model checkpoints
* Local absolute image, mask, saliency, and workstation paths
* Scratch files from non-final experiment runs

The following files are included where appropriate:

* Cleaned result tables
* Patient-level split identifiers
* Training and analysis scripts
* Lightweight logs
* Selected qualitative saliency panels
* README files explaining how each result folder should be interpreted

Split tables use `image_id` and `patient_id` identifiers rather than local image paths.

---

## Reproducibility Notes

The scripts used for each analysis are stored beside the corresponding result folder under `tools/`.

This repository is **not** a single-command reproduction package. Model training, saliency export, manual mask review, overlap computation, and manuscript-table generation were performed as separate experiment stages. Instead, each section preserves the cleaned outputs and scripts needed to audit that stage.

The Python package list used during analysis is summarized in `requirements.txt`. Some training scripts depend on framework-specific environments for TensorFlow/Keras or PyTorch/timm models.

---

## Repository Status

This repository is a cleaned final-analysis package for manuscript drafting and reviewer-facing documentation.

It prioritizes:

* patient-level evaluation traceability,
* readable result tables,
* clear separation between primary and sensitivity analyses,
* cleaned GitHub-facing documentation,
* and reproducible analysis scripts where possible.

It does not include raw dataset files, full model checkpoints, or local workstation paths.
