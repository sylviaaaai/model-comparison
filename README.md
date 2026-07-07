# Mammography Four-Model Classification and Saliency-Mask Overlap Analysis

## Purpose

This repository is the cleaned final-analysis package for a binary mammography ROI study. The classification task is **Benign vs Cancer**, with **Cancer as the positive class**.

The project has two connected parts:

1. Compare four final deep-learning models on the same patient-level data split.
2. Evaluate whether each model's saliency or attention maps spatially align with annotated lesion ROI masks.

This repository is organized for manuscript and supervisor review, not as a chronological experiment log. Each major section has its own README with more detailed guidance.

## Models

| Model | Model family | Explanation method used for overlap analysis |
| --- | --- | --- |
| ResNet50 | Classical CNN | Grad-CAM |
| ViT-B/16 | Vision Transformer | Attention rollout |
| ConvNeXt-Base | Modern ConvNet | Grad-CAM |
| Swin-Base | Hierarchical Transformer | Grad-CAM |

## Recommended Reading Order

For a first pass, read these files in order:

1. [1_final_four_model_classification_comparison/README.md](1_final_four_model_classification_comparison/README.md)
   - Overview of patient-level splits, final model comparison, threshold selection, and paired AUC analysis.

2. [best_final_models_comparison.md](1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/best_final_models_comparison.md)
   - Main four-model classification comparison table.

3. [validation_selected_threshold_and_paired_auc_analysis/README.md](1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/README.md)
   - Validation-selected thresholds and paired test AUC comparison.

4. [2_main_saliency_mask_overlap_analysis/README.md](2_main_saliency_mask_overlap_analysis/README.md)
   - Overview of the saliency-mask overlap chapter and its reading order.

5. [main_vs_sensitivity_overlap_result/README.md](2_main_saliency_mask_overlap_analysis/main_vs_sensitivity_overlap_result/README.md)
   - Main saliency-mask overlap conclusion and mask-QC sensitivity result.

6. [main_vs_sensitivity_overlap_result/results/README.md](2_main_saliency_mask_overlap_analysis/main_vs_sensitivity_overlap_result/results/README.md)
   - Detailed guide for the main overlap CSV outputs and numeric summaries.

7. [paired_overlap_model_comparison/README.md](2_main_saliency_mask_overlap_analysis/paired_overlap_model_comparison/README.md)
   - Paired ConvNeXt-vs-other overlap comparison on the same images.

8. [bright_dot_marker_artifact_sensitivity/README.md](2_main_saliency_mask_overlap_analysis/bright_dot_marker_artifact_sensitivity/README.md)
   - Marker-artifact limitation and sensitivity analysis.

## Repository Structure

| Section | Folder | What it contains |
| --- | --- | --- |
| Classification analysis | [1_final_four_model_classification_comparison/](1_final_four_model_classification_comparison/) | Patient-level data splits, four-model training/selection summaries, validation-selected thresholds, locked test evaluation, and paired AUC analysis. |
| Saliency-mask overlap analysis | [2_main_saliency_mask_overlap_analysis/](2_main_saliency_mask_overlap_analysis/) | Saliency/mask alignment QC, primary overlap metrics, main-vs-sensitivity overlap results, paired overlap comparison, outside-context analysis, subgroup analysis, and marker-artifact sensitivity. |

## Key Findings

### Classification

The classification analysis uses patient-level data splitting. Model selection was performed inside the training pool, and final evaluation was conducted on a locked test set.

Main classification findings:

- ViT-B/16 achieved the highest test AUC point estimate in the validation-selected threshold analysis.
- ResNet50 showed lower test AUC than ViT-B/16, ConvNeXt-Base, and Swin-Base in paired patient-level bootstrap comparisons.
- Differences among ViT-B/16, ConvNeXt-Base, and Swin-Base were smaller, with confidence intervals overlapping or crossing zero in paired comparisons.
- The models showed different diagnostic trade-offs, so AUC should not be interpreted alone.

### Saliency-Mask Overlap

The saliency-mask analysis evaluates whether model explanation maps align with lesion annotation masks.

Main overlap findings:

- The strict primary overlap cohort is `main_yes`, containing 183 images with usable masks.
- The broader mask-QC sensitivity cohort is `sensitivity_yes_cautious`, containing 279 images.
- ConvNeXt-Base shows the strongest overall lesion-mask overlap across energy-based, peak-based, and top-20% binary overlap metrics.
- ResNet50 is the closest overlap comparator, but ResNet50 zero-saliency cases reduce valid counts for peak-distance metrics.
- Swin-Base is intermediate.
- ViT-B/16 attention rollout shows the weakest lesion-mask localization under these overlap metrics.
- The ordering remains stable after adding cautious masks.

Primary overlap denominators:

| Scope | Rule | N images | Role |
| --- | --- | ---: | --- |
| `main_yes` | `usable_for_overlap == yes` | 183 | Primary overlap result. |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | Mask-QC sensitivity result. |
| `excluded_no` | `usable_for_overlap == no` | 11 | Excluded/limitation cases only. |
| `all_images` | All final test images | 290 | Context only, not a primary overlap denominator. |

### Sensitivity And Limitations

- Mask quality is handled explicitly: strict usable masks define the primary overlap result, and cautious masks are added only in sensitivity analysis.
- ResNet50 zero-saliency cases are reported because they affect valid peak-distance counts.
- Marker-positive images are analyzed separately. They do not materially change the global overlap ranking, but they are an interpretability limitation because marker-positive benign cases show elevated false-positive behavior for some models.
- Saliency-mask overlap should be interpreted as spatial alignment evidence, not causal proof of which pixels caused a model prediction.

## Main Result Entry Points

| Topic | File |
| --- | --- |
| Four-model classification comparison | [best_final_models_comparison.md](1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/best_final_models_comparison.md) |
| Patient-level bootstrap CI | [image_level_patient_bootstrap_ci_summary.md](1_final_four_model_classification_comparison/classification_model_comparison/results/four_model_comparison_tables/image_level_patient_bootstrap_ci_summary.md) |
| Validation-selected threshold analysis | [validation_selected_threshold/README.md](1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/validation_selected_threshold/README.md) |
| Paired test AUC analysis | [paired_auc_analysis/README.md](1_final_four_model_classification_comparison/validation_selected_threshold_and_paired_auc_analysis/paired_auc_analysis/README.md) |
| Saliency-mask overlap chapter guide | [2_main_saliency_mask_overlap_analysis/README.md](2_main_saliency_mask_overlap_analysis/README.md) |
| Main overlap result | [main_vs_sensitivity_overlap_result/README.md](2_main_saliency_mask_overlap_analysis/main_vs_sensitivity_overlap_result/README.md) |
| Detailed main overlap CSV guide | [main_vs_sensitivity_overlap_result/results/README.md](2_main_saliency_mask_overlap_analysis/main_vs_sensitivity_overlap_result/results/README.md) |
| Paired overlap comparison | [paired_overlap_model_comparison/README.md](2_main_saliency_mask_overlap_analysis/paired_overlap_model_comparison/README.md) |
| Outside/perilesional saliency analysis | [outside_context_partition_saliency_energy_analysis/README.md](2_main_saliency_mask_overlap_analysis/outside_context_partition_saliency_energy_analysis/README.md) |
| Subgroup overlap analysis | [subgroup_analysis/README.md](2_main_saliency_mask_overlap_analysis/subgroup_analysis/README.md) |
| Marker-artifact sensitivity | [bright_dot_marker_artifact_sensitivity/README.md](2_main_saliency_mask_overlap_analysis/bright_dot_marker_artifact_sensitivity/README.md) |

## Data, Checkpoint, And Path Policy

This GitHub-facing package is designed for manuscript traceability and result review.

Not included:

- Raw mammography images
- Large trained model checkpoints
- Local absolute image, mask, saliency, and workstation paths
- Scratch files from non-final experiment runs

Included where appropriate:

- Cleaned result tables
- Patient-level split identifiers
- Training and analysis scripts
- Lightweight logs
- Selected qualitative saliency panels
- README files explaining how each result folder should be interpreted

Split tables use `image_id` and `patient_id` identifiers rather than local image paths.

## Reproducibility Notes

The scripts used for each analysis are stored beside the corresponding result folder under `tools/`.

This repository is not a single-command reproduction package. Model training, saliency export, manual mask review, overlap computation, and manuscript-table generation were performed as separate experiment stages. The repository preserves the cleaned outputs and scripts needed to audit each stage.

The Python package list used during analysis is summarized in `requirements.txt`. Environment and hardware notes are summarized in [environment_notes.md](environment_notes.md), including the RTX 5090 resource-timing context and the current status of package-version pinning. Some training scripts depend on framework-specific TensorFlow/Keras or PyTorch/timm environments.

## Repository Status

This is a cleaned final-analysis package for manuscript drafting and reviewer-facing documentation. It prioritizes patient-level evaluation traceability, readable result tables, separation between primary and sensitivity analyses, and clear documentation of limitations.
