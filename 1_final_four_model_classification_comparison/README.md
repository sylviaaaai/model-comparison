# 1. Final Four-Model Classification Comparison

## Purpose

This chapter contains the final classification comparison for the four mammography ROI models. The task is binary classification: benign vs cancer, with cancer treated as the positive class.

The chapter answers three questions:

1. How were the patient-level train/validation/test splits defined?
2. Which final model setting was selected for each architecture?
3. How do the four final models compare on the locked final test set, both at validation-selected operating thresholds and by threshold-free paired AUC?

This classification chapter is separate from the saliency-mask overlap chapter. The classification analysis evaluates prediction performance; the overlap chapter evaluates whether explanation maps spatially align with lesion masks.

## Recommended Reading Order

Read the folders and files in this order:

1. [data_splits/README.md](data_splits/README.md)
   - Start here to understand the patient-level split design and the locked 80/10/10 train/validation/test split.

2. [classification_model_comparison/README.md](classification_model_comparison/README.md)
   - Read this for the four final model settings, model-selection records, final test results, and resource comparison.

3. [best_final_models_comparison.md](classification_model_comparison/results/four_model_comparison_tables/best_final_models_comparison.md)
   - Main four-model classification comparison table.

4. [image_level_patient_bootstrap_ci_summary.md](classification_model_comparison/results/four_model_comparison_tables/image_level_patient_bootstrap_ci_summary.md)
   - Final test performance with patient-level bootstrap confidence intervals.

5. [validation_selected_threshold_and_paired_auc_analysis/README.md](validation_selected_threshold_and_paired_auc_analysis/README.md)
   - Read this for validation-selected thresholds, locked-threshold test performance, fixed-0.5 comparison, and paired AUC analysis.

6. [classification_threshold_auc_readable_tables.md](validation_selected_threshold_and_paired_auc_analysis/classification_threshold_auc_readable_tables.md)
   - Reader-ready summary combining threshold-dependent results and paired AUC comparison.

7. [paired_auc_analysis/README.md](validation_selected_threshold_and_paired_auc_analysis/paired_auc_analysis/README.md)
   - Use this when you need the model-to-model paired AUC interpretation.

## Chapter Structure

| Subfolder | What it does | How to use it |
| --- | --- | --- |
| [data_splits/](data_splits/) | Documents patient-level splitting, 5-fold model-selection folds, and final 80/10/10 train/validation/test split. | Use this to verify that patients do not leak across split components. |
| [classification_model_comparison/](classification_model_comparison/) | Contains the four final model folders, selected model settings, final test summaries, resource tables, and cross-model comparison tables. | Use this as the main final-model comparison section. |
| [validation_selected_threshold_and_paired_auc_analysis/](validation_selected_threshold_and_paired_auc_analysis/) | Selects operating thresholds on validation data only, applies them once to the locked test set, and compares test AUCs using paired patient-level bootstrap. | Use this for threshold-dependent operating point results and threshold-free paired AUC comparisons. |

## Data Split Design

The split is patient-level, so images from the same patient do not appear across train, validation, and test sets.

| Split component | Role |
| --- | --- |
| Training pool | Used for model training and 5-fold model-selection experiments. |
| Validation split | Used to select locked operating thresholds after final models are chosen. |
| Test split | Held out for final reporting only. |

The final fixed split follows an 80/10/10 design. The test set contains 290 images from 135 patients, with 147 malignant and 143 benign images.

## Final Models

The four final architectures are:

| Model | Final setting summary | Explanation method used later |
| --- | --- | --- |
| ResNet50 | conv3-5, dropout 0.4, batch size 16 | Grad-CAM |
| ViT-B/16 | all blocks, dropout 0.4, batch size 16 | Attention rollout |
| ConvNeXt-Base | stage 3-4, dropout 0.4, batch size 32 | Grad-CAM |
| Swin-Base | stage 2-4, dropout 0.4, batch size 16 | Grad-CAM |

The explanation methods listed here are used later in `../2_main_saliency_mask_overlap_analysis/`. The classification chapter itself focuses on predictive performance.

## Main Classification Result Snapshot

The final four-model comparison reports these test-set AUC point estimates:

| Model | Test AUC | Accuracy | F1 |
| --- | ---: | ---: | ---: |
| ResNet50 | 0.701 | 0.624 | 0.620 |
| ViT-B/16 | 0.819 | 0.776 | 0.755 |
| ConvNeXt-Base | 0.791 | 0.728 | 0.723 |
| Swin-Base | 0.781 | 0.762 | 0.723 |

Key reading:

- ViT-B/16 has the highest test AUC point estimate.
- ResNet50 has lower test AUC than the other three models.
- ConvNeXt-Base and Swin-Base are close to ViT-B/16, and differences among these three stronger models are smaller.
- Accuracy, sensitivity, specificity, precision, and F1 depend on the chosen threshold, so AUC should not be read alone.

## Threshold And Paired AUC Results

The threshold analysis selects one threshold per model using validation predictions only, then applies that threshold once to the final test set.

| Model | Validation-selected threshold | Locked test AUC | Locked test accuracy | Locked test F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | 0.607 | 0.701 | 0.655 | 0.615 |
| ViT-B/16 | 0.558 | 0.819 | 0.779 | 0.754 |
| ConvNeXt-Base | 0.693 | 0.791 | 0.759 | 0.727 |
| Swin-Base | 0.461 | 0.781 | 0.752 | 0.717 |

The paired AUC analysis compares models on the same final test patients using patient-level bootstrap resampling.

Main paired-AUC interpretation:

- ResNet50 is clearly lower than ViT-B/16, ConvNeXt-Base, and Swin-Base.
- ViT-B/16 is numerically highest among the three stronger models.
- Differences among ViT-B/16, ConvNeXt-Base, and Swin-Base have confidence intervals crossing zero, so they should be described as numerical differences rather than clearly separated performance tiers.

## Main Result Files

| Topic | File |
| --- | --- |
| Patient-level split overview | [data_splits/README.md](data_splits/README.md) |
| Four-model classification comparison | [best_final_models_comparison.md](classification_model_comparison/results/four_model_comparison_tables/best_final_models_comparison.md) |
| Patient-level bootstrap CI summary | [image_level_patient_bootstrap_ci_summary.md](classification_model_comparison/results/four_model_comparison_tables/image_level_patient_bootstrap_ci_summary.md) |
| Validation-selected threshold summary | [validation_selected_threshold/README.md](validation_selected_threshold_and_paired_auc_analysis/validation_selected_threshold/README.md) |
| Reader-ready threshold/AUC tables | [classification_threshold_auc_readable_tables.md](validation_selected_threshold_and_paired_auc_analysis/classification_threshold_auc_readable_tables.md) |
| Paired AUC comparison | [paired_auc_analysis/README.md](validation_selected_threshold_and_paired_auc_analysis/paired_auc_analysis/README.md) |
| Paired AUC reader-ready CSV | [tableS_classification_paired_auc_readable.csv](validation_selected_threshold_and_paired_auc_analysis/paired_auc_analysis/results/tableS_classification_paired_auc_readable.csv) |

## Relationship To Saliency-Mask Overlap Analysis

This chapter identifies and compares the final classification models. The saliency-mask overlap chapter then asks a different question: whether each model's saliency or attention maps align with lesion annotation masks.

Read next: [../2_main_saliency_mask_overlap_analysis/README.md](../2_main_saliency_mask_overlap_analysis/README.md).

Important distinction:

- The qualitative explainability panels inside each model folder often visualize the model's predicted class.
- The quantitative saliency-mask overlap analysis later uses a fixed overlap target where possible, especially malignant-class Grad-CAM for ResNet50, ConvNeXt-Base, and Swin-Base.

## Caveats

- This chapter reports image-level classification metrics with patient-level split design and patient-level bootstrap uncertainty.
- Threshold-dependent metrics depend on the selected operating threshold.
- The final test set is locked and should not be used for model or threshold selection.
- Large trained model checkpoint files are not included in the GitHub package.
- Raw mammography images and local workstation paths are not included.

## Conclusion

The classification analysis supports ViT-B/16 as the model with the highest test AUC point estimate, while ConvNeXt-Base and Swin-Base are close and ResNet50 is lower. The paired AUC analysis shows a clear gap between ResNet50 and the three stronger models, while differences among ViT-B/16, ConvNeXt-Base, and Swin-Base should be interpreted conservatively because their paired-bootstrap confidence intervals cross zero.
