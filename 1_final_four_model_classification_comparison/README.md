# 1. Final Four-Model Classification Comparison

This section contains the final classification comparison materials. The data split is patient-level, the model-selection phase uses 5-fold validation inside the training set, and the final fixed split uses 80% training, 10% validation, and 10% test images.

| Subfolder | Role |
| --- | --- |
| `data_splits/` | Patient-level split code and split CSVs cleaned for GitHub. Local image paths are replaced by image IDs. |
| `classification_model_comparison/` | Four final model folders and cross-model result tables. |
| `validation_selected_threshold_and_paired_auc_analysis/` | Validation-selected threshold analysis and paired patient-level bootstrap AUC comparison. |
