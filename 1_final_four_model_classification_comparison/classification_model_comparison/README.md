# Classification Model Comparison

This folder contains the final four-model image-classification comparison package.
It includes the selected model artifacts, model-selection records, final test
results, resource summaries, and example explainability figures for the four
classification backbones.

Earlier discarded runs such as smoke tests and non-final batch-size trials are
intentionally excluded.

## Start Here

For the cross-model classification results, start with:

```text
results/four_model_comparison_tables/best_final_models_comparison.md
results/four_model_comparison_tables/image_level_patient_bootstrap_ci_summary.md
```

For the per-model experiment records, use:

```text
models/resnet50/
models/vit_b16/
models/convnext_base/
models/swin_base/
```

Related folders in the same section are:

```text
../data_splits/                                      train/validation/test split records
../validation_selected_threshold_and_paired_auc_analysis/  locked-threshold and paired AUC analysis
../../2_main_saliency_mask_overlap_analysis/        Grad-CAM / attention mask-overlap analysis
```

## Final Models

| Model | Final setting | Validation AUC | Test AUC | Test AUC 95% CI | Accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | conv3_5, dropout=0.4, batch size=16 | 0.7337 +/- 0.0356 | 0.7014 | [0.6239, 0.7769] | 0.6241 | 0.6202 |
| ViT-B/16 | all blocks, dropout=0.4, batch size=16 | 0.7803 +/- 0.0286 | 0.8190 | [0.7546, 0.8775] | 0.7759 | 0.7547 |
| ConvNeXt-Base | stage3_4, dropout=0.4, batch size=32 | 0.7883 +/- 0.0309 | 0.7910 | [0.7298, 0.8468] | 0.7276 | 0.7228 |
| Swin-Base | stage2_4, dropout=0.4, batch size=16 | 0.7846 +/- 0.0373 | 0.7813 | [0.7101, 0.8450] | 0.7621 | 0.7229 |



## Folder Structure

```text
classification_model_comparison/
├── README.md
├── models/
│   ├── resnet50/
│   ├── vit_b16/
│   ├── convnext_base/
│   └── swin_base/
└── results/
    └── four_model_comparison_tables/
        ├── best_final_models_comparison.csv
        ├── best_final_models_comparison.md
        ├── final_resource_comparison.csv
        ├── final_resource_comparison.md
        ├── image_level_patient_bootstrap_ci_summary.csv
        └── image_level_patient_bootstrap_ci_summary.md
```

Each model folder contains:

- `*.py`: final training/evaluation code for that architecture.
- `tools/`: plotting and explainability scripts.
- `selection_process/`: cross-validation logs, AUC summaries, and process plots
  used to select the final setting.
- `final_results/`: final test-set metrics, training log, resource table, and
  explainability outputs.

## Included Result Tables

The current comparison tables are in:

```text
results/four_model_comparison_tables/
```

This folder contains:

- `best_final_models_comparison.csv/.md`
- `final_resource_comparison.csv/.md`
- `image_level_patient_bootstrap_ci_summary.csv/.md`

`image_level_patient_bootstrap_ci_summary.csv/.md` reports final test
image-level performance with patient-level bootstrap 95% confidence intervals.

## Explainability Figures

Example explainability figures are included inside each model's
`final_results/` directory:

- ResNet50: `models/resnet50/final_results/gradcam/`
- ViT-B/16: `models/vit_b16/final_results/attention_rollout/`
- ConvNeXt-Base: `models/convnext_base/final_results/gradcam/`
- Swin-Base: `models/swin_base/final_results/gradcam/`

The displayed panels are sampled from the final test set by confusion group
(`correct_benign`, `correct_cancer`, `false_positive`, and `false_negative`).
These model-folder explainability panels are qualitative predicted-class
visualizations: for each image, the saliency target is the class predicted by
the model. Therefore, benign predictions visualize evidence supporting the
benign decision, while malignant predictions visualize evidence supporting the
malignant decision. This differs from the quantitative mask-overlap analysis,
which uses malignant-class saliency maps for ResNet50, ConvNeXt-Base, and
Swin-Base to keep the overlap target class fixed across images.

## Model Weights

Large trained model files are not included in this GitHub package because they
are hundreds of MB each and exceed normal repository limits. The excluded file
types include `.pt`, `.pth`, `.keras`, and `.h5`.

The included training logs, result tables, and explainability panels preserve
the final model performance and selection process.
