# Model Comparison

This repository contains the final four-model comparison package for the image classification experiments.

Only the final selected models, their training/selection process records, explainability figures, and summary result tables are included. Earlier discarded runs such as smoke tests and non-final batch-size trials are intentionally excluded.

## Final Models

| Model | Final setting | Validation AUC | Test AUC | Accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 | conv3_5, dropout=0.4, batch size=16 | 0.7337 +/- 0.0356 | 0.7016 | 0.6241 | 0.6202 |
| ViT-B/16 | all blocks, dropout=0.4, batch size=16 | 0.7803 +/- 0.0286 | 0.8190 | 0.7759 | 0.7547 |
| ConvNeXt-Base | stage3_4, dropout=0.4, batch size=32 | 0.7883 +/- 0.0309 | 0.7910 | 0.7276 | 0.7228 |
| Swin-Base | stage2_4, dropout=0.4, batch size=16 | 0.7846 +/- 0.0373 | 0.7813 | 0.7621 | 0.7229 |

The full comparison tables are in `summary/four_model_comparison_tables/`.

## Repository Structure

```text
model-comparison-repo/
├── data_splits/
│   ├── split_summary.json
│   ├── train_patients.csv
│   └── test_patients.csv
├── models/
│   ├── resnet50/
│   ├── vit_b16/
│   ├── convnext_base/
│   └── swin_base/
└── summary/
    └── four_model_comparison_tables/
```

Each model folder contains:

- `*.py`: final training/evaluation code for that architecture.
- `tools/`: plotting and explainability scripts.
- `selection_process/`: cross-validation logs, AUC summaries, and process plots used to select the final setting.
- `final_results/`: final test-set metrics, training log, resource table, and explainability outputs.

## Included Results

The summary folder contains:

- `best_final_models_comparison.csv/.md`
- `best_model_val_auc_ranges.csv/.md`
- `final_performance_comparison.csv/.md`
- `final_resource_comparison.csv/.md`
- `model_selection_auc_summary.csv/.md`

Explainability figures are included inside each model's `final_results/` directory:

- ResNet50: Grad-CAM
- ViT-B/16: attention maps and attention rollout
- ConvNeXt-Base: Grad-CAM
- Swin-Base: Grad-CAM

## Model Weights

Large trained model files are not included in this GitHub package because they are hundreds of MB each and exceed normal repository limits. The excluded file types include `.pt`, `.pth`, `.keras`, and `.h5`.

The included training logs and result tables preserve the final model performance and selection process.
