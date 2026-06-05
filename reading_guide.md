# Supervisor Reading Guide

## Recommended Reading Order

1. `README.md`

   Start here for the final four-model performance table and repository structure.

2. `summary/four_model_comparison_tables/best_final_models_comparison.md`

   Main final comparison across ResNet50, ViT-B/16, ConvNeXt-Base, and Swin-Base.

3. `data_splits/README.md`

   Explains the patient-level split strategy and how leakage was avoided.

4. `models/<model_name>/selection_process/`

   Cross-validation logs and plots used for model selection.

5. `models/<model_name>/final_results/`

   Final test-set results, training logs, resource summaries, and explainability figures.

## What Is Included

- Final training/evaluation code for each model.
- Patient-level data split code and actual split CSV files.
- Cross-validation selection records.
- Final result tables.
- Grad-CAM or attention-based explainability figures.

## What Is Not Included

- Large trained model weights (`.pt`, `.keras`, etc.).
- Raw image dataset files.
- Discarded smoke tests and non-final exploratory runs.

Large model weights were excluded because each final model is hundreds of MB and not suitable for a normal GitHub repository.
