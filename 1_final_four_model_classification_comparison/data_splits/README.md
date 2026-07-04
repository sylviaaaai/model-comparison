# Data Splits

This folder documents the patient-level split strategy used by all four classification models.

## Strategy

The dataset is split at the patient level so images from the same patient do not appear across training, validation, and test sets. The final design is an 80-10-10 split:

| Split component | Role |
| --- | --- |
| 80% training pool | Used for 5-fold model selection. |
| 10% validation | Used to choose locked operating thresholds after model selection. |
| 10% test | Used only for final reporting. |

The 5-fold files under `folds/` subdivide the training pool for model-selection experiments. The `final_80_10_10/` folder stores the locked final train/validation/test split.

## GitHub Cleaning

Original split CSVs contained local image paths. In this repository those columns were replaced by `image_id` so the tables remain useful without exposing machine-specific paths.
