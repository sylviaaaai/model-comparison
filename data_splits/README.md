# Data Split Strategy

This folder contains the exact data split metadata and the scripts used to create it.

## Split Principle

The dataset was split at the patient level, not the image level. All images from the same patient stay in the same partition so that validation or test images cannot share a patient identity with training images.

The split is stratified by class label to keep the benign/cancer distribution balanced across partitions as much as possible.

## Split Workflow

1. Build image metadata from the ROI dataset.

   `code/Splits.py` scans the class folders, excludes mask/overlay files, assigns each image to a patient ID, and checks that each patient belongs to only one class.

2. Hold out an independent test pool.

   The first split is an 80/20 patient-level stratified split:

   - `train_patients.csv`
   - `train_images.csv`
   - `test_patients.csv`
   - `test_images.csv`

3. Create 5-fold cross-validation inside the training patients only.

   The held-out test pool is not used during model selection. The 5-fold files under `folds/` are generated only from training patients.

   - `train_cv_fold_assignments.csv` records which validation fold each training patient belongs to.
   - `folds/fold_*_train_patients.csv` and `folds/fold_*_val_patients.csv` store the patient-level fold splits.
   - `folds/fold_*_train_images.csv` and `folds/fold_*_val_images.csv` store the corresponding image-level fold splits.

4. Create the final 80/10/10 split.

   `code/create_final_80_10_10_split.py` keeps the original 80% training patients and splits the original 20% held-out test pool evenly by class into:

   - 10% final validation
   - 10% final test

   These files are stored in `final_80_10_10/`.

## Reproducibility

The split seed is fixed to `123` in the saved metadata and scripts.

The important leakage check is patient disjointness:

- train patients do not overlap with held-out patients
- cross-validation train patients do not overlap with validation patients inside each fold
- final train, final validation, and final test patients are mutually disjoint

## Files

- `code/Splits.py`: creates the initial patient-level train/test split and 5-fold CV files.
- `code/create_final_80_10_10_split.py`: converts the initial 80/20 split into the final 80/10/10 split.
- `split_summary.json`: summary for the initial 80/20 split and 5-fold CV.
- `final_80_10_10/summary.json`: summary for the final train/validation/test split.
