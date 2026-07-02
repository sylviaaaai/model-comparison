# Grad-CAM Mask Overlap Experiment

This folder stores metadata and derived files for measuring overlap between
model saliency maps and DDSM lesion masks in ROI coordinates.

## GitHub Package Note

This analysis is stored in the same repository as the four-model
classification comparison, but the results are intentionally separated into
two analysis lines:

- `classification_analysis/`: validation-selected operating thresholds and
  final test-set classification/AUC comparisons.
- `overlap_analysis/`: Grad-CAM / attention overlap with manually reviewed
  lesion masks.

These answer different questions. Classification asks whether the model
predicts the correct label. Overlap asks whether the model's saliency or
attention is spatially aligned with the lesion mask.

The paper-facing entry point is:

```text
paper_draft_materials/00_results_index.md
```

The closeout summary is:

```text
paper_draft_materials/01_experiment_closeout.md
```

## Core Join Key

All tables use `image_id` as the stable image-level key.

Example:

```text
0217_C_0217_1.RIGHT_CC
```

## Tables

- `metadata/test_images_manifest.csv`
  - One row per final test ROI image.
  - Links the model-input ROI image to the original mammogram image.

- `metadata/test_roi_mask_manifest.csv`
  - One row per test ROI image/mask pair.
  - The ROI mask files have been generated for the full 290-image final test
    set.
  - Crop coordinates, mask areas, image/mask dimensions, and QC status are
    filled during mask generation.

- `metadata/model_predictions.csv`
  - One row per model/image prediction from the full patient-level bootstrap CI
    prediction files.
  - Current coverage is the full final test set: 4 models x 290 images.

- `metadata/saliency_manifest.csv`
  - One row per model/image saliency map entry.
  - Current coverage is based on existing Grad-CAM/attention summary CSVs, so it
    is still a subset until raw maps are fully generated.
  - Raw saliency map paths are currently pending export.

- `metadata/overlap_metrics.csv`
  - Empty schema for overlap results.
  - One row should represent one saliency map, one ROI mask, and one threshold
    setting.

## Generated Mask Layer

Generated ROI-space masks:

```text
data/test_roi_masks/benign/
data/test_roi_masks/cancer/
```

QC overlays:

```text
data/qc_overlays/benign/
data/qc_overlays/cancer/
```

Current status:

```text
290 / 290 test ROI masks generated
143 benign masks
147 cancer masks
91 image/mask pairs had 1-pixel-level source size differences that were
recorded as size_mismatch_tolerated in qc_note
```

## Primary Mask Rule

The primary analysis should reproduce the original ROI generation logic:

```text
_Mask.jpg -> threshold at 127 -> boundingRect -> square crop with padding=0.1
-> resize image to 224 with linear interpolation
-> resize mask to 224 with nearest-neighbor interpolation
```

Bounding boxes are used only to reproduce ROI cropping. Overlap metrics should
be computed against the filled binary lesion mask in ROI coordinates, not
against the bounding box.
