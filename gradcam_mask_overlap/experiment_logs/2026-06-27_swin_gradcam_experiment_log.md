# Experiment Log: Swin Raw Grad-CAM Export

Date: 2026-06-27
Workspace: `/mnt/e/SW整合`

## Goal

Continue the raw saliency-map export workflow from Swin, matching the ResNet50
and ConvNeXt format from 2026-06-26.

## Model Choice

The active four-model comparison metadata uses:

```text
Swin/results/swin_base_torch_final_stage2_4_dropout04_bs16/final/stage2_4/final_model.pt
```

This is the model referenced by:

```text
results/patient_level_bootstrap_ci/swin_base_test_predictions.csv
gradcam_mask_overlap/metadata/model_predictions.csv
```

Note: `Swin/results/swin_base_torch_final_stage2_4_dropout03_bs16/` exists and
was open in the IDE, but the current four-model comparison line uses
`dropout=0.4, bs16`.

## New Scripts

Added:

```text
gradcam_mask_overlap/tools/export_swin_raw_gradcam.py
gradcam_mask_overlap/tools/update_analysis_ready_from_saliency.py
```

`export_swin_raw_gradcam.py` exports the same two target modes as the previous
models:

- `gradcam` / `predicted_label`
- `gradcam_malignant` / `cancer_score`

Swin target layer:

```text
layers.3.blocks.1
```

The script handles timm Swin BHWC activations by converting them to BCHW before
the Grad-CAM channel weighting step.

## Smoke Test

CPU smoke tests were run on 2 images for both target modes using temporary
outputs under `/tmp`.

Results:

```text
predclass: 2 / 2 exported
malignant: 2 / 2 exported
heatmap shape: (224, 224)
heatmap dtype: float32
heatmap range: 0-1
all-zero heatmap: false
```

## Full Export

Full export was run outside the sandbox with GPU visible.

Both runs printed:

```text
Device: cuda
```

Outputs:

```text
gradcam_mask_overlap/data/saliency_maps/swin_base/gradcam_predclass/
gradcam_mask_overlap/data/saliency_maps/swin_base/gradcam_malignant/
```

Run summaries:

```text
predclass: processed_predictions=290, exported=290, failed=0
malignant: processed_predictions=290, exported=290, failed=0
```

Verification:

```text
gradcam_predclass: 290 npy, 290 png
gradcam_malignant: 290 npy, 290 png
manifest_swin_raw: 580
gradcam: 290
gradcam_malignant: 290
missing saliency paths: 0
prediction label mismatch: 0
probability max diff vs source predictions: 0.0025286
probability mean diff vs source predictions: 0.0002292
```

The probability drift is larger than ConvNeXt but did not change any predicted
labels. It is treated as normal re-evaluation variation for this export.

## Metadata Updates

Updated:

```text
gradcam_mask_overlap/metadata/saliency_manifest.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest_summary.json
gradcam_mask_overlap/metadata/runs.jsonl
```

Current summary:

```text
saliency_manifest rows: 1915
duplicate saliency_id: 0
analysis_ready_overlap_manifest rows: 2030
duplicate analysis_id: 0
ready_for_overlap_rows: 1740
ready missing paths: 0
```

Ready rows:

```text
resnet50:gradcam = 290
resnet50:gradcam_malignant = 290
convnext_base:gradcam = 290
convnext_base:gradcam_malignant = 290
swin_base:gradcam = 290
swin_base:gradcam_malignant = 290
```

Remaining pending model:

```text
ViT still needs raw saliency map export before the four-model overlap comparison.
```

## Recommended Next Steps

1. Export ViT raw saliency/attention maps in manifest-compatible format.
2. Run `update_analysis_ready_from_saliency.py --model-name vit_b16` after ViT
   saliency rows are available.
3. Start overlap metrics using the main analysis filter:

```text
saliency_method == "gradcam_malignant"
```

