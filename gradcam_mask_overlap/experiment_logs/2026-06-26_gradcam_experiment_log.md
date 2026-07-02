# Experiment Log: Grad-CAM Raw Map Export and QC

Date: 2026-06-26  
Workspace: `/mnt/e/SW整合`

## Goal

Prepare model Grad-CAM outputs for later mask-overlap / mIoU analysis, while keeping separate versions for qualitative visualization and quantitative overlap.

## Main Decisions

- Use two Grad-CAM targets per model:
  - `gradcam` / `predicted_label`: explains the model's actual predicted class; use for TP/TN/FP/FN visual examples.
  - `gradcam_malignant` / `cancer_score`: always explains the Cancer score; use for the main overlap / mIoU experiment.
- Store raw quantitative maps as 224x224 `.npy` arrays plus grayscale PNG previews.
- Keep color overlay panels only for presentation; do not use color-mapped images for overlap metrics.

## ResNet50 Status

Final model:

```text
results/resnet50_final_conv3_5_preprocess_dropout04/final/conv3_5/final_model.keras
```

Grad-CAM layer:

```text
resnet50/conv5_block3_out
```

Important fix:

- The final ResNet50 model was trained with `keras.applications.resnet50.preprocess_input`.
- The saved Keras graph did not expose that preprocessing op as a normal layer when rebuilding the Grad-CAM path.
- The raw export script explicitly reapplies `preprocess_input` inside the Grad-CAM model path.

Outputs:

```text
gradcam_mask_overlap/data/saliency_maps/resnet50/gradcam_predclass/
gradcam_mask_overlap/data/saliency_maps/resnet50/gradcam_malignant/
```

Verification:

```text
gradcam: 290 raw maps
gradcam_malignant: 290 raw maps
paths exist: 580 / 580
probability max diff vs source predictions: ~4.7e-07
probability diff > 1e-6: 0
```

Display panels were regenerated using predicted-class Grad-CAM and synced to:

```text
results/resnet50_final_conv3_5_preprocess_dropout04/gradcam/
model-comparison-repo/models/resnet50/final_results/gradcam/
```

## ConvNeXt Status

Final model:

```text
ConvNeXt/results/convnext_base_torch_final_stage3_4_dropout04/final/stage3_4/final_model.pt
```

Fine-tuning setting:

```text
stage3_4
```

Clarification:

- `stage3_4` is a training-strategy label, not a literal timm module name.
- timm ConvNeXt has `stages.0` through `stages.3`.
- The final model unfroze `stages.2`, `stages.3`, and the classifier head.
- Norm parameters remained frozen via `freeze_norm=True`.

Grad-CAM target layer:

```text
stages.3.blocks.2
```

Activation check:

```text
activation shape: (1, 1024, 7, 7)
gradient shape: (1, 1024, 7, 7)
```

This confirms the target layer is `[B, C, H, W]`, so the Grad-CAM channel-weighting code is dimensionally correct.

New raw export script:

```text
gradcam_mask_overlap/tools/export_convnext_raw_gradcam.py
```

Outputs:

```text
gradcam_mask_overlap/data/saliency_maps/convnext_base/gradcam_predclass/
gradcam_mask_overlap/data/saliency_maps/convnext_base/gradcam_malignant/
```

Verification:

```text
gradcam: 290 raw maps
gradcam_malignant: 290 raw maps
paths exist: 580 / 580
heatmap shape: (224, 224)
heatmap dtype: float32
heatmap range: 0-1
all-zero heatmap: false
probability max diff vs source predictions: 2.17e-05
probability mean diff vs source predictions: 4.25e-06
probability diff > 1e-4: 0
prediction label mismatch: 0
```

The ConvNeXt probability difference is small and does not affect labels. It is treated as normal PyTorch/GPU floating-point variation.

Display panels were regenerated using predicted-class Grad-CAM and synced to:

```text
ConvNeXt/results/convnext_base_torch_final_stage3_4_dropout04/gradcam/
model-comparison-repo/models/convnext_base/final_results/gradcam/
```

Display panel counts:

```text
correct_cancer: 10
correct_benign: 10
false_positive: 10
false_negative: 10
```

## Metadata Updates

Updated:

```text
gradcam_mask_overlap/metadata/saliency_manifest.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest_summary.json
project_table_inventory/table_inventory_for_next_steps.md
project_table_inventory/table_inventory_for_next_steps.csv
```

Current summary:

```text
saliency_manifest rows: 1500
duplicate saliency_id: 0
analysis_ready_overlap_manifest rows: 1740
ready_for_overlap_rows: 1160
```

Ready rows:

```text
resnet50:gradcam = 290
resnet50:gradcam_malignant = 290
convnext_base:gradcam = 290
convnext_base:gradcam_malignant = 290
```

Remaining pending/missing saliency entries:

```text
Swin and ViT still need raw saliency map export before four-model overlap comparison.
```

## Cleanup

Removed temporary / non-final artifacts:

```text
__pycache__ / *.pyc files in touched project folders
ConvNeXt smoke output directories
Matplotlib cache directories
```

Removed smoke directories:

```text
ConvNeXt/results/convnext_base_torch_final_stage3_4_dropout04/gradcam_smoke
ConvNeXt/results/convnext_small_smoke
ConvNeXt/results/convnext_tiny_smoke
```

Final cleanup check:

```text
No project-level *.pyc, *.tmp, *.bak, *~, .DS_Store found in touched areas.
No remaining ConvNeXt smoke directories.
No remaining project-level .matplotlib cache directories checked today.
```

## GPU Note

Default sandbox could not see the GPU:

```text
torch.cuda.is_available() = False inside sandbox
nvidia-smi blocked inside sandbox
```

After running outside the sandbox, PyTorch detected:

```text
NVIDIA GeForce RTX 5090
torch cuda available: True
```

ConvNeXt full raw Grad-CAM export was run with GPU outside the sandbox.

## Files to Remember Tomorrow

Primary manifests:

```text
gradcam_mask_overlap/metadata/model_predictions.csv
gradcam_mask_overlap/metadata/test_roi_mask_manifest.csv
gradcam_mask_overlap/metadata/saliency_manifest.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
mask_qc_outputs/original_mask_binary_qc/manual_test_labels/manual_test_labels_image_level_english.csv
```

Main scripts:

```text
gradcam_mask_overlap/tools/export_resnet50_raw_gradcam.py
gradcam_mask_overlap/tools/export_convnext_raw_gradcam.py
ResNet50/tools/gradcam_resnet50.py
ConvNeXt/tools/gradcam_convnext_torch.py
```

## Recommended Next Steps

1. Export Swin raw Grad-CAM in the same two-version format:
   - `gradcam` / `predicted_label`
   - `gradcam_malignant` / `cancer_score`
2. Export ViT raw attention/saliency maps in a matching manifest-compatible format.
3. Rebuild `analysis_ready_overlap_manifest.csv` after each model export.
4. Start overlap metrics using the main analysis filter:

```text
saliency_method == "gradcam_malignant"
```

5. Summarize overlap/mIoU overall, by model, by prediction group, and by manual annotation type.

## Short Takeaway

ResNet50 and ConvNeXt are now ready for overlap analysis. Both have complete predicted-class Grad-CAM for visualization and malignant/Cancer-score Grad-CAM for quantitative mask-overlap experiments. Swin and ViT are the next models to bring into the same structure.
