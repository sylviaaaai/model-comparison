# Saliency Map Export and Alignment QC

This document summarizes how the final saliency maps were exported and how they
were checked before lesion-mask overlap analysis.

## Purpose

Before computing overlap metrics, we needed to confirm that each saliency map
matched the correct final model, test image, prediction record, and ROI mask.
This step was treated as an input-quality check rather than a result table.

## Final Saliency Methods

The primary overlap analysis used one saliency method per final model:

| Model | Primary saliency for overlap | Explanation family | Class-specific | Test images |
| --- | --- | --- | --- | ---: |
| ResNet50 | malignant-class Grad-CAM | Grad-CAM | yes | 290 |
| ConvNeXt-Base | malignant-class Grad-CAM | Grad-CAM | yes | 290 |
| Swin-Base | malignant-class Grad-CAM | Grad-CAM | yes | 290 |
| ViT-B/16 | attention rollout | Attention rollout | no | 290 |

For ResNet50, ConvNeXt-Base, and Swin-Base, predicted-class Grad-CAM maps were
also exported for qualitative review, but the primary overlap experiment used
malignant-class Grad-CAM. ViT-B/16 used standard attention rollout for the
overlap analysis.

## Predicted-Class Maps vs Malignant-Class Maps

We kept qualitative visualization and quantitative overlap analysis separate.

| Use case | Saliency target | Reason |
| --- | --- | --- |
| Qualitative display panels | Predicted class | Shows the evidence for the class the model actually predicted, so it is most interpretable for TP/TN/FP/FN example panels. |
| Quantitative overlap analysis | Malignant class / cancer score | Keeps the target class consistent across all test images and measures whether malignant evidence spatially overlaps the lesion annotation. |

Therefore, the heatmaps shown in qualitative example figures may be
predicted-class maps, while the main overlap metrics use malignant-class maps
for ResNet50, ConvNeXt-Base, and Swin-Base. This design avoids mixing two
different questions:

```text
Display question:
  What visual evidence supported the model's actual prediction?

Overlap question:
  Where is the model's malignant-class evidence, and does it overlap the lesion mask?
```

This distinction is also important for interpreting zero-saliency cases. A
malignant-class Grad-CAM map can be zero when a model predicts benign and has
little positive evidence for the malignant class at the selected layer.

## Exported Saliency Files

All raw saliency maps were exported as normalized `224 x 224` arrays, with PNG
previews for visual inspection.

| Saliency folder | NPY | PNG |
| --- | ---: | ---: |
| `resnet50/gradcam_predclass` | 290 | 290 |
| `resnet50/gradcam_malignant` | 290 | 290 |
| `convnext_base/gradcam_predclass` | 290 | 290 |
| `convnext_base/gradcam_malignant` | 290 | 290 |
| `swin_base/gradcam_predclass` | 290 | 290 |
| `swin_base/gradcam_malignant` | 290 | 290 |
| `vit_b16/attention_rollout_predclass` | 290 | 290 |

## Alignment Checks

The prediction files were checked against the unified prediction metadata table.
Each model had 290 prediction rows, and all labels/predicted classes matched
the metadata table.

| Model | Prediction rows | Metadata rows | Max probability diff | Label mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 290 | 290 | 0.0 | 0 | 0 |
| ViT-B/16 | 290 | 290 | 0.0 | 0 | 0 |
| ConvNeXt-Base | 290 | 290 | 0.0 | 0 | 0 |
| Swin-Base | 290 | 290 | 0.0 | 0 | 0 |

The saliency manifest was then checked against recomputed model probabilities.
All differences were very small and did not create any label or prediction
mismatch.

| Model | Saliency rows checked | Max probability diff |
| --- | ---: | ---: |
| ResNet50 | 580 | 4.7e-7 |
| ViT-B/16 | 290 | 1.23e-6 |
| ConvNeXt-Base | 580 | 2.17e-5 |
| Swin-Base | 580 | 1.08e-6 |

## Cache Corrections Before Analysis

Two historical prediction-cache issues were resolved before the final overlap
analysis:

- Swin-Base had a small mismatch between an older prediction CSV and
  probabilities recomputed from the current final checkpoint. The stale
  prediction CSV was backed up, predictions were regenerated, and the saliency
  export script was aligned with the final training/inference path.
- ViT-B/16 had a smaller cache difference from the current final checkpoint.
  The stale prediction CSV was backed up, predictions were regenerated, and
  the metadata, saliency manifest, and analysis-ready manifest were updated.

After these corrections, the final overlap analysis used the synchronized
prediction and saliency metadata.

## Analysis-Ready Manifest

The final analysis-ready manifest contained all expected rows:

```text
analysis_manifest_rows = 2030
ready_for_overlap_rows = 2030
saliency_status_counts = {"raw_map_available": 2030}
```

Ready rows by primary and supplementary method:

| Model-method | Ready rows |
| --- | ---: |
| `resnet50:gradcam` | 290 |
| `resnet50:gradcam_malignant` | 290 |
| `convnext_base:gradcam` | 290 |
| `convnext_base:gradcam_malignant` | 290 |
| `swin_base:gradcam` | 290 |
| `swin_base:gradcam_malignant` | 290 |
| `vit_b16:attention_rollout` | 290 |

## Takeaway

The final overlap analysis was performed only after final model predictions,
saliency maps, image IDs, and metadata were synchronized. All primary saliency
maps were available for the 290 final test images, and prediction/saliency
probability agreement was within numerical tolerance.

## Audit Files

- `gradcam_mask_overlap/experiment_logs/2026-06-29_prediction_saliency_alignment_log.md`
- `gradcam_mask_overlap/metadata/model_predictions.csv`
- `gradcam_mask_overlap/metadata/saliency_manifest.csv`
- `gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv`
