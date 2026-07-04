# Primary Overlap Metrics

This folder summarizes the first numerical saliency-mask overlap computation. The analysis asks how much each model's saliency is spatially aligned with the annotated lesion mask for the same final test image.

## Inputs and Metrics

The primary model-method pairs are ResNet50 Grad-CAM, ConvNeXt-Base Grad-CAM, Swin-Base Grad-CAM, and ViT-B/16 attention rollout. The raw image-method table contains 290 test images per model-method pair. Manual mask-usability filtering is applied later in the main-vs-sensitivity analysis.

Continuous metrics include `cam_energy_in_mask`, `normalized_energy_in_mask`, `peak_in_mask`, and `peak_distance_to_mask_px`. Binary metrics include IoU, Dice, mask coverage, and saliency precision after fixed-threshold and top-fraction saliency binarization. The compact downstream summaries use the top-20% binary saliency setting.

## Results

| Table | Meaning |
| --- | --- |
| `results/overlap_continuous_model_summary.csv` | Model-level continuous saliency-mask localization metrics. |
| `results/overlap_selected_model_summary.csv` | Selected compact overlap metrics used for manuscript-level comparisons. |

## QC Panels

`example_qc_panels/` contains one clean usable example, one cautious/difficult mask example, and one ResNet50 zero-saliency example. These panels demonstrate image-mask-heatmap alignment; the full qualitative saliency comparison is in `../analysis_cohort_and_input_alignment/qualitative_saliency_examples_across_mask_types_and_artifacts/`.

## Zero-Saliency Caveat

ResNet50 zero-saliency mainly affects paired overlap `peak_distance_px`, because no saliency peak exists when the malignant-class Grad-CAM map is all zero. Therefore, ConvNeXt vs ResNet50 paired `peak_distance_px` uses 167/183 valid images in `main_yes` and 250/279 valid images in `sensitivity_yes_cautious`.
