# Primary Overlap Metrics

## Purpose

This folder contains the first numerical saliency-mask overlap computation. It asks how strongly each model's primary saliency map is spatially aligned with the annotated lesion mask for the same final test images.

This is the upstream metric-generation step for later analyses. It uses all 290 final test images per primary model-method pair. Manual mask-usability filtering (`yes`, `cautious`, `no`) is applied later in the main-vs-sensitivity analysis, so the summaries here should be read as the full initial overlap metric summary rather than the final strict usable-mask result.

## Inputs

The primary model-method pairs are:

| Model | Saliency method | Explanation family |
| --- | --- | --- |
| ResNet50 | `gradcam_malignant` | Grad-CAM |
| ConvNeXt-Base | `gradcam_malignant` | Grad-CAM |
| Swin-Base | `gradcam_malignant` | Grad-CAM |
| ViT-B/16 | `attention_rollout` | Attention rollout |

Masks and saliency maps are evaluated in the common 224 x 224 ROI analysis space. Each model has 290 image-method pairs in the output summaries.

## Analysis Workflow

1. Load the analysis-ready saliency/mask manifest.
2. Keep the primary saliency method for each model.
3. Resize lesion masks to the 224 x 224 ROI space.
4. Normalize each saliency map to 0-1.
5. Compute continuous lesion-localization metrics.
6. Binarize saliency maps using fixed thresholds and top-fraction thresholds.
7. Compute binary overlap metrics between the binarized saliency region and lesion mask.
8. Summarize results at the model level.

## Metric Definitions

### Continuous Metrics

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `cam_energy_in_mask` | Fraction of total saliency energy inside the lesion mask. | Higher is better. |
| `normalized_energy_in_mask` | `cam_energy_in_mask` divided by lesion area fraction. Values above 1 mean saliency is more concentrated in the lesion than expected from area alone. | Higher is better. |
| `peak_in_mask` / `peak_in_mask_rate` | Whether the maximum saliency pixel lies inside the lesion mask. The model summary reports the fraction of images where this is true. | Higher is better. |
| `peak_distance_to_mask_px` | Pixel distance from the saliency peak to the lesion mask. If the peak is inside the mask, the distance is 0. | Lower is better. |
| `mask_area_fraction` | Fraction of the ROI occupied by the lesion mask. | Context variable, not a model-performance metric. |
| `cam_sum` | Total normalized saliency energy in the ROI. | QC/context variable, not a direct overlap score. |

### Binary Overlap Metrics

The saliency map is also converted into a binary saliency region. The binary region is compared with the lesion mask.

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `iou` | Intersection-over-union between binary saliency region and lesion mask. | Higher is better. |
| `dice` | Dice score between binary saliency region and lesion mask. | Higher is better. |
| `mask_coverage_by_cam` | Fraction of lesion-mask pixels covered by the binary saliency region. | Higher is better. |
| `saliency_precision` | Fraction of binary saliency-region pixels that fall inside the lesion mask. | Higher is better. |
| `cam_area_fraction` | Fraction of the ROI selected as binary saliency region. | Context variable for interpreting thresholded metrics. |

### Threshold Settings

Two thresholding strategies are used:

| Threshold type | Examples | Meaning |
| --- | --- | --- |
| Fixed threshold | `fixed_0.1`, `fixed_0.2`, ..., `fixed_0.9` | Select pixels whose normalized saliency value is at least the fixed threshold. |
| Top-fraction threshold | `top_10pct`, `top_20pct`, `top_30pct`, `top_40pct` | Select the top fraction of ROI pixels by saliency value. |

The compact downstream summaries focus on `top_20pct`, because it gives every non-zero saliency map the same saliency-region size and avoids choosing a model-specific fixed intensity threshold.

## Results

| File | Meaning |
| --- | --- |
| `results/overlap_continuous_model_summary.csv` | Model-level summary of continuous localization metrics: energy in mask, normalized energy, peak-in-mask rate, peak distance, mask area fraction, and saliency sum. |
| `results/overlap_selected_model_summary.csv` | Compact model-level summary combining continuous metrics with selected binary overlap settings: `fixed_0.1`, `top_10pct`, and `top_20pct`. |

Recommended reading order:

1. Start with `results/overlap_continuous_model_summary.csv` to understand basic lesion localization.
2. Then read `results/overlap_selected_model_summary.csv` for the binary overlap metrics used in downstream model comparisons.
3. Use `example_qc_panels/` to visually check image-mask-heatmap alignment and edge cases.

## Main Result Snapshot

The table below summarizes the all-image primary overlap result across 290 image-method pairs per model.

| Model | N | Peak-in-mask rate | Zero-saliency pairs | Energy in mask mean | Normalized energy mean | Peak distance mean px | Top-20% IoU mean | Top-20% Dice mean | Top-20% mask coverage mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Base | 290 | 0.786 | 0 | 0.743 | 1.410 | 3.266 | 0.271 | 0.419 | 0.291 |
| ResNet50 | 290 | 0.707 | 29 | 0.641 | 1.220 | 3.994 | 0.247 | 0.379 | 0.264 |
| Swin-Base | 290 | 0.531 | 0 | 0.577 | 1.092 | 19.409 | 0.174 | 0.281 | 0.195 |
| ViT-B/16 | 290 | 0.128 | 0 | 0.381 | 0.714 | 41.055 | 0.059 | 0.110 | 0.076 |

Key reading:

- ConvNeXt-Base has the strongest overall lesion-mask overlap in this all-image primary summary.
- ResNet50 is the closest comparator on continuous localization metrics, but it has 29 zero-saliency pairs.
- Swin-Base is intermediate.
- ViT-B/16 attention rollout is the least lesion-localized by these overlap metrics.

## QC Panels

`example_qc_panels/` contains rendered examples for quick visual inspection:

| Folder | Purpose |
| --- | --- |
| `example_qc_panels/clean_usable_mask/` | Clean example where image, mask, and heatmap alignment are straightforward. |
| `example_qc_panels/cautious_difficult_mask/` | Difficult/cautious mask example showing why later mask-usability sensitivity analysis is needed. |
| `example_qc_panels/resnet50_zero_saliency/` | ResNet50 zero-saliency example showing why some peak-distance values are invalid. |

These panels are smoke/QC examples, not the full qualitative saliency analysis. The broader qualitative comparison is in `../analysis_cohort_and_input_alignment/qualitative_saliency_examples_across_mask_types_and_artifacts/`.

## Zero-Saliency Caveat

ResNet50 has 29 zero-saliency image-method pairs in the all-image summary. For these rows, the saliency map has no meaningful peak, so `peak_distance_to_mask_px` is invalid at the image level.

This matters most in later paired comparisons involving `peak_distance_px`: ConvNeXt vs ResNet50 uses 167/183 valid images in `main_yes` and 250/279 valid images in `sensitivity_yes_cautious`. Therefore, ResNet50 peak-distance comparisons should be interpreted with this validity caveat.

## Relationship To Later Analyses

This folder provides the base overlap metrics. Later folders use these metrics for more focused questions:

| Later analysis | How it uses these metrics |
| --- | --- |
| `main_vs_sensitivity_overlap_result/` | Applies strict vs cautious mask-usability scopes and reports the main overlap conclusion. |
| `paired_overlap_model_comparison/` | Performs paired ConvNeXt-vs-other comparisons on the same images. |
| `outside_context_partition_saliency_energy_analysis/` | Extends the question from lesion overlap to lesion/perilesional/outside saliency distribution. |

## Conclusion

Across the initial all-image primary overlap metrics, ConvNeXt-Base shows the strongest lesion-mask alignment, ResNet50 is the closest comparator, Swin-Base is intermediate, and ViT-B/16 attention rollout is weakest. These results motivate the later strict usable-mask, sensitivity, and paired analyses.

