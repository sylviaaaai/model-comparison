# Results CSV Guide

This directory contains the CSV outputs for the main-vs-sensitivity saliency/mask overlap analysis. The files are related, but they are not duplicates: each table answers a different question.

## Recommended Reading Order

1. `overlap_usable_scope_counts.csv`
   - Read this first to understand the denominator for each analysis scope.
   - It tells you how many images are included in the strict main set, the broader sensitivity set, the excluded set, and the all-image context set.

2. `overlap_usable_scope_model_metric_summary.csv`
   - Read this second for the actual model comparison.
   - It summarizes the overlap metrics for each `scope x model x metric`.

3. `overlap_peak_distance_validity_by_scope.csv`
   - Read this last as a validity check for `peak_distance_px`.
   - It explains why some ResNet50 peak-distance rows have fewer valid observations than the total image count.

## How The Tables Differ

| CSV | Role |
| --- | --- |
| `overlap_usable_scope_counts.csv` |  Defines the denominators before interpreting model metrics. |
| `overlap_usable_scope_model_metric_summary.csv` | Provides the core mean/median/SD/min/max results for each model and metric. |
| `overlap_peak_distance_validity_by_scope.csv` |  Documents invalid peak-distance cases, mainly ResNet50 zero-saliency maps. |

## Analysis Scopes

| Scope | Meaning | Role |
| --- | --- | --- |
| `main_yes` | `usable_for_overlap == yes` | Primary result set. This is the main denominator for the overlap conclusion. |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | Sensitivity analysis. Checks whether the main conclusion remains stable after including cautious masks. |
| `excluded_no` | `usable_for_overlap == no` | Excluded/limitation cases only. Do not use as the primary performance denominator. |
| `all_images` | All final test images | Context only. Useful for checking the overall dataset, not for the primary conclusion. |

## Table 1: `overlap_usable_scope_counts.csv`

This is the denominator and composition table.

Columns:

| Column | Meaning |
| --- | --- |
| `scope` | Analysis scope. |
| `n_images` | Number of images in that scope. |
| `n_malignant` | Number of malignant cases. |
| `n_benign` | Number of benign cases. |
| `usable_yes` | Count with strict usable overlap mask. |
| `usable_cautious` | Count with cautious usable overlap mask. |
| `usable_no` | Count excluded from usable overlap analysis. |
| `marker_positive` | Number of marker-positive images. |
| `marker_negative` | Number of marker-negative images. |

Main contents:

| Scope | Images | Malignant | Benign | Usable yes | Usable cautious | Usable no | Marker + | Marker - |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_yes` | 183 | 107 | 76 | 183 | 0 | 0 | 18 | 165 |
| `sensitivity_yes_cautious` | 279 | 141 | 138 | 183 | 96 | 0 | 22 | 257 |
| `excluded_no` | 11 | 6 | 5 | 0 | 0 | 11 | 3 | 8 |
| `all_images` | 290 | 147 | 143 | 183 | 96 | 11 | 25 | 265 |

Key reading:

- The primary analysis uses 183 strict usable images.
- The sensitivity analysis adds 96 cautious images, increasing the denominator to 279.
- There are 11 excluded `usable_no` images.
- The full final-test context contains 290 images.

## Table 2: `overlap_usable_scope_model_metric_summary.csv`

This is the core model-performance table. It has 112 data rows: 4 scopes x 4 models x 7 metrics.

Columns:

| Column | Meaning |
| --- | --- |
| `scope` | Analysis scope. |
| `model` | Model name: `ResNet50`, `ViT_B16`, `ConvNeXt_Base`, or `Swin_Base`. |
| `metric` | Overlap metric being summarized. |
| `n_images` | Total images in that scope. |
| `n_valid` | Valid observations for that metric. Usually equals `n_images`, except some `peak_distance_px` rows. |
| `mean` | Mean metric value. |
| `median` | Median metric value. |
| `sd` | Standard deviation. |
| `min` | Minimum value. |
| `max` | Maximum value. |

Metrics:

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `energy_inside_mask` | Fraction of saliency energy inside the lesion mask. | Higher is better. |
| `normalized_energy` | Mask energy normalized against expected/background energy. | Higher is better. |
| `peak_in_mask` | Whether the saliency peak falls inside the lesion mask. Mean is the fraction of images where this is true. | Higher is better. |
| `peak_distance_px` | Pixel distance from saliency peak to lesion mask/target. | Lower is better. |
| `mask_coverage_top20` | Fraction of lesion mask covered by the top 20% saliency region. | Higher is better. |
| `iou_top20` | IoU between lesion mask and top 20% saliency region. | Higher is better. |
| `dice_top20` | Dice score between lesion mask and top 20% saliency region. | Higher is better. |

### Primary Scope: `main_yes`

Mean values by model:

| Metric | Best model by mean | ResNet50 | ViT_B16 | ConvNeXt_Base | Swin_Base |
| --- | --- | ---: | ---: | ---: | ---: |
| `energy_inside_mask` | ConvNeXt_Base | 0.672 | 0.392 | 0.755 | 0.593 |
| `normalized_energy` | ConvNeXt_Base | 1.262 | 0.725 | 1.406 | 1.107 |
| `peak_in_mask` | ConvNeXt_Base | 0.743 | 0.137 | 0.792 | 0.607 |
| `peak_distance_px` | ResNet50 | 3.310 | 38.373 | 3.359 | 16.245 |
| `mask_coverage_top20` | ConvNeXt_Base | 0.278 | 0.080 | 0.291 | 0.201 |
| `iou_top20` | ConvNeXt_Base | 0.264 | 0.063 | 0.273 | 0.182 |
| `dice_top20` | ConvNeXt_Base | 0.402 | 0.116 | 0.421 | 0.290 |

Key reading:

- ConvNeXt_Base is best on 6 of 7 metrics in the primary analysis.
- ResNet50 is best only on `peak_distance_px`, where lower is better. Its mean peak distance is 3.310 px, very close to ConvNeXt_Base at 3.359 px.
- This ResNet50 `peak_distance_px` result should be interpreted cautiously: in `main_yes`, 16 ResNet50 zero-saliency cases have invalid peak distance and are excluded from that mean (`n_valid = 167` out of 183). Therefore, the apparent ResNet50 advantage on this one metric may be affected by missing/invalid zero-saliency maps. In the broader `sensitivity_yes_cautious` set, ResNet50 has 29 invalid peak-distance cases and ConvNeXt_Base becomes best on `peak_distance_px`.
- ViT_B16 is consistently weakest across the overlap metrics, especially `peak_in_mask`, `iou_top20`, and `dice_top20`.
- Swin_Base is generally between ResNet50/ConvNeXt_Base and ViT_B16.

### Sensitivity Scope: `sensitivity_yes_cautious`

Mean values by model:

| Metric | Best model by mean | ResNet50 | ViT_B16 | ConvNeXt_Base | Swin_Base |
| --- | --- | ---: | ---: | ---: | ---: |
| `energy_inside_mask` | ConvNeXt_Base | 0.637 | 0.379 | 0.744 | 0.574 |
| `normalized_energy` | ConvNeXt_Base | 1.211 | 0.710 | 1.410 | 1.086 |
| `peak_in_mask` | ConvNeXt_Base | 0.703 | 0.133 | 0.792 | 0.534 |
| `peak_distance_px` | ConvNeXt_Base | 3.968 | 40.798 | 3.211 | 19.606 |
| `mask_coverage_top20` | ConvNeXt_Base | 0.262 | 0.074 | 0.291 | 0.193 |
| `iou_top20` | ConvNeXt_Base | 0.245 | 0.057 | 0.271 | 0.172 |
| `dice_top20` | ConvNeXt_Base | 0.377 | 0.107 | 0.419 | 0.278 |

Key reading:

- After adding cautious masks, ConvNeXt_Base is best on all 7 metrics.
- The main ordering remains stable: ConvNeXt_Base is strongest overall, ResNet50 is the closest comparator, Swin_Base is intermediate, and ViT_B16 is weakest.
- The sensitivity set increases the denominator from 183 to 279 images, so it supports the robustness of the primary conclusion.

### Auxiliary Scopes: `excluded_no` and `all_images`

Use these as context, not as the main result:

- `excluded_no` has only 11 images and represents cases excluded from the primary usable-mask analysis. It is too small and selected to serve as a primary model-comparison denominator.
- `all_images` has 290 images and combines strict usable, cautious, and excluded images. It is useful for a whole-dataset context check, but the primary interpretation should come from `main_yes`, with robustness checked by `sensitivity_yes_cautious`.

In `all_images`, ConvNeXt_Base remains best on all 7 metrics by mean, including `peak_distance_px` where lower is better.

## Table 3: `overlap_peak_distance_validity_by_scope.csv`

This table documents which `peak_distance_px` values are valid.

Columns:

| Column | Meaning |
| --- | --- |
| `model` | Model name. |
| `scope` | Analysis scope. |
| `n_images` | Total images in the scope. |
| `n_valid_peak_distance` | Images with valid peak-distance values. |
| `n_invalid` | Images where peak distance could not be computed. |
| `reason` | Explanation for invalid values. |

Main contents:

| Scope | Model affected | Images | Valid peak distance | Invalid | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| `main_yes` | ResNet50 | 183 | 167 | 16 | Peak distance was computed only for valid non-zero saliency maps. |
| `sensitivity_yes_cautious` | ResNet50 | 279 | 250 | 29 | Peak distance was computed only for valid non-zero saliency maps. |
| `excluded_no` | None | 11 | 11 for every model | 0 | All models valid. |
| `all_images` | ResNet50 | 290 | 261 | 29 | Peak distance was computed only for valid non-zero saliency maps. |

All ViT_B16, ConvNeXt_Base, and Swin_Base rows have valid peak-distance values for every image in every scope.

Key reading:

- This table is not another performance summary.
- It only explains `peak_distance_px` validity.
- It matters because ResNet50 has zero-saliency cases, so its `peak_distance_px` mean is computed on fewer images than its other metrics in `main_yes`, `sensitivity_yes_cautious`, and `all_images`.

## Overall Interpretation

The primary result should be read from `main_yes` in `overlap_usable_scope_model_metric_summary.csv`, using `overlap_usable_scope_counts.csv` to confirm the denominator. The sensitivity result should then be checked in `sensitivity_yes_cautious`.

The main conclusion is stable:

- ConvNeXt_Base has the strongest lesion-mask overlap overall.
- ResNet50 is the closest comparator, especially for peak location.
- Swin_Base is intermediate.
- ViT_B16 attention rollout has the weakest lesion-mask overlap.
