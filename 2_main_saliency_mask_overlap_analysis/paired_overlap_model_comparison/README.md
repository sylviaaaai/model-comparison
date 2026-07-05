# Paired Overlap Model Comparison

## Purpose

This folder tests whether ConvNeXt-Base has stronger lesion-mask overlap than the other models when the comparison is made on the same images.

The analysis is paired: for each image, metric, scope, and comparator model, the script compares ConvNeXt-Base against one other model using only rows where both models have a valid value. This is different from simply comparing model-level means, because the paired design controls for image difficulty and keeps the denominator matched within each comparison.

ConvNeXt-Base is used as the reference model because the main overlap summaries identify it as the strongest overall model.

## Analysis Design

For each analysis scope and overlap metric:

1. Select the image set for the scope.
2. Pair ConvNeXt-Base with one comparator model on the same image.
3. Keep only valid image-level pairs where both models have a finite value for that metric.
4. Compute the image-level paired difference.
5. Convert the difference so that a positive effect always favors ConvNeXt-Base.
6. Estimate the mean paired effect and a 95% confidence interval using paired bootstrap resampling.

The two scopes are:

| Scope | Rule | Role |
| --- | --- | --- |
| `main_yes` | `usable_for_overlap == yes` | Primary strict usable-mask analysis. |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | Sensitivity analysis including cautious masks. |

## Metrics

The paired analysis uses the same seven compact overlap metrics as the main overlap summary:

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `energy_inside_mask` | Fraction of saliency energy inside the lesion mask. | Higher is better. |
| `normalized_energy` | Mask energy normalized against expected/background energy. | Higher is better. |
| `peak_in_mask` | Whether the saliency peak falls inside the lesion mask. | Higher is better. |
| `peak_distance_px` | Pixel distance from saliency peak to lesion mask/target. | Lower is better. |
| `mask_coverage_top20` | Fraction of lesion mask covered by the top 20% saliency region. | Higher is better. |
| `iou_top20` | IoU between lesion mask and top 20% saliency region. | Higher is better. |
| `dice_top20` | Dice score between lesion mask and top 20% saliency region. | Higher is better. |

## Effect Direction

Positive paired effects favor ConvNeXt-Base in all result tables.

For higher-is-better metrics, the effect is:

```text
ConvNeXt value - comparator value
```

For `peak_distance_px`, lower is better, so the sign is reversed:

```text
comparator distance - ConvNeXt distance
```

This means a positive `peak_distance_px` effect still favors ConvNeXt-Base because it indicates that ConvNeXt has a smaller, better peak distance.

## Results

| File | Meaning |
| --- | --- |
| `results/00_convnext_paired_key_ci_table.csv` | Compact ConvNeXt-vs-other paired comparison table with mean effects, 95% CIs, and better-rate values. This is the best first table to read. |
| `results/01_convnext_vs_others_paired_summary.csv` | Full paired summary by scope, metric, and comparator, including model means, raw deltas, effect deltas, bootstrap CIs, better/worse/tie counts, and pair-validity notes. |
| `results/03_convnext_vs_others_per_image_long.csv` | Per-image audit table. This is large and is intended for traceability/debugging, not as the main reading table. |

Recommended reading order:

1. Start with `results/00_convnext_paired_key_ci_table.csv` for the compact result.
2. Use `results/01_convnext_vs_others_paired_summary.csv` when you need full paired statistics and validity counts.
3. Use `results/03_convnext_vs_others_per_image_long.csv` only when you need to inspect image-level paired differences.

## Column Guide

### `00_convnext_paired_key_ci_table.csv`

| Column | Meaning |
| --- | --- |
| `scope` | Analysis scope. |
| `metric` | Overlap metric. |
| `comparator_model` | Model compared against ConvNeXt-Base. |
| `n_pairs` | Number of valid paired images for that comparison. |
| `effect_mean_delta_positive_favors_ConvNeXt` | Mean paired effect, signed so positive values favor ConvNeXt-Base. |
| `ci_low`, `ci_high` | 95% paired bootstrap confidence interval for the mean effect. |
| `ci_excludes_zero` | Whether the 95% CI excludes 0. |
| `convnext_better_rate` | Fraction of valid image pairs where ConvNeXt-Base is better than the comparator. |

### `01_convnext_vs_others_paired_summary.csv`

| Column | Meaning |
| --- | --- |
| `direction` | Whether the raw metric is `higher_is_better` or `lower_is_better`. |
| `model_a`, `model_b` | Paired models. In this output, `model_a` is ConvNeXt-Base. |
| `n_scope_images` | Number of images in the scope before pair-validity filtering. |
| `n_pairs` | Number of images where both models have finite values for the metric. |
| `n_invalid_pairs` | Number of scope images excluded because at least one model value is invalid. |
| `model_a_mean`, `model_b_mean` | Mean metric value for each model on the valid paired rows. |
| `raw_mean_delta_model_a_minus_model_b` | Raw mean difference before sign correction. |
| `effect_mean_delta_positive_favors_model_a` | Mean paired effect after sign correction, so positive favors ConvNeXt-Base. |
| `effect_median_delta_positive_favors_model_a` | Median paired effect after sign correction. |
| `effect_bootstrap_ci_low`, `effect_bootstrap_ci_high` | 95% paired bootstrap CI for the mean paired effect. |
| `model_a_better_n`, `model_b_better_n`, `tie_n` | Counts of valid image pairs where ConvNeXt-Base is better, worse, or tied. |
| `model_a_better_rate` | `model_a_better_n / n_pairs`. |
| `pair_validity_note` | Note shown when invalid pairs reduce the paired denominator. |

### `03_convnext_vs_others_per_image_long.csv`

| Column | Meaning |
| --- | --- |
| `image_id` | Image identifier. |
| `reference_model` | Reference model, ConvNeXt-Base. |
| `comparator_model` | Model compared against ConvNeXt-Base. |
| `reference_value`, `comparator_value` | Image-level metric values. |
| `raw_delta_reference_minus_comparator` | Raw image-level difference before sign correction. |
| `effect_delta_positive_favors_reference` | Image-level effect after sign correction. Positive favors ConvNeXt-Base. |
| `valid_pair` | Whether both models had valid values for this image and metric. |

## Validity Notes

Most metrics use the full paired denominator within each scope. The main exception is `peak_distance_px` for ResNet50 comparisons, because ResNet50 has zero-saliency cases where peak distance is invalid.

In the key table:

| Scope | Comparison | Metric | Scope images | Valid pairs | Invalid pairs |
| --- | --- | --- | ---: | ---: | ---: |
| `main_yes` | ConvNeXt-Base vs ResNet50 | `peak_distance_px` | 183 | 167 | 16 |
| `sensitivity_yes_cautious` | ConvNeXt-Base vs ResNet50 | `peak_distance_px` | 279 | 250 | 29 |

Therefore, the ConvNeXt-vs-ResNet50 `peak_distance_px` result should be interpreted cautiously. Its paired CI is computed only on images where both models have valid finite peak-distance values.

Better-rate values should also be interpreted with ties in mind. Binary or near-discrete metrics such as `peak_in_mask`, and distance metrics with many zero distances, can have many ties, so the mean effect and confidence interval are usually more informative than the better-rate alone.

## Result Snapshot

Positive deltas favor ConvNeXt-Base. For `peak_distance_px`, the sign is reversed internally so a positive effect still means ConvNeXt has the smaller, better distance.

| Scope | Metric | Comparator | Paired N | Mean paired effect | 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| `main_yes` | Dice top 20% | ResNet50 | 183 | 0.019 | [-0.003, 0.043] |
| `main_yes` | Dice top 20% | ViT_B16 | 183 | 0.305 | [0.289, 0.321] |
| `main_yes` | Dice top 20% | Swin_Base | 183 | 0.131 | [0.108, 0.154] |
| `sensitivity_yes_cautious` | Dice top 20% | ResNet50 | 279 | 0.042 | [0.022, 0.062] |
| `sensitivity_yes_cautious` | Dice top 20% | ViT_B16 | 279 | 0.312 | [0.297, 0.325] |
| `sensitivity_yes_cautious` | Dice top 20% | Swin_Base | 279 | 0.141 | [0.122, 0.159] |

## Conclusion

ConvNeXt-Base is consistently higher than ViT_B16 and Swin_Base across the paired overlap metrics. Its comparison with ResNet50 is closer in the strict `main_yes` set: several ResNet50 comparisons have small paired effects and confidence intervals that cross zero. In the broader `sensitivity_yes_cautious` set, ConvNeXt-Base more clearly favors the top-20% overlap metrics, including Dice, IoU, and mask coverage.

Overall, the paired analysis supports the main overlap-summary conclusion: ConvNeXt-Base has the strongest lesion-mask overlap pattern overall, while ResNet50 is the closest comparator.
