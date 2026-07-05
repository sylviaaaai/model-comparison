# Main vs Sensitivity Overlap Result

## Purpose

This folder reports the main saliency-mask overlap result and tests whether that result is robust to a broader mask-usability definition.

The primary question is: among images with usable lesion masks, which model's saliency maps align most strongly with the annotated lesion ROI? The sensitivity question is whether the model ordering changes when `cautious` masks are added to the strict usable-mask cohort.

This is the main result folder for the saliency-mask overlap chapter. Other folders provide upstream metric generation, paired statistical comparison, subgroup interpretation, outside-context analysis, and artifact sensitivity.

## Analysis Design

The analysis uses the final test images after manual mask-usability review. Images are grouped by the final `usable_for_overlap` label:

| Mask-usability label | Meaning | Role in this folder |
| --- | --- | --- |
| `yes` | Mask is suitable for strict lesion-overlap analysis. | Included in the primary result. |
| `cautious` | Mask is usable with caution, usually because localization or interpretation is less clean. | Added only in the sensitivity analysis. |
| `no` | Mask is not suitable for overlap claims. | Excluded from primary and sensitivity conclusions; reported only as context/limitation. |

For each scope, overlap metrics are summarized by model. The model ranking is computed within the same scope and metric.

## Analysis Scopes

| Scope | Rule | N | Role |
| --- | --- | ---: | --- |
| `main_yes` | `usable_for_overlap == yes` | 183 | Primary strict usable-mask overlap result. |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | Sensitivity result after adding cautious masks. |
| `excluded_no` | `usable_for_overlap == no` | 11 | Excluded/limitation cases only. |
| `all_images` | All final test images | 290 | Context only, not a primary denominator. |

The primary conclusion should be based on `main_yes`. The `sensitivity_yes_cautious` result is used to check whether that conclusion remains stable when the denominator is broadened.

## Metrics

The analysis uses seven compact overlap metrics:

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `energy_inside_mask` | Fraction of saliency energy inside the lesion mask. | Higher is better. |
| `normalized_energy` | Mask energy normalized against expected/background energy. | Higher is better. |
| `peak_in_mask` | Whether the saliency peak falls inside the lesion mask. | Higher is better. |
| `peak_distance_px` | Pixel distance from saliency peak to lesion mask/target. | Lower is better. |
| `mask_coverage_top20` | Fraction of lesion mask covered by the top 20% saliency region. | Higher is better. |
| `iou_top20` | IoU between lesion mask and top 20% saliency region. | Higher is better. |
| `dice_top20` | Dice score between lesion mask and top 20% saliency region. | Higher is better. |

## Results

The `results/` folder contains one guide file plus three CSV result tables.

| File | Meaning |
| --- | --- |
| `results/README.md` | Detailed guide for the CSV files, including reading order, column definitions, validity notes, and key numeric summaries. |
| `results/overlap_usable_scope_counts.csv` | Denominator and composition table for the analysis scopes. |
| `results/overlap_usable_scope_model_metric_summary.csv` | Core model-level metric summary by `scope x model x metric`. |
| `results/overlap_peak_distance_validity_by_scope.csv` | Valid/invalid peak-distance counts, especially for ResNet50 zero-saliency cases. |

Recommended reading order:

1. Read `results/README.md` for the guided explanation of the CSV outputs and compact numeric summaries.
2. Open `results/overlap_usable_scope_counts.csv` to confirm scope denominators.
3. Use `results/overlap_usable_scope_model_metric_summary.csv` for the main model comparison.
4. Check `results/overlap_peak_distance_validity_by_scope.csv` when interpreting `peak_distance_px`.

## Main Result Snapshot

In the strict primary set (`main_yes`, N = 183), ConvNeXt-Base ranks first on six of seven compact overlap metrics. ResNet50 is the closest comparator and is slightly better only on `peak_distance_px`, but that result should be read cautiously because 16 ResNet50 zero-saliency cases have invalid peak distance and are excluded from that mean (`n_valid = 167` out of 183).

When cautious masks are added (`sensitivity_yes_cautious`, N = 279), ConvNeXt-Base ranks first on all seven metrics. The broader sensitivity set therefore preserves the main ordering: ConvNeXt-Base strongest, ResNet50 closest, Swin-Base intermediate, and ViT-B/16 weakest.

For the metric-by-metric numeric tables, read `results/README.md`.

## Validity Notes

`peak_distance_px` requires a valid non-zero saliency map. ResNet50 has zero-saliency cases, so some ResNet50 peak-distance rows are invalid: 16 in `main_yes`, 29 in `sensitivity_yes_cautious`, and 29 in `all_images`. This validity issue is documented in `results/overlap_peak_distance_validity_by_scope.csv` and explained in `results/README.md`.

## Caveats

- This analysis evaluates saliency-mask spatial alignment, not classification accuracy.
- The primary result depends on manual mask-usability review. The strict `main_yes` denominator is the cleanest denominator for overlap claims.
- `excluded_no` and `all_images` are context scopes, not primary performance denominators.
- ResNet50 zero-saliency cases affect peak-distance validity.
- Saliency overlap should not be interpreted as causal proof of which pixels caused a model prediction.

## Conclusion

The primary and sensitivity analyses support the same main conclusion: ConvNeXt-Base has the strongest lesion-mask overlap overall. ResNet50 is the closest comparator, Swin-Base is intermediate, and ViT-B/16 attention rollout has the weakest lesion-mask overlap.

Adding cautious masks increases the denominator from 183 to 279 images and preserves this ordering. The result is therefore robust to the planned mask-QC sensitivity check.
