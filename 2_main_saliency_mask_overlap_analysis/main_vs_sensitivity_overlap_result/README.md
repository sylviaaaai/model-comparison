# Main vs Sensitivity Overlap Result

This folder compares the primary strict usable-mask result with a broader mask-QC sensitivity result.

## Analysis Scopes

| Scope | Rule | N | Role |
| --- | --- | ---: | --- |
| `main_yes` | `usable_for_overlap == yes` | 183 | Primary overlap result |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | Mask-QC sensitivity result |
| `excluded_no` | `usable_for_overlap == no` | 11 | Excluded / limitation cases only |
| `all_images` | all final test images | 290 | Context only, not a primary overlap denominator |

## Result Logic

For each scope, the analysis summarizes seven compact overlap metrics by model: energy inside mask, normalized energy, peak-in-mask, peak distance in pixels, top-20% mask coverage, top-20% IoU, and top-20% Dice. Higher is better for all metrics except `peak_distance_px`, where lower is better. Model ranking is computed within each scope and metric.

## Results

| Table | Meaning |
| --- | --- |
| `results/overlap_usable_scope_counts.csv` | Denominators for main, sensitivity, excluded, and all-image context. |
| `results/overlap_usable_scope_model_metric_summary.csv` | Mean, median, SD, min, and max for each scope x model x metric. |
| `results/overlap_peak_distance_validity_by_scope.csv` | Valid/invalid peak-distance counts, especially for ResNet50 zero-saliency cases. |

## Conclusion

In the primary `main_yes` set, ConvNeXt-Base ranks first for six of the seven compact overlap metrics. ResNet50 is the closest comparator, Swin-Base is intermediate, and ViT-B/16 attention rollout has the weakest lesion-mask overlap. Adding cautious masks increases the denominator from 183 to 279 images and preserves the main ordering: ConvNeXt-Base remains the top overlap model.
