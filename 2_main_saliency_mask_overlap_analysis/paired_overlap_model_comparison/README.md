# Paired Overlap Model Comparison

This folder tests pairwise model differences for overlap metrics on the same images. The analysis is paired: each comparison keeps only images where both models have a valid metric value and computes per-image differences.

## Method

For each scope and metric, paired bootstrap resampling estimates the mean paired difference and confidence interval. Positive or negative direction depends on the metric definition; for distance metrics, lower values are better.

## Results

| Table | Meaning |
| --- | --- |
| `results/00_convnext_paired_key_ci_table.csv` | Compact key comparison table focused on ConvNeXt-Base. |
| `results/01_convnext_vs_others_paired_summary.csv` | Full ConvNeXt-vs-other paired summary by scope and metric. |
| `results/03_convnext_vs_others_per_image_long.csv` | Per-image paired difference audit table. |

The main result should be read for both `main_yes` and `sensitivity_yes_cautious`. ConvNeXt-Base is used as the primary reference because it is the strongest model in the main overlap summaries.

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

ConvNeXt-Base is consistently higher than ViT_B16 and Swin_Base across the paired overlap metrics. Its comparison with ResNet50 is closer in the strict `main_yes` set, but the broader yes+cautious sensitivity set still favors ConvNeXt for top-20% Dice.
