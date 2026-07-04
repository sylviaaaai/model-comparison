# Outside / Context-Partition Saliency Energy Analysis

This folder asks where each model's saliency energy is concentrated: inside the lesion, in a perilesional ring, or outside the dilated lesion region.

## Region Definitions

The lesion region is the ROI mask. The perilesional region is a ring produced by dilating the mask by a fixed pixel radius and subtracting the original mask. The outside region is the remaining ROI area outside the dilated lesion. Pixel distance (`px`) is measured in the common 224 x 224 analysis space. The primary ring is 20 px; 10 px is used as a ring-size sensitivity check.

Zero-saliency rows have no meaningful regional energy distribution and are excluded from model-level means for these metrics. Very large dilated regions are flagged because they can shrink the outside region and make perilesional interpretation less stable.

## Results

| Table | Meaning |
| --- | --- |
| `results/00_context_partition_key_summary.csv` | Key four-condition summary: `main_yes` 20 px, `sensitivity_yes_cautious` 20 px, `main_yes` 10 px, and `sensitivity_yes_cautious` 10 px. |
| `results/01_context_partition_summary_by_model.csv` | Full model-level summary of lesion, perilesional, and outside saliency energy. |

## Main Result Snapshot

For the primary `main_yes`, 20 px ring setting:

| Model | Valid N / Images | Lesion energy mean | Outside normalized energy mean | Zero-saliency rows |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt_Base | 183 / 183 | 0.755 | 0.386 | 0 |
| ResNet50 | 167 / 183 | 0.736 | 0.491 | 16 |
| Swin_Base | 183 / 183 | 0.593 | 0.990 | 0 |
| ViT_B16 | 183 / 183 | 0.392 | 1.652 | 0 |

## Conclusion

ConvNeXt-Base and ResNet50 are most lesion-concentrated. ViT-B/16 has the greatest outside-region saliency concentration, consistent with a more spatially distributed attention profile. Swin-Base is intermediate. The same broad pattern is retained across the 20 px primary ring and 10 px sensitivity ring.
