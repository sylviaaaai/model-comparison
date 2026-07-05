# Outside / Context-Partition Saliency Energy Analysis

## Purpose

This supplementary analysis quantifies where each model's saliency energy is concentrated within the 224 x 224 ROI analysis space:

1. inside the lesion mask,
2. in adjacent perilesional tissue around the lesion, or
3. in the more distant outside/background region.

It complements the main lesion-mask overlap metrics. It should be interpreted as a spatial alignment / interpretability-pattern analysis, not as causal evidence about which pixels the model used for classification.

## Experimental Workflow

1. Start from the same final test images and lesion masks used in the main overlap analysis.
2. Use the same model-specific primary saliency maps:
   - ResNet50, ConvNeXt-Base, and Swin-Base: malignant-class Grad-CAM.
   - ViT-B/16: attention rollout.
3. Resize masks and saliency maps into the common 224 x 224 ROI analysis space.
4. Normalize each saliency map to non-negative 0-1 values.
5. Partition the ROI into lesion, perilesional, and outside regions.
6. Measure how much saliency energy falls in each region.
7. Repeat the analysis for two mask-usability scopes and two perilesional ring sizes:
   - `main_yes`, 20 px ring: primary setting.
   - `sensitivity_yes_cautious`, 20 px ring: mask-QC sensitivity setting.
   - `main_yes`, 10 px ring: ring-size sensitivity setting.
   - `sensitivity_yes_cautious`, 10 px ring: combined mask-QC and ring-size sensitivity setting.

## Region Definitions

For each ROI, the three regions are mutually exclusive and together cover the full 224 x 224 analysis space:

| Region | Definition | Interpretation |
| --- | --- | --- |
| `lesion` | Original binary lesion mask. | Saliency directly inside the annotated lesion. |
| `perilesional` | Dilated lesion mask minus the original lesion mask. | Saliency in the immediate tissue/context surrounding the lesion. |
| `outside` | ROI pixels outside the dilated lesion mask. | Saliency farther away from the lesion and perilesional ring. |

The dilated lesion mask is created from the original lesion mask using a circular structuring element and `scipy.ndimage.binary_dilation`. This avoids adding OpenCV as a dependency while preserving the intended disk/ellipse-style ring expansion.

Pixel distance (`px`) is measured in the common 224 x 224 analysis space. The primary ring is 20 px. A 10 px ring is included as a sensitivity check because a 20 px dilation can occupy a large fraction of the ROI for large lesions. The 10 px and 20 px settings are fixed-pixel outward dilations.

## Metric Definitions

The analysis reports two related metric families: raw regional energy fractions and area-normalized regional concentration scores.

### Energy Fractions

For non-zero saliency maps, total saliency energy is the sum of all non-negative normalized saliency values in the ROI. Regional energy is the fraction of that total energy falling in a region.

| Metric column | Meaning | How to read it |
| --- | --- | --- |
| `lesion_energy_mean` | Mean fraction of total saliency energy inside the lesion mask. | Higher means the model's saliency is more lesion-centered. |
| `perilesional_energy_mean` | Mean fraction of total saliency energy in the perilesional ring. | Higher means more saliency falls in tissue immediately around the lesion. |
| `outside_energy_mean` | Mean fraction of total saliency energy outside the dilated lesion region. | Higher means more saliency lies away from the lesion and immediate ring. |

For each valid image, `lesion_energy + perilesional_energy + outside_energy` is approximately 1. The three values are therefore best read as a partition of the saliency map, not as independent scores.

### Area-Normalized Concentration Scores

Area-normalized metrics divide the regional energy fraction by the regional area fraction:

```text
region_norm = region_energy_fraction / region_area_fraction
```

| Metric column | Meaning | How to read it |
| --- | --- | --- |
| `lesion_norm_mean` | Mean lesion energy after adjusting for lesion area. | Values above 1 mean saliency is more concentrated in the lesion than expected from area alone. |
| `perilesional_norm_mean` | Mean perilesional energy after adjusting for ring area. | Values above 1 mean disproportionate saliency concentration in the ring. |
| `outside_norm_mean` | Mean outside energy after adjusting for outside-region area. | Values above 1 mean disproportionate saliency concentration outside the lesion/ring. |

The normalized metrics are especially useful because the lesion, ring, and outside regions can have very different sizes. For example, a large `outside_energy_mean` can partly reflect the outside region being large; `outside_norm_mean` asks whether outside saliency is high relative to outside area.

### Count And Area Columns

| Column | Meaning |
| --- | --- |
| `n_images` | Number of image-model rows in that scope, model, and ring setting. |
| `n_valid` | Number with non-zero saliency and valid regional energy metrics. |
| `n_zero_saliency` | Number of zero-saliency maps excluded from model-level energy/norm means. |
| `n_large_dilated_region` | Number of rows where the dilated lesion covers more than 80% of the ROI. |
| `lesion_area_fraction_mean` | Mean fraction of the ROI occupied by the lesion mask. |
| `perilesional_area_fraction_mean` | Mean fraction of the ROI occupied by the perilesional ring. |
| `outside_area_fraction_mean` | Mean fraction of the ROI outside the dilated lesion. |
| `dilated_area_fraction_mean` | Mean fraction of the ROI occupied by lesion plus perilesional ring. |

Zero-saliency rows have no meaningful regional energy distribution and are excluded from model-level means for these metrics. Very large dilated regions are flagged because they can shrink the outside region and make perilesional/outside interpretation less stable.

## Results

| File | Meaning |
| --- | --- |
| `results/00_context_partition_key_summary.csv` | Key four-condition summary: `main_yes` 20 px, `sensitivity_yes_cautious` 20 px, `main_yes` 10 px, and `sensitivity_yes_cautious` 10 px. |
| `results/01_context_partition_summary_by_model.csv` | Full model-level summary of lesion, perilesional, and outside saliency energy. |

Recommended reading order:

1. Start with `results/00_context_partition_key_summary.csv` for the compact four-condition summary.
2. Use `results/01_context_partition_summary_by_model.csv` when you need medians, SDs, area fractions, and the full model-level detail.
3. Read the 20 px `main_yes` rows as the primary result, then check the 20 px `sensitivity_yes_cautious` rows for mask-QC robustness and the 10 px rows for ring-size robustness.

## Main Result Snapshot

For the primary `main_yes`, 20 px ring setting:

| Model | Valid N / Images | Lesion energy mean | Perilesional energy mean | Outside energy mean | Lesion norm mean | Perilesional norm mean | Outside norm mean | Zero-saliency rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt_Base | 183 / 183 | 0.755 | 0.151 | 0.094 | 1.406 | 0.628 | 0.386 | 0 |
| ResNet50 | 167 / 183 | 0.736 | 0.162 | 0.102 | 1.383 | 0.674 | 0.491 | 16 |
| Swin_Base | 183 / 183 | 0.593 | 0.199 | 0.207 | 1.107 | 0.822 | 0.990 | 0 |
| ViT_B16 | 183 / 183 | 0.392 | 0.266 | 0.343 | 0.725 | 1.092 | 1.652 | 0 |

Key reading:

- ConvNeXt_Base has the highest lesion-centered saliency in the primary setting by both `lesion_energy_mean` and `lesion_norm_mean`.
- ResNet50 is also strongly lesion-centered, but its summary excludes 16 zero-saliency rows from the regional energy/norm means.
- ViT_B16 has the lowest lesion energy and the highest outside energy/outside norm, consistent with a more spatially distributed attention pattern.
- Swin_Base is intermediate: less lesion-centered than ConvNeXt_Base/ResNet50, but less outside-concentrated than ViT_B16.
- Perilesional values should be interpreted together with ring size. The 20 px ring produces many large dilated regions, so the 10 px sensitivity rows are useful for checking whether the pattern depends on the chosen ring width.

## Conclusion

ConvNeXt-Base and ResNet50 are most lesion-concentrated. ViT-B/16 has the greatest outside-region saliency concentration, consistent with a more spatially distributed attention profile. Swin-Base is intermediate. The same broad pattern is retained across the 20 px primary ring and 10 px sensitivity ring.
