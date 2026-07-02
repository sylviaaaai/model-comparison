# Paired Overlap Model Comparison

Reference model: `ConvNeXt_Base`.

Effect-delta convention:

```text
For higher-is-better metrics:
  effect_delta = ConvNeXt_Base - comparator

For peak_distance_px:
  effect_delta = comparator - ConvNeXt_Base

Therefore, positive effect_delta always favors ConvNeXt_Base.
```

Bootstrap mean-delta CI: 5000 resamples, seed = 20260701.

## main_yes

| Metric | Comparator | N pairs | Mean effect delta | 95% bootstrap CI | ConvNeXt better rate | Note |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `energy_inside_mask` | `ResNet50` | 183 | 0.0828325 | 0.0399988 to 0.125735 | 0.606557 |  |
| `energy_inside_mask` | `Swin_Base` | 183 | 0.161105 | 0.133064 to 0.189261 | 0.803279 |  |
| `energy_inside_mask` | `ViT_B16` | 183 | 0.363058 | 0.337352 to 0.387519 | 0.939891 |  |
| `normalized_energy` | `ResNet50` | 183 | 0.144722 | 0.068457 to 0.2233 | 0.606557 |  |
| `normalized_energy` | `Swin_Base` | 183 | 0.299575 | 0.242232 to 0.357694 | 0.803279 |  |
| `normalized_energy` | `ViT_B16` | 183 | 0.681427 | 0.630213 to 0.731734 | 0.939891 |  |
| `peak_in_mask` | `ResNet50` | 183 | 0.0491803 | -0.0327869 to 0.125683 | 0.169399 |  |
| `peak_in_mask` | `Swin_Base` | 183 | 0.185792 | 0.103825 to 0.26776 | 0.273224 |  |
| `peak_in_mask` | `ViT_B16` | 183 | 0.655738 | 0.579235 to 0.73224 | 0.677596 |  |
| `peak_distance_px` | `ResNet50` | 167 | 0.126152 | -1.53271 to 1.78387 | 0.161677 | Paired comparison uses rows where both models have finite metric values. |
| `peak_distance_px` | `Swin_Base` | 183 | 12.8865 | 9.52615 to 16.4013 | 0.387978 |  |
| `peak_distance_px` | `ViT_B16` | 183 | 35.0144 | 31.0106 to 39.052 | 0.836066 |  |
| `mask_coverage_top20` | `ResNet50` | 183 | 0.0125526 | -0.00308359 to 0.029037 | 0.497268 |  |
| `mask_coverage_top20` | `Swin_Base` | 183 | 0.0903784 | 0.0747895 to 0.106108 | 0.797814 |  |
| `mask_coverage_top20` | `ViT_B16` | 183 | 0.210922 | 0.199354 to 0.222317 | 0.994536 |  |
| `iou_top20` | `ResNet50` | 183 | 0.00931918 | -0.0073673 to 0.0262583 | 0.497268 |  |
| `iou_top20` | `Swin_Base` | 183 | 0.0912648 | 0.0752266 to 0.107555 | 0.797814 |  |
| `iou_top20` | `ViT_B16` | 183 | 0.210278 | 0.198006 to 0.222355 | 0.994536 |  |
| `dice_top20` | `ResNet50` | 183 | 0.0192116 | -0.00345712 to 0.0428307 | 0.497268 |  |
| `dice_top20` | `Swin_Base` | 183 | 0.131255 | 0.108425 to 0.154268 | 0.797814 |  |
| `dice_top20` | `ViT_B16` | 183 | 0.305103 | 0.289031 to 0.321306 | 0.994536 |  |

## sensitivity_yes_cautious

| Metric | Comparator | N pairs | Mean effect delta | 95% bootstrap CI | ConvNeXt better rate | Note |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `energy_inside_mask` | `ResNet50` | 279 | 0.107427 | 0.0726008 to 0.14349 | 0.609319 |  |
| `energy_inside_mask` | `Swin_Base` | 279 | 0.170175 | 0.146919 to 0.193755 | 0.817204 |  |
| `energy_inside_mask` | `ViT_B16` | 279 | 0.364522 | 0.343282 to 0.385217 | 0.953405 |  |
| `normalized_energy` | `ResNet50` | 279 | 0.198885 | 0.129077 to 0.268386 | 0.609319 |  |
| `normalized_energy` | `Swin_Base` | 279 | 0.324613 | 0.278128 to 0.372348 | 0.817204 |  |
| `normalized_energy` | `ViT_B16` | 279 | 0.700329 | 0.65662 to 0.745412 | 0.953405 |  |
| `peak_in_mask` | `ResNet50` | 279 | 0.0896057 | 0.0250896 to 0.154122 | 0.197133 |  |
| `peak_in_mask` | `Swin_Base` | 279 | 0.258065 | 0.189964 to 0.326165 | 0.333333 |  |
| `peak_in_mask` | `ViT_B16` | 279 | 0.659498 | 0.598566 to 0.716846 | 0.681004 |  |
| `peak_distance_px` | `ResNet50` | 250 | 0.827168 | -0.685047 to 2.33957 | 0.184 | Paired comparison uses rows where both models have finite metric values. |
| `peak_distance_px` | `Swin_Base` | 279 | 16.3953 | 13.4962 to 19.536 | 0.448029 |  |
| `peak_distance_px` | `ViT_B16` | 279 | 37.5871 | 34.2718 to 40.9871 | 0.845878 |  |
| `mask_coverage_top20` | `ResNet50` | 279 | 0.0287442 | 0.0147894 to 0.0432344 | 0.544803 |  |
| `mask_coverage_top20` | `Swin_Base` | 279 | 0.0975793 | 0.0843974 to 0.110831 | 0.810036 |  |
| `mask_coverage_top20` | `ViT_B16` | 279 | 0.216917 | 0.206594 to 0.227128 | 0.992832 |  |
| `iou_top20` | `ResNet50` | 279 | 0.0257954 | 0.0115716 to 0.0403075 | 0.544803 |  |
| `iou_top20` | `Swin_Base` | 279 | 0.0989216 | 0.0858993 to 0.112248 | 0.810036 |  |
| `iou_top20` | `ViT_B16` | 279 | 0.213669 | 0.203019 to 0.224026 | 0.992832 |  |
| `dice_top20` | `ResNet50` | 279 | 0.0415775 | 0.021553 to 0.061808 | 0.544803 |  |
| `dice_top20` | `Swin_Base` | 279 | 0.140799 | 0.122293 to 0.159168 | 0.810036 |  |
| `dice_top20` | `ViT_B16` | 279 | 0.311608 | 0.297315 to 0.325046 | 0.992832 |  |

## Interpretation Notes

- Positive effect deltas favor ConvNeXt_Base for every metric.
- `peak_distance_px` uses the same positive-favors-ConvNeXt convention by reversing the raw delta.
- Rows with invalid metric values are excluded pairwise. This mainly affects ResNet50 `peak_distance_px` because of zero-saliency maps.
- Use the paired rows to support statements about consistency across the same image set, not just model-level mean ranking.
