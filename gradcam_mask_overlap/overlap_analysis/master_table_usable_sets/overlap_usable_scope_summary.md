# Master-Table Overlap Analysis by Usable Set

Source table: `metadata/test_image_master_table.csv`.

Scopes:

- `main_yes`: `usable_for_overlap == yes`.
- `sensitivity_yes_cautious`: `usable_for_overlap in {yes, cautious}`.
- `excluded_no`: `usable_for_overlap == no`.
- `all_images`: full 290-image table for context only.

## Scope Counts

| Scope | N | Malignant | Benign | Yes | Cautious | No | Marker-positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_yes` | 183 | 107 | 76 | 183 | 0 | 0 | 18 |
| `sensitivity_yes_cautious` | 279 | 141 | 138 | 183 | 96 | 0 | 22 |
| `excluded_no` | 11 | 6 | 5 | 0 | 0 | 11 | 3 |
| `all_images` | 290 | 147 | 143 | 183 | 96 | 11 | 25 |

## Main Set Model Ranking

Ranking is by mean value; higher is better except `peak_distance_px`, where lower is better.

| Metric | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
| --- | --- | --- | --- | --- |
| `energy_inside_mask` | `ConvNeXt_Base` (0.754565) | `ResNet50` (0.671732) | `Swin_Base` (0.59346) | `ViT_B16` (0.391507) |
| `normalized_energy` | `ConvNeXt_Base` (1.40647) | `ResNet50` (1.26175) | `Swin_Base` (1.1069) | `ViT_B16` (0.725045) |
| `peak_in_mask` | `ConvNeXt_Base` (0.79235) | `ResNet50` (0.743169) | `Swin_Base` (0.606557) | `ViT_B16` (0.136612) |
| `peak_distance_px` | `ResNet50` (3.31036) | `ConvNeXt_Base` (3.35891) | `Swin_Base` (16.2454) | `ViT_B16` (38.3733) |
| `mask_coverage_top20` | `ConvNeXt_Base` (0.290899) | `ResNet50` (0.278347) | `Swin_Base` (0.200521) | `ViT_B16` (0.0799779) |
| `iou_top20` | `ConvNeXt_Base` (0.272865) | `ResNet50` (0.263546) | `Swin_Base` (0.1816) | `ViT_B16` (0.062587) |
| `dice_top20` | `ConvNeXt_Base` (0.421089) | `ResNet50` (0.401878) | `Swin_Base` (0.289835) | `ViT_B16` (0.115986) |

## Sensitivity Set Model Ranking

| Metric | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
| --- | --- | --- | --- | --- |
| `energy_inside_mask` | `ConvNeXt_Base` (0.744003) | `ResNet50` (0.636575) | `Swin_Base` (0.573827) | `ViT_B16` (0.379481) |
| `normalized_energy` | `ConvNeXt_Base` (1.41026) | `ResNet50` (1.21138) | `Swin_Base` (1.08565) | `ViT_B16` (0.709936) |
| `peak_in_mask` | `ConvNeXt_Base` (0.792115) | `ResNet50` (0.702509) | `Swin_Base` (0.53405) | `ViT_B16` (0.132616) |
| `peak_distance_px` | `ConvNeXt_Base` (3.21054) | `ResNet50` (3.96829) | `Swin_Base` (19.6059) | `ViT_B16` (40.7977) |
| `mask_coverage_top20` | `ConvNeXt_Base` (0.290894) | `ResNet50` (0.26215) | `Swin_Base` (0.193315) | `ViT_B16` (0.0739769) |
| `iou_top20` | `ConvNeXt_Base` (0.271164) | `ResNet50` (0.245369) | `Swin_Base` (0.172243) | `ViT_B16` (0.0574952) |
| `dice_top20` | `ConvNeXt_Base` (0.418542) | `ResNet50` (0.376964) | `Swin_Base` (0.277742) | `ViT_B16` (0.106934) |

## Peak Distance Validity

`peak_distance_px` is summarized only for rows with a valid saliency peak.

| Scope | Model | N images | Valid peak distance | Invalid | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| `main_yes` | `ResNet50` | 183 | 167 | 16 | Peak distance was computed only for valid non-zero saliency maps. |
| `main_yes` | `ViT_B16` | 183 | 183 | 0 |  |
| `main_yes` | `ConvNeXt_Base` | 183 | 183 | 0 |  |
| `main_yes` | `Swin_Base` | 183 | 183 | 0 |  |
| `sensitivity_yes_cautious` | `ResNet50` | 279 | 250 | 29 | Peak distance was computed only for valid non-zero saliency maps. |
| `sensitivity_yes_cautious` | `ViT_B16` | 279 | 279 | 0 |  |
| `sensitivity_yes_cautious` | `ConvNeXt_Base` | 279 | 279 | 0 |  |
| `sensitivity_yes_cautious` | `Swin_Base` | 279 | 279 | 0 |  |
| `excluded_no` | `ResNet50` | 11 | 11 | 0 |  |
| `excluded_no` | `ViT_B16` | 11 | 11 | 0 |  |
| `excluded_no` | `ConvNeXt_Base` | 11 | 11 | 0 |  |
| `excluded_no` | `Swin_Base` | 11 | 11 | 0 |  |
| `all_images` | `ResNet50` | 290 | 261 | 29 | Peak distance was computed only for valid non-zero saliency maps. |
| `all_images` | `ViT_B16` | 290 | 290 | 0 |  |
| `all_images` | `ConvNeXt_Base` | 290 | 290 | 0 |  |
| `all_images` | `Swin_Base` | 290 | 290 | 0 |  |
