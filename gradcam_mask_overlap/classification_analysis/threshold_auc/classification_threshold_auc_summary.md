# Classification Threshold and Paired AUC Analysis

## Design

- Thresholds were selected only on the final validation split.
- The selected thresholds were then locked and applied to the final test split.
- Test-set AUC CIs and paired AUC differences used patient-level bootstrap resampling.
- Classification analysis uses all final test images and is separate from the mask-overlap usable set.

## Validation-Selected Thresholds

| Model | Threshold | Val AUC | Val sensitivity | Val specificity | Val F1 | Val TP/TN/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet50 | 0.6070 | 0.774 | 0.658 | 0.806 | 0.716 | 96/108/26/50 |
| ViT-B/16 | 0.5581 | 0.834 | 0.678 | 0.888 | 0.762 | 99/119/15/47 |
| ConvNeXt-Base | 0.6931 | 0.812 | 0.651 | 0.896 | 0.745 | 95/120/14/51 |
| Swin-Base | 0.4614 | 0.803 | 0.685 | 0.836 | 0.746 | 100/112/22/46 |

## Locked Test Performance

| Model | Threshold | Test AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | Test TP/TN/FP/FN |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| ResNet50 | 0.6070 | 0.701 [0.623, 0.777] | 0.655 [0.590, 0.724] | 0.544 [0.442, 0.645] | 0.769 [0.686, 0.851] | 0.615 [0.522, 0.698] | 80/110/33/67 |
| ViT-B/16 | 0.5581 | 0.819 [0.758, 0.877] | 0.779 [0.718, 0.837] | 0.667 [0.567, 0.761] | 0.895 [0.841, 0.942] | 0.754 [0.675, 0.820] | 98/128/15/49 |
| ConvNeXt-Base | 0.6931 | 0.791 [0.734, 0.845] | 0.759 [0.702, 0.813] | 0.633 [0.546, 0.719] | 0.888 [0.833, 0.939] | 0.727 [0.655, 0.790] | 93/127/16/54 |
| Swin-Base | 0.4614 | 0.781 [0.717, 0.846] | 0.752 [0.689, 0.812] | 0.619 [0.523, 0.717] | 0.888 [0.827, 0.943] | 0.717 [0.635, 0.789] | 91/127/16/56 |

## Fixed 0.5 Comparator

| Model | Test AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | Test TP/TN/FP/FN |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet50 | 0.701 [0.623, 0.777] | 0.624 [0.559, 0.691] | 0.605 [0.503, 0.704] | 0.643 [0.554, 0.731] | 0.620 [0.532, 0.699] | 89/92/51/58 |
| ViT-B/16 | 0.819 [0.758, 0.877] | 0.776 [0.716, 0.832] | 0.680 [0.583, 0.772] | 0.874 [0.818, 0.925] | 0.755 [0.678, 0.819] | 100/125/18/47 |
| ConvNeXt-Base | 0.791 [0.734, 0.845] | 0.728 [0.673, 0.782] | 0.701 [0.617, 0.781] | 0.755 [0.685, 0.825] | 0.723 [0.653, 0.785] | 103/108/35/44 |
| Swin-Base | 0.781 [0.717, 0.846] | 0.762 [0.697, 0.824] | 0.612 [0.514, 0.711] | 0.916 [0.859, 0.965] | 0.723 [0.640, 0.796] | 90/131/12/57 |

## Paired Test AUC

| Model | Test AUC (95% CI) |
| --- | --- |
| ResNet50 | 0.701 [0.623, 0.777] |
| ViT-B/16 | 0.819 [0.758, 0.877] |
| ConvNeXt-Base | 0.791 [0.734, 0.845] |
| Swin-Base | 0.781 [0.717, 0.846] |

Pairwise paired-bootstrap AUC differences are in:

```text
/mnt/e/SW整合/gradcam_mask_overlap/classification_analysis/threshold_auc/classification_test_auc_pairwise_bootstrap.csv
```

Positive pairwise deltas favor `model_a`.

## Notes

- Validation CSV filenames retain the source script's `_test_predictions.csv` suffix, but the directory and split column identify them as validation predictions.
- The ResNet50 validation predictions were exported with TensorFlow CPU; TensorFlow reported no available GPU devices.
