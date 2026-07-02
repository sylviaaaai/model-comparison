# ResNet50 Grad-CAM Verification

Generated during ResNet50 raw Grad-CAM export QC.

## Prediction Probability Check

The ResNet50 final model was reloaded from:

```text
results/resnet50_final_conv3_5_preprocess_dropout04/final/conv3_5/final_model.keras
```

Predictions were recomputed on all 290 final test ROI images using the same
dataset function as the original prediction export:

```text
decode image -> resize to 224 x 224 with bilinear interpolation -> cast to float32
```

The recomputed probabilities were compared against:

```text
results/patient_level_bootstrap_ci/resnet50_test_predictions.csv
```

The first direct final-model prediction check gave:

```text
n_model = 290
n_csv = 290
max_abs_diff = 0.0
mean_abs_diff = 0.0
allclose_1e-7 = True
```

Therefore, the Grad-CAM export path uses prediction probabilities that are
identical to the existing ResNet50 prediction CSV.

After adding probability logging to the Grad-CAM manifest, an important
implementation issue was detected: the manually reconstructed Grad-CAM model
initially skipped the internal `keras.applications.resnet50.preprocess_input`
operation because that TensorFlow op is not listed as a normal Keras layer after
reloading the saved Functional model. The Grad-CAM export script was fixed to
explicitly apply `preprocess_input` when reconstructing the path to the nested
ResNet50 backbone.

After the fix, all ResNet50 Grad-CAM maps were regenerated. The manifest
probability check now gives:

```text
n = 290
max_abs_diff(recomputed_cancer_probability, model_predictions.cancer_probability) = 4.732208251922998e-07
mean_abs_diff = 8.266626384268756e-08
rows_with_diff_gt_1e-6 = 0
```

This confirms that the Grad-CAM model path now matches the final prediction
path to within float formatting precision.

## Preprocessing Check

The final training result records:

```text
use_resnet_preprocess_input = True
```

The training/prediction dataset function performs only:

```text
tf.io.read_file
tf.image.decode_image(..., channels=3)
tf.image.resize(..., (224, 224), method="bilinear")
tf.cast(..., tf.float32)
```

The saved Keras model then contains the ResNet preprocessing inside the model
graph before the ResNet50 backbone. There is no separate `rescaling` layer in
the loaded final model. The loaded top-level model begins with:

```text
input_layer_1
data_augmentation
resnet50
global_average_pooling
dropout
cancer_probability
```

`data_augmentation` is present in the model but is inactive during inference
because Grad-CAM export and prediction checks call the model with
`training=False`.

The raw Grad-CAM export script uses the same external image preprocessing:

```text
decode image -> resize to 224 x 224 with bilinear interpolation -> cast to float32
```

When reconstructing the Grad-CAM model, the script explicitly adds the saved
model's ResNet preprocessing step before the augmentation/backbone path:

```text
keras.applications.resnet50.preprocess_input(inputs)
```

This is required because the preprocessing op is inside the saved Functional
graph but is not exposed as a normal layer in `model.layers`. Therefore,
preprocessing is now consistent with the final prediction export and training
configuration.

## Manifest Correction

During verification, old ResNet50 rows in `saliency_manifest.csv` were found to
have stale `target_class_value` values from prior visualization metadata. The
raw Grad-CAM maps had been exported using the current `predicted_label`, but the
manifest field was updated afterward for consistency.

Result:

```text
updated target_class_value fields = 54
remaining ResNet50 target/prediction mismatches = 0
```

All ResNet50 saliency rows now have:

```text
target_class_for_cam = predicted_label
target_class_value = model_predictions.predicted_label
status = raw_map_available
```
