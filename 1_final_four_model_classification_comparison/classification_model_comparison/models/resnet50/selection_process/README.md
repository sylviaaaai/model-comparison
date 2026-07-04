# ResNet50 Selection Process

This folder records the ResNet50 model-selection path used before the final test-set evaluation.

## Process Stages

1. `initial_no_preprocess_baseline/`

   Initial 5-fold ResNet50 baseline before enabling the ResNet-specific input preprocessing. This stage used the same patient-level folds and the same unfreezing strategy search, but the input normalization was only simple pixel rescaling.

   In the code, the no-preprocess baseline used:

   ```python
   keras.layers.Rescaling(1.0 / 255)
   ```

   This converts raw image values from approximately `[0, 255]` to `[0, 1]`. It is a common generic image normalization, but it does not match the input distribution expected by the ImageNet-pretrained Keras ResNet50 backbone.

   The no-preprocess run had lower validation AUCs. For example, the best mean validation AUC in this stage was around `0.6631` (`conv2_5`), while the later ResNet-preprocessed run improved the best mean validation AUC to around `0.7288` (`conv3_5`).

2. `preprocess_all_strategies/`

   Baseline preprocessed 5-fold comparison across unfreezing strategies:

   - `head_only`
   - `conv2_5`
   - `conv3_5`
   - `conv4_5`
   - `conv5`
   - `all_non_bn`

3. `preprocess_dropout03/`

   Follow-up 5-fold experiment using dropout 0.3 after narrowing the candidate settings. This stage includes `all_non_bn` and `conv3_5`.

4. `preprocess_dropout04_final_check/`

   Final 5-fold validation check for `conv3_5` with dropout 0.4. This setting was selected for the final ResNet50 test evaluation.

The original top-level files in this folder also correspond to the dropout 0.4 final-check run and are kept for backward compatibility with the first repository organization.

## What `preprocess` Means Here

In the ResNet50 experiments, `preprocess` specifically means enabling Keras' ResNet50 input preprocessing:

```python
keras.applications.resnet50.preprocess_input(inputs)
```

This replaced the earlier generic `Rescaling(1.0 / 255)` input layer.

The practical difference is:

- The initial baseline scaled image pixels to `[0, 1]`.
- The preprocessed runs used the same convention as Keras' ImageNet-pretrained ResNet50 model.
- This preprocessing converts RGB input into the ResNet50 expected channel/value format, including the ImageNet-style mean subtraction used by the pretrained backbone.

This was important because the ResNet50 backbone was initialized with ImageNet weights. Using the matching preprocessing made the mammography ROI inputs closer to the distribution expected by the pretrained feature extractor, and the validation AUC improved clearly during the unfreeze-strategy comparison.

The training-time data augmentation was separate from this preprocessing step. Augmentation remained inside the model and included random horizontal flip, rotation, zoom, translation, and contrast adjustment.

## Final ResNet50 Setting

- Architecture: ResNet50
- Fine-tuning setting: `conv3_5`
- Preprocessing: enabled
- Dropout: 0.4
- Batch size: 16
- Final result folder: `../final_results/`
