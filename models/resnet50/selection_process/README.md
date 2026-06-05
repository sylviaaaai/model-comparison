# ResNet50 Selection Process

This folder records the ResNet50 model-selection path used before the final test-set evaluation.

## Process Stages

1. `preprocess_all_strategies/`

   Baseline preprocessed 5-fold comparison across unfreezing strategies:

   - `head_only`
   - `conv2_5`
   - `conv3_5`
   - `conv4_5`
   - `conv5`
   - `all_non_bn`

2. `preprocess_dropout03/`

   Follow-up 5-fold experiment using dropout 0.3 after narrowing the candidate settings. This stage includes `all_non_bn` and `conv3_5`.

3. `preprocess_dropout04_final_check/`

   Final 5-fold validation check for `conv3_5` with dropout 0.4. This setting was selected for the final ResNet50 test evaluation.

The original top-level files in this folder also correspond to the dropout 0.4 final-check run and are kept for backward compatibility with the first repository organization.

## Final ResNet50 Setting

- Architecture: ResNet50
- Fine-tuning setting: `conv3_5`
- Preprocessing: enabled
- Dropout: 0.4
- Batch size: 16
- Final result folder: `../final_results/`
