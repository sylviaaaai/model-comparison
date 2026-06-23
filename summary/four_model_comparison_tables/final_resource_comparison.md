| Model | Total Params | Trainable Params | Non-Trainable Params | Model Size (MB) | Training Time (s) | Avg Epoch Time (s) | Test Inference Time (s) | Test ms/step | Epochs Ran | Batch Size | Setting | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50-conv3-5 (preprocess, dropout=0.4) | 23589761 | 23259649 | 330112 | 268.15 | 109.11 | 5.46 | 1.41 | 74.23 | 20 | 16 | conv3_5 | 
| ViT-B/16-All blocks (dropout=0.4) | 85799425 | 85055233 | 744192 | 327.36 | 60.77 | 3.38 | 2.65 | 139.26 | 18 | 16 | all_blocks |
| ConvNeXt-Base-Stage 3-4 (dropout=0.4) | 87567489 | 85362689 | 2204800 | 334.17 | 81.53 | 3.54 | 1.19 | 118.54 | 23 | 32 | stage3_4 |
| Swin-Base-Stage 2-4 (dropout=0.4, bs16) | 86744249 | 86283505 | 460744 | 331.03 | 80.34 | 5.02 | 0.53 | 27.74 | 16 | 16 | stage2_4 | 
