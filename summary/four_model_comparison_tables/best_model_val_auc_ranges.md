| Model | Final Model | Fine-tuning Setting | Validation AUC (mean+-SD) | Test AUC | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 | Batch Size | Pretraining |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50 | ResNet50-conv3-5 (preprocess, dropout=0.4) | conv3_5 | 0.7337+-0.0356 | 0.7016 | 0.6241 | 0.6054 | 0.6434 | 0.6357 | 0.6202 | 16 | ImageNet-1K |
| ViT-B/16 | ViT-B/16-All blocks (dropout=0.4) | all_blocks | 0.7803+-0.0286 | 0.8190 | 0.7759 | 0.6803 | 0.8741 | 0.8475 | 0.7547 | 16 | ImageNet-21K -> 1K |
| ConvNeXt-Base | ConvNeXt-Base-Stage 3-4 (dropout=0.4, bs32) | stage3_4 | 0.7883+-0.0309 | 0.7910 | 0.7276 | 0.7007 | 0.7552 | 0.7464 | 0.7228 | 32 | ImageNet-22K -> 1K |
| Swin-Base | Swin-Base-Stage 2-4 (dropout=0.4, bs16) | stage2_4 | 0.7846+-0.0373 | 0.7813 | 0.7621 | 0.6122 | 0.9161 | 0.8824 | 0.7229 | 16 | ImageNet-22K -> 1K |
