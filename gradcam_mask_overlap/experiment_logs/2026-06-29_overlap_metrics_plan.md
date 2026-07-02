# Plan: Lesion-Mask Overlap Metrics

Date: 2026-06-29
Workspace: `/mnt/e/SW整合`

## Execution Environment

Host/runtime:

```text
OS/runtime: WSL2 Linux on DESKTOP-3CQ0U76
Kernel: 6.6.114.1-microsoft-standard-WSL2
Workspace: /mnt/e/SW整合
Timezone at planning time: Asia/Shanghai
```

Python environments observed:

```text
system python: Python 3.14.4
torch environment: .venv-torch5090/bin/python = Python 3.11.15
```

Primary ML/runtime packages in `.venv-torch5090`:

```text
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
CUDA build: 12.8
timm: 1.0.27
numpy: 2.4.4
Pillow: 12.2.0
```

GPU availability:

```text
GPU: NVIDIA GeForce RTX 5090
Driver: 596.49
Memory: 32607 MiB
```

Important operational note:

```text
Inside the default sandbox, GPU/NVML access is blocked, so torch reports
cuda_available=False and nvidia-smi fails with "GPU access blocked by the
operating system".

GPU is usable only after requesting elevated execution. Full Swin and ViT
prediction/saliency recomputation jobs were run with elevated GPU access, where
the scripts printed cuda / NVIDIA GeForce RTX 5090.
```

The overlap metrics stage itself should be lightweight enough to run on CPU
because it reads precomputed `.npy` saliency maps and binary masks. GPU should
only be needed again if saliency maps or model predictions are regenerated.

## Current Starting Point

The prediction and saliency alignment audit is complete. The overlap stage can
start from:

```text
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
```

Current readiness:

```text
analysis_manifest_rows = 2030
ready_for_overlap_rows = 2030
saliency_status_counts = {"raw_map_available": 2030}
```

Primary analysis methods:

| Model | Primary saliency method | Explanation family | Class-specific | Rows |
| --- | --- | --- | --- | ---: |
| ResNet50 | `gradcam_malignant` | Grad-CAM | True | 290 |
| ConvNeXt-Base | `gradcam_malignant` | Grad-CAM | True | 290 |
| Swin-Base | `gradcam_malignant` | Grad-CAM | True | 290 |
| ViT-B/16 | `attention_rollout` | Attention rollout | False | 290 |

Predicted-class Grad-CAM is available for ResNet50, ConvNeXt, and Swin, but it
should be treated as supplementary/sensitivity analysis rather than the primary
four-model comparison.

## Overall Order

The next stage should proceed in this order:

1. Overlay smoke QC
2. Overlap metric script and smoke run
3. Full primary overlap run
4. Integrity checks
5. Main model-level summary
6. Optional threshold/top-k/Otsu sensitivity analyses
7. Grouped summaries
8. Figures and representative cases

This order keeps file matching and image alignment checks ahead of any numerical
interpretation.

## Stage 1: Overlay Smoke QC

Goal: verify that ROI masks and saliency maps are spatially aligned.

Use only the primary methods:

```text
resnet50:gradcam_malignant
convnext_base:gradcam_malignant
swin_base:gradcam_malignant
vit_b16:attention_rollout
```

Sampling plan:

```text
sample image_id-level cases first:
  TP: 3 cases
  TN: 3 cases
  FP: 3 cases
  FN: 3 cases

for each sampled image_id:
  render all four primary model-method maps side by side
```

The preferred QC layout is case-centric rather than model-random. Showing the
same image_id across ResNet50, ConvNeXt, Swin, and ViT makes it easier to catch
model-specific flips, path mismatches, mask shifts, and ViT resize artifacts.

Recommended QC panel contents:

```text
ROI image
lesion mask
raw heatmap
heatmap overlay
top-10% binary saliency
mask + binary saliency
```

Suggested output:

```text
gradcam_mask_overlap/qc_figures/overlap_check/
  sampled_cases_index.csv
  TP/
  TN/
  FP/
  FN/
```

Stop criteria:

- mask appears outside the ROI lesion/finding
- mask/heatmap/image are mismatched by image_id
- heatmap appears flipped, rotated, or shifted
- saliency or mask shape is not 224 x 224
- mask is empty
- saliency map is all zero or contains NaN

## Stage 2: Overlap Metrics Script

Create:

```text
gradcam_mask_overlap/tools/compute_overlap_metrics.py
```

Input:

```text
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
```

Default behavior:

- run only the primary four model-method pairs
- write output to `gradcam_mask_overlap/metadata/overlap_metrics.csv`
- write an integrity summary JSON
- include `explanation_family` and `class_specific` columns in output tables

Suggested output summary:

```text
gradcam_mask_overlap/metadata/overlap_metrics_summary.json
```

## Stage 3: Continuous Metrics

These metrics do not require thresholding and should be part of every output
row or written once per image-method pair.

Metrics:

```text
cam_energy_in_mask
normalized_energy_in_mask
peak_in_mask
peak_x
peak_y
peak_value
peak_distance_to_mask_px
mask_area_px
mask_area_fraction
cam_sum
```

Definitions:

```text
cam_energy_in_mask = heatmap[mask == 1].sum() / (heatmap.sum() + 1e-8)
normalized_energy_in_mask = cam_energy_in_mask / (mask_area_fraction + 1e-8)
peak_in_mask = mask[argmax(heatmap)] == 1
```

`normalized_energy_in_mask` is a primary metric. A value greater than 1 means
the saliency energy is more concentrated inside the annotation region than
expected from a spatially uniform heatmap. This helps correct for the fact that
large masks naturally capture more heatmap energy.

Continuous output should cover:

```text
290 images x 4 primary methods = 1160 image-method pairs
```

## Stage 4: Thresholded Metrics

For each binary saliency map, compute:

```text
iou
dice
mask_coverage_by_cam
saliency_precision
cam_area_px
cam_area_fraction
intersection_area_px
union_area_px
```

Metric meanings:

```text
mask_coverage_by_cam = intersection / mask_area
saliency_precision = intersection / cam_area
```

### Fixed Threshold Sweep

Candidate fixed thresholds:

```text
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
```

Expected primary rows:

```text
290 images x 4 methods x 9 thresholds = 10440 rows
```

Use this for threshold curves and best-threshold analysis.

Best fixed threshold should be treated as descriptive/sensitivity analysis, not
as the main conclusion. Selecting a threshold by mean IoU on the test set has a
small test-set tuning flavor. The main result should prioritize continuous
metrics and fixed-area top-k results.

### Top-K / Top-X Thresholds

Candidate top-k settings to discuss:

```text
top 10%
top 20%
top 30%
top 40%
```

Expected primary rows if all four are used:

```text
290 images x 4 methods x 4 top-k settings = 4640 rows
```

Reason to include:

- controls salient area across models
- often fairer than a fixed raw heatmap threshold because Grad-CAM and
  attention rollout may have different intensity distributions

Question to decide:

```text
Should top-k be included in the first full run, or only after fixed-threshold
results pass QC?
```

### Otsu Threshold

Otsu is useful as a supplementary/sensitivity analysis, but not ideal as the
main result because each image gets its own adaptive threshold.

Question to decide:

```text
Should Otsu be implemented in the initial script as an optional threshold_type,
or deferred until after fixed/top-k results are reviewed?
```

If included, record:

```text
otsu_threshold
iou
dice
mask_coverage_by_cam
saliency_precision
cam_area_fraction
```

## Stage 5: Integrity Checks

After a smoke or full run, verify:

- expected row counts
- each `image_id x model_name x saliency_method` primary pair is unique
- no NaN or infinite metric values
- all metrics are in [0, 1] where applicable
- `peak_in_mask` is boolean
- `mask_area_px > 0`
- `cam_sum > 0`
- all saliency maps and masks have shape 224 x 224
- heatmap min/max are within 0-1
- masks are binary after loading
- ViT primary rows use `attention_rollout`, not `attention_rollout_magma_discard09`
- the same image_id across all four primary methods points to the same mask_path
- fixed threshold area decreases as threshold increases
- top-k area fraction is close to the requested k
- top-k area fraction may deviate slightly when many pixels share identical
  heatmap values; record the realized area fraction rather than forcing it

Suggested summary file:

```text
gradcam_mask_overlap/metadata/overlap_metrics_summary.json
```

## Stage 6: Main Summary Tables

After full metrics pass integrity checks, generate:

```text
gradcam_mask_overlap/results/overlap_main_summary_by_model.csv
```

One row per primary model-method pair:

```text
model_name
saliency_method
explanation_family
class_specific
n_images
mean_cam_energy_in_mask
mean_normalized_energy_in_mask
hit_rate
top10_mean_iou
top10_mean_dice
top20_mean_iou
top20_mean_dice
top20_mean_saliency_precision
top20_mean_mask_coverage
best_fixed_threshold_by_mean_iou
best_threshold_mean_iou
best_threshold_mean_dice
best_threshold_mean_mask_coverage
best_threshold_mean_saliency_precision
```

Important wording:

- call fixed-threshold selected results `best-threshold IoU`
- report best-threshold results as exploratory/descriptive or supplementary
- do not call per-image oracle threshold results simply `IoU`
- do not describe ViT attention rollout as class-specific Grad-CAM
- for benign images, describe masks as annotated abnormality/finding regions,
  not malignant lesion regions
- for FP images, the relevant question is whether malignant-class saliency
  overlaps the benign abnormality or nearby dense tissue/finding region
- for TN images, be cautious: normalized CAMs always have a peak, so
  `peak_in_mask` does not necessarily mean strong malignant evidence

## Stage 7: Grouped Summaries

Only after the main model-level result is stable, generate:

```text
gradcam_mask_overlap/results/overlap_summary_by_true_label.csv
gradcam_mask_overlap/results/overlap_summary_by_confusion_group.csv
gradcam_mask_overlap/results/overlap_summary_by_mask_type.csv
```

Grouping dimensions:

```text
true_label / true_class
prediction_group: TP, TN, FP, FN
manual mask type if available
```

Interpretation questions:

- do TP cases have higher energy/hit rate than FN cases?
- do malignant cases have higher overlap than benign cases?
- do false positives highlight benign findings or non-mask regions?
- do mass-like and calcification-like masks behave differently?

Benign-case interpretation note:

For benign samples, the mask represents a benign abnormality annotation, not a
malignant lesion. For Grad-CAM malignant maps, phrase benign overlap results as
"malignant-class saliency overlapping the annotated abnormality region" rather
than "attention to malignant lesion."

## Stage 8: Figures and Representative Cases

Threshold curves:

```text
gradcam_mask_overlap/results/figures/threshold_vs_iou.png
gradcam_mask_overlap/results/figures/threshold_vs_dice.png
gradcam_mask_overlap/results/figures/threshold_vs_coverage.png
gradcam_mask_overlap/results/figures/threshold_vs_precision.png
```

Representative cases:

```text
gradcam_mask_overlap/results/figures/representative_cases/
```

Each case should show:

```text
ROI image
mask
ResNet50 malignant Grad-CAM
ConvNeXt malignant Grad-CAM
Swin malignant Grad-CAM
ViT attention rollout
prediction probability
true label
predicted label
IoU / energy inside mask
normalized energy inside mask
```

## Immediate Next Decision

Before implementing `compute_overlap_metrics.py`, decide the first-run
threshold scope:

1. Fixed thresholds only: `0.1` to `0.9`
2. Fixed thresholds plus top-k: `top10`, `top20`, `top30`, `top40`
3. Fixed thresholds plus top-k plus optional Otsu

Recommended first implementation:

```text
fixed thresholds + top-k
defer Otsu until after first full QC and main summary review
```

Reason:

- fixed thresholds support threshold curves and best-threshold analysis
- top-k controls salient area and is useful for cross-model fairness
- Otsu is helpful but can be added later without blocking the main analysis
- main table should prioritize `normalized_energy_in_mask`, `hit_rate`,
  `top10/top20 IoU`, and `top20 Dice`; best-threshold IoU should not be the
  headline result
