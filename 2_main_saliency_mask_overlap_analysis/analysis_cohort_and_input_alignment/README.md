# Analysis Cohort and Input Alignment

This folder documents the pre-analysis audit for the saliency-mask overlap
study. Before any numerical overlap metric can be interpreted, three things
must be true:

1. The saliency maps must come from the final selected models.
2. The saliency maps, model predictions, test images, and ROI masks must be
   aligned to the same image IDs.
3. The lesion masks must be reviewed so that the main overlap cohort uses
   reliable lesion-reference masks.

This folder is therefore the input and cohort-definition section for the main
overlap analysis. It explains what data were allowed into the overlap analysis,
what was excluded, and why.

## Start Here

Read these two documents first:

| File | Why it matters |
| --- | --- |
| `saliency_map_export_and_alignment_qc.md` | Documents saliency export, model/prediction alignment, image-ID checks, and why qualitative predicted-class maps are separated from quantitative malignant-class overlap maps. |
| `manual_mask_usability_and_analysis_cohort.md` | Documents manual mask review, usability labels, final cohort counts, and why the primary overlap analysis uses only the strict `yes` cohort. |

Together, these two files define the evidence chain before quantitative overlap
results are reported.

Recommended reading order:

1. `saliency_map_export_and_alignment_qc.md`
2. `manual_mask_usability_and_analysis_cohort.md`
3. `results/overlap_usable_scope_counts.csv`
4. `results/test_image_master_table.csv`
5. `qualitative_saliency_examples_across_mask_types_and_artifacts/README.md`


## Key Design Choices

### Saliency target

The quantitative overlap analysis uses a consistent target class:

| Model | Quantitative overlap saliency |
| --- | --- |
| ResNet50 | malignant-class Grad-CAM |
| ConvNeXt-Base | malignant-class Grad-CAM |
| Swin-Base | malignant-class Grad-CAM |
| ViT-B/16 | attention rollout |

Predicted-class maps may be used for qualitative display panels because they
show evidence for the class the model actually predicted. The numerical overlap
analysis is kept separate and focuses on malignant-class evidence where the
method supports class-specific saliency.

### Cohort definition

Manual mask review assigns every final test image to one of three usability
groups:

| Usability group | N images | Role |
| --- | ---: | --- |
| `yes` | 183 | Strict main overlap cohort. |
| `cautious` | 96 | Added only in sensitivity analysis. |
| `no` | 11 | Excluded from overlap claims. |
| Total | 290 | Final test images. |

Bright-dot or marker artifacts are not used to decide mask usability. They are
tracked separately and analyzed in `../bright_dot_marker_artifact_sensitivity/`.

## Analysis Scopes

The downstream overlap analysis uses these scopes:

| Scope | Definition | N | Malignant | Benign | Marker-positive |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `usable_for_overlap == yes` | 183 | 107 | 76 | 18 |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | 141 | 138 | 22 |
| `excluded_no` | `usable_for_overlap == no` | 11 | 6 | 5 | 3 |
| `all_images` | All final test images | 290 | 147 | 143 | 25 |

The `main_yes` scope is the primary denominator for overlap claims. The
`sensitivity_yes_cautious` scope tests whether conclusions are robust when less
localized or less certain masks are added.

## Result and Manifest Files

| File | Contents | How to use it |
| --- | --- | --- |
| `results/test_image_master_table.csv` | Image-level master table with labels, mask review fields, model predictions, selected overlap metrics, inclusion flags, and missing/exclusion reasons. | Main audit table for checking one image across masks, predictions, and models. |
| `results/overlap_usable_scope_counts.csv` | Counts for `main_yes`, `sensitivity_yes_cautious`, `excluded_no`, and `all_images`. | Quick cohort-size reference. |
| `results/saliency_manifest.csv` | Saliency-map manifest with model, method, target class, normalization, status, and probability-alignment checks. | Confirms available saliency maps and target-class definitions. |
| `results/analysis_ready_overlap_manifest.csv` | Long-form model/image/saliency manifest used as input to numerical overlap analysis. | Analysis-ready table for downstream overlap computation. |

Local raw image, mask, and saliency file paths were removed from the repository
package where appropriate. The included CSVs are intended to document the
analysis structure and cohort membership without requiring local private paths.
The public result tables omit the intermediate `mask_qc_status` generation
field; the final manual mask-quality interpretation is recorded in
`mask_qc_type`, and the final overlap-use decision is recorded in
`usable_for_overlap`.

## Qualitative Panels

The qualitative example panels are in:

```text
qualitative_saliency_examples_across_mask_types_and_artifacts/
```

Use the final upload-ready panel index:

```text
qualitative_saliency_examples_across_mask_types_and_artifacts/panels_by_case_type_final/panel_index.csv
```

These panels illustrate clean masks, cautious masks, marker-positive cases,
excluded mask examples, model-disagreement cases, and ResNet50 zero-saliency
examples. They are qualitative examples only; the quantitative overlap results
are reported in the later overlap-analysis folders.

## Tools

The `tools/` folder contains scripts used to build this audit layer:

| Script | Purpose |
| --- | --- |
| `export_resnet50_raw_gradcam.py` | Export ResNet50 raw Grad-CAM maps. |
| `export_convnext_raw_gradcam.py` | Export ConvNeXt-Base raw Grad-CAM maps. |
| `export_swin_raw_gradcam.py` | Export Swin-Base raw Grad-CAM maps. |
| `export_vit_raw_attention_rollout.py` | Export ViT-B/16 attention rollout maps. |
| `build_test_image_master_table.py` | Build the image-level master audit table. |
| `build_manual_mask_review_table.py` | Prepare manual mask-review records. |
| `fill_usable_for_overlap_from_mask_qc.py` | Fill overlap-usability labels from mask QC review. |

These scripts are included for transparency and reproducibility context. Large
raw model checkpoints and private local image paths are not included in the
GitHub package.

## Relationship to Later Analysis

This folder does not report the primary numerical overlap conclusions. Instead,
it defines the valid inputs and cohorts used by later sections:

- `../primary_overlap_metrics/`
- `../main_vs_sensitivity_overlap_result/`
- `../paired_overlap_model_comparison/`
- `../subgroup_analysis/`
- `../outside_context_partition_saliency_energy_analysis/`
- `../bright_dot_marker_artifact_sensitivity/`

In short: this folder explains why the downstream overlap results are computed
on the chosen images, masks, and saliency maps.
