# Qualitative Saliency Examples Across Mask Types and Artifacts

This folder contains representative four-model saliency-mask panels selected for supervisor-facing review. The final upload-ready panels are organized by mask type and artifact type, not by a single model confusion group.

## Display Convention

- Panels use the primary overlap saliency target: malignant-class Grad-CAM for ResNet50, ConvNeXt-Base, and Swin-Base; attention rollout for ViT-B/16.

- Predicted-class maps are not mixed into these localization panels.

- TP/TN/FP/FN labels are retained as metadata because each model can assign a different confusion group to the same image.

- These examples illustrate qualitative saliency-mask alignment patterns and do not replace the quantitative overlap tables.

## Final Upload-Ready Folder

Use this folder for GitHub upload:

```text
panels_by_case_type_final/
```

It contains two panels per category:

| Category folder | Purpose |
| --- | --- |
| `01_clean_localized_yes` | Clean main-set localized masks. |
| `02_calcification_cautious` | Calcification-region cautious masks.  |
| `03_architectural_distortion_cautious` | Architectural-distortion cautious masks. |
| `04_diffuse_uncertain_cautious` | Diffuse large or uncertain cautious masks. |
| `05_overinclusive_or_border_cautious` | Overinclusive or border-touching cautious masks. |
| `06_marker_positive_benign` | Marker-positive benign cases. |
| `07_marker_positive_malignant` | Marker-positive malignant cases. |
| `08_usability_no_examples` | Excluded not-usable masks for limitation discussion. |
| `09_model_disagreement_examples` | Cases where model classifications disagree. |
| `10_resnet50_zero_saliency_examples` | ResNet50 malignant-class Grad-CAM zero-saliency examples: one TN and one FN. |

## Image Index

Use `panel_index.csv` as the source of truth for the final panel set.
The CSV contains one row per image panel and records:

- display order and category folder
- image ID and true label
- manually reviewed mask type and mask QC type
- overlap-usability group
- manually reviewed marker/artifact status
- each model's TP/TN/FP/FN prediction type
- repository-relative panel path
- selection reason

The README intentionally does not repeat the full 20-case table, so the CSV and
panel files stay synchronized.

## Notes


- The two `usable_for_overlap = no` examples are included only to explain mask-quality limitations, not as primary overlap-analysis examples.

- The two ResNet50 zero-saliency examples are included to explain expected malignant-class Grad-CAM behavior when ResNet50 predicts benign; they should be interpreted as a saliency-target limitation example, not as missing heatmap files.
