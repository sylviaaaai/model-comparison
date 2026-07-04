# Manual Mask Usability and Analysis Cohort

This document summarizes how the final test images were manually reviewed for
mask usability before lesion-mask overlap analysis.

## Purpose

The overlap experiment compares model saliency maps against lesion-region masks.
Because not every available mask is an equally reliable lesion reference, the
test images were manually reviewed and assigned to analysis groups before the
main overlap results were interpreted.

This manual review defined:

- a strict main overlap cohort,
- a broader sensitivity cohort,
- and a small excluded mask-quality limitation set.

Bright-dot and marker artifacts were not used to decide mask usability. They
were analyzed separately as an artifact-sensitivity limitation.

## Manual Review Fields

The image-level master table records the manual review using these fields. The
mask-QC fields and the bright-dot/marker artifact field were assigned by human
visual review, not by an automated detector:

| Field | Meaning |
| --- | --- |
| `manual_mask_type` | What type of finding or annotation the mask appears to represent. |
| `mask_qc_type` | Whether the mask is spatially usable as a lesion-reference region. |
| `usable_for_overlap` | Final overlap-usability label: `yes`, `cautious`, or `no`. |
| `manual_review_notes` | Free-text notes from manual review. |
| `bright_dot_or_marker` | Manually reviewed marker/artifact flag, not used to define usability. |

## Usability Rules

The final rule was:

| Usability label | Interpretation | Use in analysis |
| --- | --- | --- |
| `yes` | Mask is a localized and reasonably tight lesion reference. | Main overlap analysis. |
| `cautious` | Mask is usable only with caution because the annotation is less localized, overinclusive, border-touching, diffuse, or uncertain. | Sensitivity analysis only. |
| `no` | Mask is not reliable as a lesion-reference region. | Excluded from overlap claims. |

Operationally:

- `yes` cases were localized mass-like or localized malignant annotations with
  `tight_or_reasonable` mask QC.
- `cautious` cases included calcification regions, architectural-distortion-like
  findings, uncertain annotations, overinclusive masks, border-touching masks,
  or diffuse large regions.
- `no` cases included cropped/truncated masks or foreground-like/contour masks
  that could make numerical overlap misleading.

## Final Cohort Counts

The final test set contained 290 images.

| Usability group | N images | Role |
| --- | ---: | --- |
| `yes` | 183 | Strict main overlap cohort. |
| `cautious` | 96 | Added only in sensitivity analysis. |
| `no` | 11 | Excluded from overlap claims; mask-quality limitation. |
| Total | 290 | Final test images. |

The analysis scopes used later were:

| Scope | Definition | N | Malignant | Benign | Marker-positive |
| --- | --- | ---: | ---: | ---: | ---: |
| `main_yes` | `usable_for_overlap == yes` | 183 | 107 | 76 | 18 |
| `sensitivity_yes_cautious` | `usable_for_overlap in {yes, cautious}` | 279 | 141 | 138 | 22 |
| `excluded_no` | `usable_for_overlap == no` | 11 | 6 | 5 | 3 |
| `all_images` | All final test images | 290 | 147 | 143 | 25 |

## Manual Mask-Type Breakdown

Manual mask-type labels by final usability group:

| Usability group | Manual mask type | N |
| --- | --- | ---: |
| `yes` | localized malignant annotation | 107 |
| `yes` | localized mass-like | 76 |
| `cautious` | calcification region | 49 |
| `cautious` | architectural-distortion-like | 18 |
| `cautious` | uncertain annotation | 18 |
| `cautious` | localized malignant annotation | 6 |
| `cautious` | localized mass-like | 5 |
| `no` | localized malignant annotation | 6 |
| `no` | localized mass-like | 3 |
| `no` | uncertain annotation | 2 |

Mask-QC labels by final usability group:

| Usability group | Mask QC type | N |
| --- | --- | ---: |
| `yes` | tight or reasonable | 183 |
| `cautious` | tight or reasonable | 65 |
| `cautious` | diffuse large | 18 |
| `cautious` | border touching | 7 |
| `cautious` | overinclusive annotation | 6 |
| `no` | cropped or truncated | 9 |
| `no` | foreground-like or contour | 2 |

## Why the Main Analysis Uses 183 Images

The primary overlap analysis uses only `usable_for_overlap = yes` because those
183 images have the most reliable lesion-reference masks. This reduces the
risk that an apparently high or low overlap score is caused by broad,
truncated, contour-like, or uncertain annotation geometry rather than model
localization behavior.

The 96 `cautious` images are still valuable, but they are used as a sensitivity
check. If the model ranking remains similar after adding these images, the
localization conclusion is less likely to depend only on the strictest mask
subset.

The 11 `no` images are retained in the master table for transparency, but they
are not used for primary overlap claims.

## Bright-Dot and Marker Handling

Bright-dot or marker artifacts were manually reviewed and intentionally kept
separate from mask usability:

```text
bright_dot_or_marker is not used to decide usable_for_overlap.
```

This matters because marker artifacts can affect classification or saliency
behavior, but they do not necessarily mean the lesion mask itself is unusable.
Marker-positive cases were therefore analyzed later as a separate limitation
track.

## Takeaway

The final overlap analysis uses a strict 183-image mask-usable cohort as the
main denominator and a 279-image yes+cautious cohort as a robustness check. The
excluded 11 images are treated as a mask-quality limitation, not as evidence for
or against model localization performance.

## Audit Files

- `gradcam_mask_overlap/metadata/test_image_master_table.csv`
- `gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_counts.csv`
- `gradcam_mask_overlap/overlap_analysis/master_table_usable_sets/overlap_usable_scope_summary.md`
- `gradcam_mask_overlap/experiment_logs/2026-07-01_master_table_overlap_and_marker_sensitivity_log.md`
