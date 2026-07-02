# Manual Mask Review Template

Review table:

```text
gradcam_mask_overlap/metadata/manual_mask_review_template.csv
```

This table is the new source of truth for manual mask review. It is one row per
test image and should be joined by `image_id`.

The first-pass `manual_mask_type`, `mask_qc_type`, and `notes` values were
seeded from reviewer notes with:

```text
gradcam_mask_overlap/tools/seed_manual_mask_review_from_notes.py
```

The `bright_dot_or_marker` values were updated from reviewer notes with:

```text
gradcam_mask_overlap/tools/update_bright_dot_marker_review.py
```

Remaining blank `bright_dot_or_marker` values were finalized as `none` with:

```text
gradcam_mask_overlap/tools/finalize_bright_dot_marker_none.py
```

`usable_for_overlap` was filled from manual mask-review fields with:

```text
gradcam_mask_overlap/tools/fill_usable_for_overlap_from_mask_qc.py
```

This rule intentionally does not use `bright_dot_or_marker`; bright-dot /
marker artifacts are handled in a separate limitation/sensitivity analysis.

## Fill Workflow

Scripts used so far:

```text
build_manual_mask_review_table.py
  Builds the image-level review template and image/mask references.

seed_manual_mask_review_from_notes.py
  Fills manual_mask_type, mask_qc_type, and notes from reviewer notes.

update_bright_dot_marker_review.py
  Fills explicitly listed bright-dot / marker cases.

finalize_bright_dot_marker_none.py
  Fills all remaining blank bright_dot_or_marker cells as none.

fill_usable_for_overlap_from_mask_qc.py
  Fills usable_for_overlap from manual_mask_type and mask_qc_type only, and
  writes a focused cautious/no review file.
```

## Columns To Fill

`manual_mask_type`

Allowed values:

```text
localized_mass_like
calcification_region
architectural_distortion_like
localized_malignant_annotation
uncertain_annotation
```

`mask_qc_type`

Allowed values:

```text
tight_or_reasonable
overinclusive_annotation
border_touching
diffuse_large
foreground_like_or_contour
cropped_or_truncated
bad_or_mismatched
```

`bright_dot_or_marker`

Allowed values:

```text
none
inside_mask
outside_mask
on_border
uncertain
```

`usable_for_overlap`

Allowed values:

```text
yes
cautious
no
```

`notes`

Free-text notes, for example:

```text
mask large lesion small
right-side bright dot
left border touching
```

## Current Seed Counts

```text
manual_mask_type:
  localized_malignant_annotation = 119
  localized_mass_like = 84
  calcification_region = 49
  uncertain_annotation = 20
  architectural_distortion_like = 18

mask_qc_type:
  tight_or_reasonable = 248
  diffuse_large = 18
  cropped_or_truncated = 9
  border_touching = 7
  overinclusive_annotation = 6
  foreground_like_or_contour = 2

bright_dot_or_marker:
  none = 265
  outside_mask = 13
  inside_mask = 9
  on_border = 2
  uncertain = 1

usable_for_overlap:
  yes = 183
  cautious = 96
  no = 11

bright_dot_or_marker: no blank rows
notes: nonblank for 54 rows
```

## Usable For Overlap Rule

The current rule uses only mask annotation and mask-QC fields:

```text
yes:
  manual_mask_type in {localized_mass_like, localized_malignant_annotation}
  and mask_qc_type = tight_or_reasonable

cautious:
  manual_mask_type in {calcification_region, architectural_distortion_like,
  uncertain_annotation}
  or mask_qc_type in {overinclusive_annotation, border_touching, diffuse_large}

no:
  mask_qc_type in {foreground_like_or_contour, cropped_or_truncated,
  bad_or_mismatched}
```

Important:

```text
bright_dot_or_marker is not used to decide usable_for_overlap.
```

Rule outputs:

```text
gradcam_mask_overlap/metadata/usable_for_overlap_rule_summary.json
gradcam_mask_overlap/metadata/usable_for_overlap_review_candidates.csv
```

The review-candidates CSV contains only the 107 `cautious` or `no` rows for
targeted manual review.

Manual correction after diffuse-large QC review:

```text
0019_C_0019_1.LEFT_CC:
  manual_mask_type changed from uncertain_annotation to localized_malignant_annotation
  mask_qc_type changed from diffuse_large to tight_or_reasonable
  usable_for_overlap changed from no to yes

0019_C_0019_1.LEFT_MLO:
  manual_mask_type changed from uncertain_annotation to localized_malignant_annotation
  mask_qc_type changed from diffuse_large to tight_or_reasonable
  usable_for_overlap changed from no to yes

0019_C_0019_1.RIGHT_CC:
  kept mask_qc_type = diffuse_large
  usable_for_overlap changed from no to cautious

0019_C_0019_1.RIGHT_MLO:
  kept mask_qc_type = diffuse_large
  usable_for_overlap changed from no to cautious
```

## Image References

Use these columns for visual review:

```text
roi_image_path
roi_mask_path
qc_overlay_path
original_image_path
original_mask_path
binary_filled_mask_path
```

## Previous Labels

The old labels are kept only as references:

```text
previous_manual_annotation_type
previous_manual_annotation_type_cn
previous_usable_for_overlap_manual
previous_mapping_scope
previous_manual_experiment_group
original_note
```

Do not treat them as final labels.

## Downstream Rule

After this CSV is completed, use it to update the manual annotation columns in:

```text
gradcam_mask_overlap/metadata/test_image_master_table.csv
gradcam_mask_overlap/metadata/analysis_ready_overlap_manifest.csv
gradcam_mask_overlap/metadata/overlap_selected_metrics_wide.csv
gradcam_mask_overlap/metadata/overlap_metrics.csv
```

The join key should be:

```text
image_id
```
