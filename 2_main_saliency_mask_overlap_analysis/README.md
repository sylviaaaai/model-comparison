# 2. Main Saliency-Mask Overlap Analysis

This section evaluates whether model saliency maps spatially align with lesion ROI masks on the final test set.

| Subfolder | Role |
| --- | --- |
| `analysis_cohort_and_input_alignment/` | Saliency export QC, prediction alignment, manual mask-usability cohort, and qualitative examples. |
| `primary_overlap_metrics/` | Continuous and thresholded overlap summaries plus QC panels. |
| `main_vs_sensitivity_overlap_result/` | Primary `main_yes` result and `yes+cautious` sensitivity result. |
| `paired_overlap_model_comparison/` | Paired model differences for overlap metrics with bootstrap confidence intervals. |
| `outside_context_partition_saliency_energy_analysis/` | Lesion, perilesional ring, and outside-region saliency energy analysis. |
| `subgroup_analysis/` | Overlap summaries by true label and model-specific confusion group. |
| `bright_dot_marker_artifact_sensitivity/` | Marker-artifact sensitivity and limitation analysis. |
