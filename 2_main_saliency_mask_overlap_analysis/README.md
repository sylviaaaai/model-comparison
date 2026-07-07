# 2. Main Saliency-Mask Overlap Analysis

## Purpose

This chapter evaluates whether each model's saliency map spatially aligns with the annotated lesion ROI mask on the final test set.

The main question is not model classification accuracy. The focus is interpretability alignment: when a model highlights image regions, how much of that saliency falls inside or near the annotated lesion mask, and does this pattern differ across models, mask-quality scopes, prediction outcomes, or artifact-sensitive cases?

## Recommended Reading Order

Read the folders in this order:

1. [analysis_cohort_and_input_alignment/](analysis_cohort_and_input_alignment/)
   - Start here to understand the final test cohort, saliency-map export, image/prediction/mask alignment, and manual mask-usability review.

2. [primary_overlap_metrics/](primary_overlap_metrics/)
   - Read this next for the first all-image overlap metric computation across the four primary model-method pairs.

3. [main_vs_sensitivity_overlap_result/](main_vs_sensitivity_overlap_result/)
   - This is the main result folder. It compares the strict `main_yes` cohort with the broader `sensitivity_yes_cautious` (`yes+cautious`) sensitivity cohort.

4. [paired_overlap_model_comparison/](paired_overlap_model_comparison/)
   - Read this after the main result to see paired ConvNeXt-vs-other model comparisons on the same images, with bootstrap confidence intervals.

5. [outside_context_partition_saliency_energy_analysis/](outside_context_partition_saliency_energy_analysis/)
   - Read this for the supplementary lesion/perilesional/outside saliency-energy distribution analysis.

6. [subgroup_analysis/](subgroup_analysis/)
   - Read this for exploratory breakdowns by true label and model-specific TP/FN/FP/TN groups.

7. [bright_dot_marker_artifact_sensitivity/](bright_dot_marker_artifact_sensitivity/)
   - Read this last as a limitation/sensitivity analysis for marker-like artifacts and bright-dot cases.

## Chapter Structure

| Subfolder | What it does | How to use it |
| --- | --- | --- |
| [analysis_cohort_and_input_alignment/](analysis_cohort_and_input_alignment/) | Defines the analysis cohort and verifies that saliency maps, predictions, images, and ROI masks are aligned. Also documents manual mask-usability labels. | Use this as the evidence chain before trusting any overlap metric. |
| [primary_overlap_metrics/](primary_overlap_metrics/) | Computes the first continuous and thresholded saliency-mask overlap metrics for all 290 final test images per primary model-method pair. | Use this as the upstream metric-generation step. |
| [main_vs_sensitivity_overlap_result/](main_vs_sensitivity_overlap_result/) | Reports the primary strict usable-mask result (`main_yes`) and the broader mask-QC sensitivity result (`sensitivity_yes_cautious`). | Use this as the main overlap conclusion. |
| [paired_overlap_model_comparison/](paired_overlap_model_comparison/) | Tests ConvNeXt-Base against other models on the same images using paired differences and bootstrap confidence intervals. | Use this to check whether model differences remain when image difficulty is paired. |
| [outside_context_partition_saliency_energy_analysis/](outside_context_partition_saliency_energy_analysis/) | Splits saliency energy into lesion, perilesional ring, and outside regions. | Use this to interpret whether saliency is lesion-centered or more context/outside distributed. |
| [subgroup_analysis/](subgroup_analysis/) | Stratifies overlap metrics by malignant/benign label and model-specific TP/FN/FP/TN groups. | Use this as exploratory interpretation of where the main overlap signal comes from. |
| [bright_dot_marker_artifact_sensitivity/](bright_dot_marker_artifact_sensitivity/) | Evaluates whether bright-dot or marker-like artifacts affect saliency attention, classification behavior, or overlap rankings. | Use this as an artifact limitation and sanity-check analysis. |

## Analysis Flow

The chapter follows this logic:

1. Define the usable analysis cohort and verify alignment.
2. Compute baseline overlap metrics for each model.
3. Restrict to reliable masks for the primary result.
4. Add cautious masks as a sensitivity check.
5. Compare models using paired image-level differences.
6. Test whether saliency is lesion-centered, perilesional, or outside-distributed.
7. Examine descriptive subgroups and artifact-sensitive cases.

This structure separates the main claim from supporting analyses. The main claim should come from [main_vs_sensitivity_overlap_result/](main_vs_sensitivity_overlap_result/), while the paired, subgroup, outside-context, and marker analyses provide robustness checks and interpretation.

## Main Takeaway

Across the strict primary set and the broader sensitivity set, ConvNeXt-Base shows the strongest lesion-mask overlap pattern overall. ResNet50 is the closest comparator, Swin-Base is intermediate, and ViT-B/16 attention rollout has the weakest lesion-mask overlap.

The paired analysis supports this ordering, especially when ConvNeXt-Base is compared with ViT-B/16 and Swin-Base. The ConvNeXt-vs-ResNet50 comparison is closer, particularly in the strict `main_yes` set.

## Important Caveats

- Mask usability matters. The primary result uses strict `usable_for_overlap == yes` masks, and `sensitivity_yes_cautious` (`yes+cautious`) is used as a sensitivity analysis.
- ResNet50 has zero-saliency cases. These mainly affect peak-distance metrics because no saliency peak exists for all-zero maps.
- ViT-B/16 uses a different explanation method. Its overlap metrics are based on attention rollout, whereas ResNet50, ConvNeXt-Base, and Swin-Base use Grad-CAM. Therefore, lower ViT-B/16 overlap should not be interpreted as proof that ViT-B/16 did not attend to the lesion; it may partly reflect that attention rollout and Grad-CAM are not directly equivalent saliency methods.
- Subgroup analyses are descriptive. FP/FN denominators can be small, and model-specific confusion groups are not the same image sets across models.
- Marker/bright-dot cases are tracked separately. They are not used to define mask usability, but they are analyzed as a limitation and artifact-sensitivity check.
- Saliency overlap should be interpreted as spatial alignment evidence, not causal proof of which pixels caused the model's prediction.

## Where To Find Details

Each subfolder has its own README with table-level and metric-level details. This top-level README is only a chapter guide; use the subfolder READMEs when you need exact CSV meanings, column definitions, denominator notes, or metric-specific caveats.
