# Bright-Dot / Marker Artifact Sensitivity

This folder evaluates whether bright-dot or marker-like artifacts near masks may affect model saliency or classification behavior. The motivation was manual image review: some masks appear near bright dots that may be clinical markers or acquisition artifacts.

## Classification of Marker Cases

The image-level master table includes a `bright_dot_or_marker` flag. This flag is separate from `usable_for_overlap`; marker status is not used to decide whether a lesion mask is usable for overlap analysis. Marker-positive images are analyzed as a limitation/sensitivity track.

## Analysis Steps

The analysis summarizes marker-positive counts, manually reviews whether top-20% saliency attends to the marker, compares classification behavior in marker-positive vs marker-negative subsets, and recomputes overlap summaries after excluding marker-positive images.

## Results

| Table | Meaning |
| --- | --- |
| `results/00_bright_dot_marker_key_summary.csv` | Compact key summary for the artifact-sensitivity result. |
| `results/01_marker_attention_summary_by_model.csv` | Manual review of marker attention by model. |
| `results/02_marker_attention_by_confusion.csv` | Marker attention summarized by model-specific confusion group. |
| `results/03_main_yes_classification_sensitivity.csv` | Main-set classification behavior by marker subset. |
| `results/04_yes_cautious_classification_sensitivity.csv` | Yes+cautious classification behavior by marker subset. |
| `results/05_main_yes_marker_exclusion_delta.csv` | Main-set deltas after marker-positive exclusion. |
| `results/06_yes_cautious_marker_exclusion_delta.csv` | Yes+cautious deltas after marker-positive exclusion. |
| `results/07_main_yes_overlap_sensitivity.csv` | Main-set overlap metrics by marker subset. |
| `results/08_yes_cautious_overlap_sensitivity.csv` | Yes+cautious overlap metrics by marker subset. |

## Result Snapshot

| Model | Marker-positive test images | Marker attention rate | Main marker-positive benign FP rate | Main FP-rate delta after marker exclusion | Main Dice top-20 delta after marker exclusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 25 | 48.0% | 0.50 | -0.004 | -0.002 |
| ViT_B16 | 25 | 84.0% | 0.50 | -0.020 | -0.002 |
| ConvNeXt_Base | 25 | 92.0% | 0.75 | -0.029 | 0.003 |
| Swin_Base | 25 | 52.0% | 0.50 | -0.023 | -0.009 |

## Conclusion

Marker artifacts do not materially shift average overlap metrics or drive the global overlap ranking. Their stronger signal is classification/artifact sensitivity: marker-positive benign cases have elevated false-positive rates, especially for ConvNeXt-Base, and manual saliency review shows frequent marker attention. This should be framed as a limitation and sanity check rather than causal proof of shortcut learning.
