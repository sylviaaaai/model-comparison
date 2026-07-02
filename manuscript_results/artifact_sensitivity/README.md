# Artifact Sensitivity Results

This folder collects the bright-dot / marker artifact sensitivity results.

## Folder Guide

```text
tables/all_images/
  Full final test-set bright-dot / marker analysis.

tables/main_yes/
  Bright-dot / marker sensitivity restricted to usable_for_overlap == yes.

tables/usable_yes_cautious/
  Bright-dot / marker sensitivity restricted to usable_for_overlap in
  {yes, cautious}.
```

## First Tables To Read

```text
tables/all_images/bright_dot_marker_sensitivity_summary.md
tables/main_yes/bright_dot_marker_sensitivity_summary.md
tables/usable_yes_cautious/bright_dot_marker_sensitivity_summary.md
```

## Main Interpretation

```text
Bright-dot / marker artifacts are a real saliency/classification limitation.
The signal is most visible for ConvNeXt-Base benign false positives.

However, marker exclusion produces only small average overlap metric changes,
so bright-dot artifacts do not drive the main overlap ranking.
```

## Writing Note

Treat this as a limitation and sensitivity analysis, not as the primary
overlap result.

