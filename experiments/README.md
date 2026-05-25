# Experiments

This folder records experiment settings and comparison plans.

## Planned Comparisons

| Experiment | Description |
|---|---|
| Baseline | SuperAD using nearest-neighbor memory-bank search |
| Proposed - Fixed K | Dual-memory bank projection with `k_bg = 30`, `k_normal = 50` |
| Proposed - Adaptive K | Category-specific tuning of `k_bg` and `k_normal` |

## Cases of Interest

Evaluate whether the proposed method improves behavior on:

- Deformable normal patterns, such as air bubbles
- Blurry objects or subsurface defects affected by glare
- Specular-highlight false positives
- Missing-component defects

Add confirmed evaluation metrics and final hyperparameter settings after implementation.
