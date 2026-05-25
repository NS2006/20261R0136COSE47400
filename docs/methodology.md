# Methodology

## 1. Baseline Model: SuperAD

The baseline model uses DINOv2-large feature extraction and nearest-neighbor matching against memory banks. Features are extracted from layers 6, 12, 18, and 24. PCA-based processing is used to reduce background effects before constructing four memory banks.

During inference, the query feature is compared with stored features in the memory banks, and the resulting anomaly information is mapped back to the image.

## 2. Identified Limitations

The presentation identifies four major challenges:

1. **Deformable normal patterns:** normal variations, such as differently shaped air bubbles, may be absent from a fixed memory set and produce false positives.
2. **Blurry or reflective objects:** surface glare may conceal subtle internal defect signals and produce false negatives.
3. **Specular highlights:** intense reflective patches can distort distances in embedding space and cause false positives.
4. **Missing components:** a missing region may resemble background texture and be incorrectly judged as normal.

## 3. Proposed Point-to-Space Mechanism

Instead of comparing a test feature only to individual stored points, the proposed model represents normal features as a vector subspace.

For a test feature vector `y` and a selected normal basis `V_k`, the normal reconstruction is:

```math
\hat{y} = V_k V_k^T y
```

The unexplained component is:

```math
r = y - \hat{y}
```

A larger residual indicates that the feature is less explainable by normal variation and may correspond to an anomaly.

## 4. Dual-Memory Bank Design

The proposed method uses two types of feature spaces:

- **Background subspace:** intended to absorb remaining background noise and specular-highlight effects.
- **Normal-object subspace:** intended to reconstruct healthy object features.

A test feature is first filtered using the background space, then evaluated against the normal-feature space. The remaining unexplained component becomes the anomaly signal.

## 5. Hyperparameter Plan

The presentation proposes testing different subspace dimensions for each category:

- `k_bg`: number of background basis vectors
- `k_normal`: number of normal-object basis vectors

A fixed initial comparison setting is:

```text
k_bg = 30
k_normal = 50
```

The next planned step is to tune these values adaptively by object category.
