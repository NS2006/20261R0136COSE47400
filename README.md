# PCA-Guided Multi-Layer Anomaly Detection with Weighted DINOv2 Heatmaps

## Project Overview

This repository contains our Deep Learning course project on robust industrial anomaly detection. The project investigates the limitations of **SuperAD**, a state-of-the-art nearest-neighbor memory-bank baseline, and proposes a highly optimized **Dual-Memory Bank with Orthogonal Projection** approach. 

By upgrading the architecture from point-to-point nearest-neighbor matching to **residual-based subspace reconstruction**, our model significantly reduces false positives caused by environmental noise and unpredictable textures.

**Current Status:** Completed Project / Final Implementation

## Motivation & Baseline Limitations

While the SuperAD baseline is highly effective, our analysis revealed four critical blind spots in real-world industrial datasets (evaluated on MVTec AD 2):

1. **Air Bubble Confusion:** Normal objects with highly variable patterns (e.g., air bubbles in fruit jelly) trigger massive false positives because a fixed memory bank cannot memorize infinite variations.
2. **Hidden by Light:** The model misses actual dark defects because bouncing light and reflections act as camouflage.
3. **Surface Glare:** Bright lighting creates specular highlights on shiny objects (e.g., metal cans) that the model incorrectly detects as physical damage.
4. **The Background Blend:** Missing-component anomalies (e.g., a hole in sheet metal) go undetected because the empty space mimics the normal dark background.

## Our Proposed Approach

To solve these limitations, our group fundamentally upgraded the anomaly scoring architecture. Our key contributions include:

* **Dual-Memory Bank Initialization:** We dynamically separate extracted DINOv2 features into a Foreground (Object) Bank and a Background Bank using custom high-resolution morphological masking.
* **Background Noise Suppression (Soft Projection):** We extract the principal axes of noise (e.g., glare, lighting shifts) from the background bank and apply a soft projection matrix to filter these out of the test features before scoring.
* **Residual-Based SVD Scoring (The Covariance Trick):** We replaced the standard K-NN search with an orthogonal projection onto a Normal SVD subspace. The unexplained variance (the residual) serves as the true anomaly score. To prevent Out-Of-Memory (OOM) errors during basis construction, we utilized a mathematical covariance trick, accelerating processing exponentially.
* **Weighted Multi-Layer Heatmap Fusion:** Instead of equally averaging the four DINOv2 layers, we apply empirical weights (`0.10, 0.18, 0.32, 0.40`). This heavily penalizes the noisy, low-level textures of early layers while prioritizing the robust semantic features of deeper layers.

## Results Summary

By combining optimal PCA projection strength ($\alpha = 0.5$) with weighted layer fusion, our model achieved a mean AUROC of **0.7705**, successfully outperforming the SuperAD baseline (0.7671) on the MVTec AD 2 dataset, with notable improvements in challenging classes like `fruit_jelly` and `sheet_metal`.

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── data/          # Dataset instructions and local data placement
├── docs/          # Project overview, PPT slides, and methodology documentation
├── experiments/   # Experiment planning and configuration notes
├── notebooks/     # Exploratory notebooks and visualization engines
├── results/       # Evaluation results, heatmaps, histograms, and CSV logs
└── src/           # Core implementation code (DINOv2, SVD logic, masking)
```

## Team Information

**Team 8 | Deep Learning Project**
* **Contributors:** 
    * Fahreen Qusyairi - [@FhreenQ](https://github.com/FhreenQ)
    * Giyeon Gwon - [@GiyeonGwon](https://github.com/GiyeonGwon)
    * Niko Sutiono - [@NS2006](https://github.com/NS2006)
