# Beyond Nearest Neighbor: Dual-Memory Bank & Orthogonal Projection for Robust Anomaly Detection

## Project Overview

This repository contains our deep learning course project on robust anomaly detection. The project investigates limitations of **SuperAD**, a nearest-neighbor memory-bank baseline, and proposes a **Dual-Memory Bank with Orthogonal Projection** approach.

**Current status:** Midterm proposal / implementation in progress.

## Motivation

The baseline model may fail in challenging cases such as:

- Normal objects with highly variable patterns, such as air bubbles in fruit jelly, which can cause false positives.
- Blurry or reflective objects, where glare can hide subtle defects and cause false negatives.
- Specular highlights on glass or metal surfaces, which can be mistaken for anomalies.
- Missing-component defects, where the missing region resembles normal background texture.

## Baseline: SuperAD

The baseline workflow presented in our project is:

1. Extract features using **DINOv2-large** at layers 6, 12, 18, and 24.
2. Apply PCA-based filtering to reduce background influence.
3. Construct four memory banks.
4. Use nearest-neighbor search during inference to map anomaly scores back to images.

## Proposed Model

Our proposed method replaces point-to-point nearest-neighbor matching with **point-to-space reconstruction**:

- Construct normal feature subspaces using Singular Value Decomposition (SVD).
- Use orthogonal projection to measure how much of a test feature cannot be explained by the normal subspace.
- Add a separate **background memory bank** so background effects and highlights can be reduced before normal-feature reconstruction.
- Study category-specific choices of `k_bg` and `k_normal` rather than using one fixed setting for every object category.

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── data/          # Dataset instructions and local data placement
├── docs/          # Project overview and methodology documentation
├── experiments/   # Experiment planning and configuration notes
├── notebooks/     # Exploratory notebooks
├── results/       # Evaluation results, figures, and tables
└── src/           # Implementation code
```

## Team Information

- Team: Team 8
- Course: Deep Learning Project
- Team Repository Owner: Fahreen Qusyairi (`FhreenQ`)

Add all teammates' names, GitHub IDs, and forked repository links here before final submission.

## GitHub Workflow

This repository is used as the **Team Repository**. Team members should fork this repository, develop changes in their own forks, and submit completed work back through Pull Requests.
