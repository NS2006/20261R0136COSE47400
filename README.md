# PCA-Guided Multi-Layer Anomaly Detection with Weighted DINOv2 Heatmaps

## Team Information

**Team 8 | Deep Learning Project**
* **Contributors:** 
    * Fahreen Qusyairi - [@FhreenQ](https://github.com/FhreenQ)
    * Giyeon Gwon - [@GiyeonGwon](https://github.com/GiyeonGwon)
    * Niko Sutiono - [@NS2006](https://github.com/NS2006)

**Key References:**
* **Baseline Framework:** [SuperAD (Zhang et al., 2025)](https://doi.org/10.48550/arXiv.2505.19750)
* **Evaluated on:** [MVTec AD 2 Dataset](https://doi.org/10.48550/arXiv.2503.21622)

## Project Overview

This repository contains our Deep Learning course project on robust industrial anomaly detection. The project investigates the limitations of **SuperAD**, a state-of-the-art nearest-neighbor memory-bank baseline, and proposes a highly optimized approach. 

By upgrading the architecture from point-to-point to **point to space**, our model significantly reduces false positives caused by environmental noise and unpredictable textures.

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

## How to Run

### 1. Install Requirements
Before running the code, ensure you have all the necessary Python dependencies installed.

Install the required packages by running:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
The model requires the MVTec AD 2 dataset to run. You can download the dataset via any of the links below:
* **[Official MVTec AD Website](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)**
* **[Google Drive: Tar.gz Version](https://drive.google.com/file/d/1EnY--wcL5g49UUZNB1XLcmAvNScs2Qfv/view?usp=sharing)**
* **[Google Drive: Pre-extracted Folder Structure Version](https://drive.google.com/drive/folders/1vMdzM4Dud_OwdxImTWS1wA4Qsc1fCMJk?usp=drive_link)**

### 3. Execute the Pipeline
To run the evaluation, use the `test_public.py` script. The `--data_root` argument is dynamic and depends on where you saved the dataset on your local machine. 

Replace `../../mvtec_ad_2` with your actual directory path:

```bash
python test_public.py --data_root ../../mvtec_ad_2
```