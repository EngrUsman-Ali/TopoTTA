# TopoTTA: Topology-Aware Test-Time Adaptation for Unsupervised Anomaly Detection

**Authors:**  
[Ali Zia](https://ali-zia.me/), [Usman Ali](https://scholar.google.com/citations?user=2A32xVQAAAAJ&hl=en) [Abdul Rehman](https://scholar.google.com.pk/citations?user=A_jBBxIAAAAJ&hl=en), [Umer Ramzan](https://scholar.google.com/citations?user=D3AhoccAAAAJ&hl=en), [Waqas Ali](https://scholar.google.com/citations?user=J8_Ko78AAAAJ&hl=en), [Wei Xiang](https://scholar.google.com/citations?user=VxQUr90AAAAJ&hl=en)

---

## 🧠 Overview

Test-time adaptation (TTA) has emerged as a powerful paradigm for handling distribution shifts in deep models, particularly for anomaly segmentation, where pixel-wise labels of anomalous regions are typically unavailable during training. We introduce TopoTTA (Topological Test-Time Adaptation), a novel framework that incorporates persistent homology, a tool from topological data analysis, into the TTA pipeline to enforce structural consistency in segmentation. By applying multi-level cubical complex filtration to anomaly score maps, TopoTTA generates robust topological pseudo-labels that guide a lightweight test-time classifier, enhancing binary segmentation quality without retraining the backbone model. Our method eliminates the need for heuristic thresholding and generalises across both 2D and 3D modalities. Extensive experiments on five standard benchmarks (MVTec AD, VisA, Real-IAD, MVTec 3D-AD, AnomalyShapeNet) demonstrate significant improvements over state-of-the-art test-time unsupervised anomaly detection and segmentation methods in terms of F1 score, particularly on anomalies with complex geometries.

---

## 🏗️ Architecture Overview

Below is the overall architecture of **TopoTTA**:

![TopoTTA Architecture](fig/arch.png)

---

## ⚙️ System Specifications

- **OS:** Ubuntu 24.04  
- **GPU:** NVIDIA RTX 5090  
- **Python:** 3.13+  

---

## 🧩 Environment Setup

Create and activate the environment:
```bash
$ conda create --name TopoTTA --file requirements.txt
$ conda activate TopoTTA
```

---

## 📊 Data Placement Guide

### 1. **Dataset Structure**
Place your dataset (e.g., **MVTec AD**, **VisA**) inside the `datasets/` directory.

Example for **MVTec AD**:
```
datasets/
└── mvtec/
    ├── bottle/
    │   ├── train/
    │   ├── test/
    │   └── ground_truth/
    ├── cable/
    └── ...
```

Update dataset paths in your config file or in the command line:
```bash
--dataset_path ./datasets/mvtec
```
---

### 2. **Anomaly Score Placement**
Each class should have its corresponding folder inside `anomaly_scores/`.

Example:
```
anomaly_scores/
└── bottle/
    ├── sample01.npy/
    ├── sample02.npy/
    └── .../
```
---

## ⚡ MLP Training Parameters

- `--mlp_few_shot`: number of few-shot samples per class (ignored if `--mlp_train_fraction` > 0)  
- `--mlp_train_fraction`: fraction of training data to use (e.g., 0.3 = 30%)  

If you pass a fraction value, the few-shot parameter will be **ignored**.

---

## 🚀 Usage Example

```bash
python main.py
```