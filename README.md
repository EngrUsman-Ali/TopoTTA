# TopoTTA: Topology-Aware Test-Time Adaptation for Unsupervised Anomaly Detection

**Authors:**  
**Authors:**  
[Ali Zia](https://scholar.google.com/citations?user=BVBJ06QAAAAJ)<sup>†,1</sup>,  
[Usman Ali](https://scholar.google.com/citations?user=2A32xVQAAAAJ)<sup>†,2</sup>,  
[Abdul Rehman](https://scholar.google.com/citations?user=ZTuS-yUAAAAJ)<sup>2</sup>,  
[Umer Ramzan](https://scholar.google.com/citations?user=D3AhoccAAAAJ)<sup>2</sup>,  
[Kang Han](https://scholar.google.com/citations?user=nIZmei8AAAAJ)<sup>1</sup>,  
[Muhammad Faheem](https://scholar.google.com/citations?user=qhs4RWQAAAAJ)<sup>2</sup>,  
[Shahnawaz Qureshi](https://scholar.google.com/citations?user=_pblOBEAAAAJ)<sup>3</sup>,  
[Wei Xiang](https://scholar.google.com/citations?user=VxQUr90AAAAJ&hl)<sup>1</sup>  

<sup>1</sup> School of Computing, Engineering and Mathematical Sciences, La Trobe University, Melbourne, Australia  
<sup>2</sup> School of Engineering and Applied Sciences, GIFT University, Gujranwala, Pakistan  
<sup>3</sup> Sino-Pak Centre for Artificial Intelligence, Pak-Austria Fachhochschule Institute of Applied Sciences and Technology, Haripur, Pakistan  

† Equal contribution  
**Corresponding Author:** [Ali Zia](mailto:A.Zia@latrobe.edu.au)


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
conda create --name TopoTTA --file requirements.txt
conda activate TopoTTA
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
