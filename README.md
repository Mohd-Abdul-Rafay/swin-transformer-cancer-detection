# Swin Transformer for Histopathologic Cancer Detection  
### Medical AI — Vision Transformers in Computational Pathology

This repository presents a transformer-based Medical AI pipeline for automated detection of metastatic cancer in histopathology image patches using the PCam (PatchCamelyon) dataset.

The project focuses on applying Vision Transformers to medical imaging while emphasizing robust evaluation under class imbalance and clinically relevant performance trade-offs.

---

## Problem Context

Histopathology cancer detection requires:

- Robust classification under class imbalance  
- Sensitivity to metastatic patterns  
- Stability across tissue orientation and rotation  
- Reliable evaluation beyond simple accuracy  

This project evaluates transformer-based modeling under these constraints.

---

## Dataset

- **Dataset:** PCam (PatchCamelyon)  
- Binary classification: Tumor vs. Non-Tumor  
- Controlled split: 60% Train / 20% Validation / 20% Test  

The dataset is not redistributed in this repository.

---

## Model Architecture

- **Backbone:** Swin Transformer (`swin_tiny_patch4_window7_224`)
- Source: `timm` pretrained weights
- Adapted for binary classification
- Fine-tuned end-to-end using PyTorch

---

## Training Setup

- Resolution: 32×32 center-cropped patches  
- Batch size: 32  
- Epochs: 25  
- Optimizer: Adam (lr=1e-4)  
- Loss: CrossEntropyLoss  
- Hardware: NVIDIA GPU  
- Framework: PyTorch  

Applied strong spatial augmentation:

- Random rotation (0–360°)
- Random affine transformation (translation + shear)

---

## Test Set Performance

| Metric | Value |
|--------|--------|
| ROC-AUC | ≈ 0.955 |
| PR-AUC | ≈ 0.99 |
| Accuracy | 0.8570 |
| Sensitivity (Recall) | ≈ 83.4% |
| Specificity | ≈ 93.6% |

Evaluation includes ROC curves, Precision–Recall analysis, and confusion matrix transparency.

---

## Results Structure

results/
├── validation/
│   ├── roc_val.png
│   ├── pr_val.png
│   └── confusion_matrix_val.png
└── test/
│   ├── roc_test.png
│   ├── pr_test.png
│   └── confusion_matrix_test.png


---

## Key Insights

- Vision Transformers perform strongly on medical patch classification tasks.
- PR-AUC is a more informative metric than accuracy under class imbalance.
- Sensitivity–specificity trade-offs are critical for medical AI deployment.
- Strong augmentation improves rotational invariance in histopathology imagery.

---

## Repository Structure

swin-transformer-cancer-detection/
├── notebooks/
│   └── swin-transformers-version2.ipynb
├── results/
│   ├── validation/
│   └── test/
├── assets/
├── requirements.txt
├── LICENSE
└── README.md

---

## Citation

If this work is useful in your research, please cite:

```bibtex
@software{rafay2025swinmedical,
  author  = {Abdul Rafay Mohd},
  title   = {Swin Transformer for Histopathologic Cancer Detection},
  year    = {2025},
  url     = {https://github.com/Mohd-Abdul-Rafay/swin-transformer-cancer-detection}
}
