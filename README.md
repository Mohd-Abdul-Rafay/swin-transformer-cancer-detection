# Swin Transformer for Histopathologic Cancer Detection  
### Medical AI — Vision Transformers in Computational Pathology

This repository presents a transformer-based Medical AI pipeline for automated detection of metastatic cancer in histopathology image patches using the PCam (PatchCamelyon) dataset.

The project investigates the practical application of Vision Transformers in computational pathology, with emphasis on generalization robustness, calibration behavior, and clinically meaningful evaluation under class imbalance.

Key objectives:

- Fine-tune a pretrained Swin Transformer for binary tumor classification  
- Establish a controlled 60/20/20 train–validation–test split  
- Evaluate performance using ROC-AUC, PR-AUC, sensitivity, specificity, and confusion matrix analysis  
- Analyze diagnostic trade-offs between false positives and false negatives in a medical setting  

This work highlights how transformer architectures behave under limited-resolution medical patch constraints and real-world diagnostic performance requirements.

---

## Problem Context

Histopathology cancer detection requires:

- Robust classification under class imbalance  
- Sensitivity to metastatic patterns  
- Stability across tissue orientation and rotation  
- Reliable evaluation beyond simple accuracy  

This project evaluates transformer-based modeling under these constraints.

---

## Role & Contribution

This project was developed as part of a 5-member team.

I served as the primary model developer and was responsible for:

- Designing and implementing the full Swin Transformer training pipeline
- Building custom Dataset and DataLoader modules in PyTorch
- Defining and enforcing the 60/20/20 train–validation–test split
- Fine-tuning the pretrained `swin_tiny_patch4_window7_224` model (timm)
- Implementing augmentation strategies (rotations, affine transforms)
- Developing evaluation pipeline (ROC-AUC, PR-AUC, confusion matrix, sensitivity/specificity)
- Conducting model performance analysis and diagnostic interpretation

Team contributions included project coordination, feature engineering support, parallel model experimentation, and deployment assistance.

---

## Dataset

This project uses the PatchCamelyon (PCam) dataset for binary classification of histopathology image patches (tumor vs. non-tumor tissue).

The dataset was originally released for the Kaggle **Histopathologic Cancer Detection** competition.

Source:
https://www.kaggle.com/competitions/histopathologic-cancer-detection

The dataset is not redistributed in this repository and must be obtained directly from Kaggle.

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

```
results/
├── validation/
│   ├── roc_val.png
│   ├── pr_val.png
│   └── confusion_matrix_val.png
└── test/
    ├── roc_test.png
    ├── pr_test.png
    └── confusion_matrix_test.png
```

---

## Key Insights

- Vision Transformers perform strongly on medical patch classification tasks.
- PR-AUC is a more informative metric than accuracy under class imbalance.
- Sensitivity–specificity trade-offs are critical for medical AI deployment.
- Strong augmentation improves rotational invariance in histopathology imagery.

---

## Repository Structure


```
swin-transformer-cancer-detection/
├── notebooks/
│   └── swin-transformers-version2.ipynb
├── results/
│   ├── validation/
│   └── test/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Author

**Abdul Rafay Mohd**  
Artificial Intelligence | Medical AI | Computer Vision  

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

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
