# Trash Recognition CNN — Deep Learning from Scratch with PyTorch

A deep learning portfolio project focused on **understanding how neural networks and Convolutional Neural Networks (CNNs) work from the ground up using PyTorch**, without relying on pretrained transfer-learning backbones.

This repository implements an end-to-end 6-class waste image classification pipeline—from data loading and augmentation to building a custom CNN architecture, writing the backpropagation training loop manually, and evaluating predictions.

---

## Project Objectives

Instead of treating deep learning frameworks as a black box, this project focuses on:
- **Building a CNN architecture from scratch** using `torch.nn.Module`, explicitly tracking tensor shapes across convolutional and pooling layers.
- **Understanding the PyTorch training loop** step by step: `model.train()` vs. `model.eval()`, `torch.no_grad()`, forward pass, loss computation, `loss.backward()`, `optimizer.step()`, and `optimizer.zero_grad()`.
- **Exploring regularization & optimization** through data augmentation, `Dropout`, `AdamW` weight decay, input normalization, and early stopping.
- **Evaluating model behavior** using learning curves, visual prediction grids, confusion matrices, and per-class precision/recall/F1 metrics.

---

## Dataset

The model is trained on a 6-class waste classification dataset (**2,527 images** total):

| Class | Train (`data_CNN/train`) | Test (`data_CNN/test`) | Total (`archive/datasets`) |
| :--- | :---: | :---: | :---: |
| **Paper** | 475 | 119 | 594 |
| **Glass** | 400 | 101 | 501 |
| **Plastic** | 385 | 97 | 482 |
| **Metal** | 328 | 82 | 410 |
| **Cardboard** | 322 | 81 | 403 |
| **Trash** | 109 | 28 | 137 |
| **Total** | **2,019** | **508** | **2,527** |

> **Note:** Raw image datasets (`archive/`, `data_CNN/`) and model weight checkpoints (`*.pth`) are excluded from version control via `.gitignore`.

---

## Model Architecture

The custom `CNN` class processes `3 × 128 × 128` RGB images through 5 convolutional feature-extraction blocks followed by a fully connected classifier:

```text
Input (3 × 128 × 128)
  ├── Conv2d(3 → 8, k=3, s=1, p=1)   → ReLU → MaxPool2d(2, 2)   [8 × 64 × 64]
  ├── Conv2d(8 → 16, k=3, s=1, p=1)  → ReLU → MaxPool2d(2, 2)   [16 × 32 × 32]
  ├── Conv2d(16 → 32, k=3, s=1, p=1) → ReLU → MaxPool2d(2, 2)   [32 × 16 × 16]
  ├── Conv2d(32 → 64, k=3, s=1, p=1) → ReLU → MaxPool2d(2, 2)   [64 × 8 × 8]
  ├── Conv2d(64 → 128, k=3, s=1, p=1)→ ReLU → MaxPool2d(2, 2)   [128 × 4 × 4]
  └── Flatten()                                                 [2048]
        ├── Linear(2048 → 256) → ReLU → Dropout(0.5)
        └── Linear(256 → 6) → LogSoftmax
```

- **Loss Function:** `nn.NLLLoss()` (paired with `nn.LogSoftmax`)
- **Optimizer:** `torch.optim.AdamW`
- **Data Augmentation (Train):** `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.5)`, `RandomRotation(15)`, `RandomResizedCrop(128, scale=(0.8, 1.0))`

---

## Experiments & Results

Two training configurations are explored in `TrashRecognition.ipynb`:
1. **Baseline (`model_TrashRecognition`)**: `batch_size=128`, `lr=0.001`, without input normalization, using early stopping via `jcopdl.callback.Callback`.
2. **With Normalization (`model_TrashRecognition_WithNormalize`)**: `batch_size=8`, `lr=0.005`, `weight_decay=1e-1`, with `transforms.Normalize(mean=0.456, std=0.224)`.

### Test Set Evaluation (508 images)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cardboard** | 0.86 | 0.75 | 0.80 | 81 |
| **Glass** | 0.71 | 0.61 | 0.66 | 101 |
| **Metal** | 0.68 | 0.70 | 0.69 | 82 |
| **Paper** | 0.80 | 0.83 | 0.82 | 119 |
| **Plastic** | 0.67 | 0.72 | 0.70 | 97 |
| **Trash** | 0.51 | 0.71 | 0.60 | 28 |
| **Accuracy** | | | **0.73** | **508** |
| **Macro Avg** | 0.71 | 0.72 | 0.71 | 508 |
| **Weighted Avg** | 0.73 | 0.73 | 0.73 | 508 |

---

## Repository Structure

```text
.
├── TrashRecognition.ipynb   # Main notebook: data pipeline, CNN architecture, training loop, & evaluation
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes datasets, model weights, checkpoints, and environment files
└── README.md                # Project documentation
```

When running locally, the notebook expects the following (git-ignored) directories:
- `data_CNN/train/` and `data_CNN/test/` — split image folders organized by class
- `model_TrashRecognition/` and `model_TrashRecognition_WithNormalize/` — training logs, configs, and checkpoints
- `trashNet_cnn_model.pth` — exported PyTorch `state_dict` weights

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   Place the dataset splits into `data_CNN/train/<class_name>/` and `data_CNN/test/<class_name>/`.

4. **Run the notebook**
   Open `TrashRecognition.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run the cells sequentially.
