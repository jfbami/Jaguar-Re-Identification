# 🐆 Jaguar Re-Identification

A deep learning pipeline for identifying individual jaguars from camera trap images using metric learning and ensemble inference.

## 🌟 Highlights

- **5-fold ensemble inference** with StratifiedGroupKFold cross-validation for robust predictions
- **ConvNeXt Base + ArcFace** architecture for powerful metric learning on fine-grained visual features
- **Near-duplicate detection** using perceptual hashing to prevent data leakage from camera trap burst photos
- **k-Reciprocal Re-ranking** to boost similarity scores between mutually similar image pairs
- **Carefully tuned augmentation** that respects the asymmetric nature of jaguar rosette patterns

## ℹ️ Overview

Camera traps capture thousands of jaguar images in the wild, but identifying *which* jaguar is in each photo is critical for conservation monitoring and population tracking. This project tackles that challenge using a Re-Identification (ReID) approach — training a model to produce embeddings where images of the same jaguar are close together and different jaguars are far apart.

The pipeline handles many challenges specific to camera trap data:
- **Burst photos**: Camera traps fire in rapid succession, producing near-identical images that can leak across train/val splits if not grouped
- **Class imbalance**: Some jaguars have 180+ images while others have fewer than 15
- **Asymmetric patterns**: Jaguar rosette patterns differ between left and right flanks, so horizontal flipping is intentionally excluded from augmentation

### How It Works

1. **Training**: Images are split using StratifiedGroupKFold (5 folds), where groups are assigned via perceptual hashing to keep burst photos together. Each fold trains a ConvNeXt Base backbone with an ArcFace metric learning head.
2. **Inference**: All 5 fold models extract embeddings independently. Embeddings are averaged across folds and L2-normalized, then k-Reciprocal Re-ranking refines the similarity matrix.
3. **Submission**: Pairwise similarity scores are computed for all query-gallery pairs and exported as a CSV.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Google Colab (optional, for GPU training)

### Installation

```bash
git clone https://github.com/jfbami/Jaguar-Re-Identification.git
cd Jaguar-Re-Identification
pip install torch torchvision timm imagehash pandas scikit-learn tqdm pillow
```

### Data Setup

Place your competition data in the `data/` directory:

```
data/
├── train.csv
├── test.csv
├── train_384x384/    # Training images (384x384)
└── test_384x384/     # Test images (384x384)
```

### Training

```bash
python train.py            # Train all 5 folds
python train.py --fold 0   # Train a specific fold
```

Checkpoints are saved to `checkpoints/fold{0-4}_best.pth`.

### Inference

```bash
python main.py
```

Loads all 5 fold checkpoints, extracts ensemble embeddings, applies re-ranking, and outputs `submission.csv`.

## 📁 Project Structure

```
├── data/                   # Training & test data
├── notebooks/              # Colab workflows & reference notebooks
├── submissions/            # Past submission files
├── checkpoints/            # Trained model weights (created by training)
│
├── config.py               # Paths, hyperparameters, and settings
├── dataset.py              # Dataset class and data loading
├── models.py               # ConvNeXt + ArcFace model architecture
├── train.py                # Training with StratifiedGroupKFold
├── main.py                 # 5-fold ensemble inference
├── inference.py            # Embedding extraction & similarity computation
├── reranking.py            # k-Reciprocal Re-ranking implementation
└── preprocess_images.py    # Image preprocessing utilities
```

## ⚙️ Configuration

Key settings in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | `convnext_base` | Backbone architecture |
| `IMG_SIZE` | `(384, 384)` | Input resolution |
| `EMBEDDING_DIM` | `512` | Feature vector dimensionality |
| `BATCH_SIZE` | `32` | Training & inference batch size |
| `USE_RERANKING` | `True` | Enable k-Reciprocal Re-ranking |

Training settings in `train.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_EPOCHS` | `20` | Training epochs per fold |
| `LR` | `1e-4` | Learning rate (AdamW) |
| `N_FOLDS` | `5` | Cross-validation folds |
| `PHASH_THRESHOLD` | `8` | Hamming distance for burst grouping |

## 🧠 Technical Details

### Augmentation Pipeline

```python
RandomResizedCrop(384, scale=(0.7, 1.0))   # Simulates varying crop/zoom
ColorJitter(b=0.3, c=0.3, s=0.2, h=0.05)  # Handles lighting variation
RandomGrayscale(p=0.1)                      # Robustness to color shifts
```

**No horizontal flip** — jaguar rosette patterns are unique to each flank. Flipping would teach the model to confuse left and right sides of the same individual.

### Why StratifiedGroupKFold?

Camera traps capture burst sequences of near-identical images. A naive random split would leak these across train/val, inflating validation scores. Perceptual hashing (pHash) clusters burst photos into groups, and StratifiedGroupKFold ensures:
- All burst images stay in the same fold
- Each fold has a proportional share of every jaguar class

### Ensemble Inference

Rather than relying on a single fold's model, we extract embeddings from all 5 fold models and average them. This reduces variance and typically yields a 1-3% score improvement over any individual fold.

## 📚 References

This project builds on the following research:

| Method | Paper | Authors | Venue |
|--------|-------|---------|-------|
| ArcFace | [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) | Deng et al. | CVPR 2019 |
| k-Reciprocal Re-ranking | [Re-ranking Person Re-identification with k-Reciprocal Encoding](https://arxiv.org/abs/1701.08398) | Zhong et al. | CVPR 2017 |
| ConvNeXt | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | Liu et al. | CVPR 2022 |
| Cosine Annealing LR | [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | Loshchilov & Hutter | ICLR 2017 |
| Label Smoothing | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | Szegedy et al. | CVPR 2016 |
| Perceptual Hashing | [Implementation and Benchmarking of Perceptual Image Hash Functions](https://www.phash.org/docs/pubs/thesis_zauner.pdf) | Zauner | 2010 |
| StratifiedGroupKFold | [Scikit-learn: Machine Learning in Python](https://arxiv.org/abs/1201.0490) | Pedregosa et al. | JMLR 2011 |

## ✍️ Author

**jfbami** — Built for the Jaguar Re-Identification competition on Kaggle.

