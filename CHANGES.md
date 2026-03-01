# Training Pipeline Changes: Stratified Group K-Fold

## Problem

The jaguar re-identification dataset has two issues that hurt training reliability:

1. **Class imbalance**: 31 jaguars, but some have hundreds of images while others have fewer than 20. A naive random split can leave rare jaguars with zero validation samples, making it impossible to evaluate how well the model learns them.

2. **Near-duplicate leakage**: Camera traps fire in bursts — multiple photos taken within seconds of the same jaguar in the same pose. If one burst photo lands in training and an almost-identical one lands in validation, the val score looks great but the model hasn't actually learned to generalize. It just memorized the scene.

## Solution: Stratified Group K-Fold

We replaced the single `train_test_split` with `StratifiedGroupKFold` (5 folds). This combines two protections:

### Stratification (balanced classes per fold)

Each fold gets a proportional share of every jaguar's images. If jaguar "Abril" has 100 images total, each fold's validation set gets ~20 of them. This means every fold can evaluate every jaguar, and no rare class gets accidentally excluded.

Without stratification, a random split could put all 8 images of a rare jaguar into training, leaving zero for validation — you'd have no idea if the model can actually recognize that individual.

### Grouping (no near-duplicate leakage)

Burst images from the same camera trap trigger are clustered into "groups" using perceptual hashing (pHash). The rule: **all images in a group stay in the same fold**. They either all go to training or all go to validation — never split across.

**How pHash grouping works:**

1. Each image gets a 64-bit perceptual hash — a fingerprint based on the image's low-frequency structure (overall shapes and tones, not pixel-exact details)
2. Two images of the same jaguar with a hamming distance <= 8 bits (out of 64) are considered near-duplicates
3. Near-duplicates are merged into groups using union-find (connected components), so if A~B and B~C, then A, B, C are all one group
4. `StratifiedGroupKFold` ensures no group is split across train/val

**Why this matters**: Without grouping, the model can "cheat" by memorizing a specific camera trap scene rather than learning the jaguar's actual spot pattern. Grouped splits force the model to generalize across different sightings of the same individual.

## What changed in the code

### train.py

| What | Before | After |
|------|--------|-------|
| Split method | `train_test_split` (single 85/15 split) | `StratifiedGroupKFold` (5 folds) |
| Label column | `'label'` | `'ground_truth'` (matches actual CSV) |
| Validation | One fixed val set | 5 different val sets, one per fold |
| Checkpoints | Single `jaguar_convnext_arcface_best.pth` | `checkpoints/fold0_best.pth` through `fold4_best.pth` |
| CLI args | None | `--fold N` to train a single fold |

**New functions added:**

- `assign_groups()` — computes pHash for each image, clusters near-duplicates within each jaguar class using union-find, returns group assignments
- `prepare_folds()` — encodes labels, runs grouping, creates 5-fold splits, verifies zero group leakage across folds
- `train_one_fold()` — wraps the training loop for a single fold

**Unchanged:**

- `train_one_epoch()` — same training step logic (ArcFace + CrossEntropy, mixed precision, gradient clipping)
- `validate()` — same cosine similarity gap metric
- `JaguarTrainDataset` — same dataset class
- Transforms, optimizer, scheduler — all identical

### config.py

- Added `DATA_DIR = PROJECT_DIR` so train.py can reference the data directory

## How to use

```bash
# Install the hashing library (one time)
pip install imagehash

# Train all 5 folds
python train.py

# Train only fold 2 (useful for limited GPU time on Colab)
python train.py --fold 2
```

## Output

The script prints fold statistics before training:

```
Fold Statistics (5-Fold StratifiedGroupKFold)
=======================================================
  Fold 0:  380 images, 31 classes,  342 groups
  Fold 1:  381 images, 31 classes,  339 groups
  Fold 2:  379 images, 31 classes,  345 groups
  Fold 3:  380 images, 31 classes,  341 groups
  Fold 4:  380 images, 31 classes,  340 groups
Group leakage check: PASSED
```

After all folds complete, it prints a cross-validation summary:

```
CROSS-VALIDATION RESULTS
============================================================
  Fold 0: gap = 0.6234
  Fold 1: gap = 0.6187
  Fold 2: gap = 0.6301
  Fold 3: gap = 0.6145
  Fold 4: gap = 0.6278
  Mean gap: 0.6229 +/- 0.0056
```

The mean gap and its standard deviation tell you how robust the model is — a low std means consistent performance regardless of which images are held out.
