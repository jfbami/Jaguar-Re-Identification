import os
import argparse
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from config import Config
from models import JaguarReIDModel


# ──────────────────────────────────────────────
# 1.  TRAINING CONFIG  (tweak these as needed)
# ──────────────────────────────────────────────
TRAIN_CSV      = Config.TRAIN_CSV
TRAIN_IMG_DIR  = Config.TRAIN_IMG_DIR

NUM_EPOCHS     = 20
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 2
SAVE_DIR       = 'checkpoints'
LABEL_COL      = 'ground_truth'
FILE_COL       = 'filename'

# Cross-validation
N_FOLDS        = 5
SEED           = 42

# Near-duplicate grouping
PHASH_THRESHOLD = 8   # hamming distance <= this → same group


# ──────────────────────────────────────────────
# 2.  DATASET
# ──────────────────────────────────────────────
def crop_alphachannel(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        bbox = img.getbbox()
        if bbox:
            return img.crop(bbox)
    return img


class JaguarTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row[FILE_COL])

        try:
            image = Image.open(img_path).convert("RGBA")
            image = crop_alphachannel(image)
            image = image.convert("RGB")
        except Exception:
            image = Image.new("RGB", Config.IMG_SIZE)

        if self.transform:
            image = self.transform(image)

        label = int(row['label'])
        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
# 3.  GROUPING & FOLD ASSIGNMENT
# ──────────────────────────────────────────────
def assign_groups(df, img_dir, threshold=PHASH_THRESHOLD):
    """
    Derive burst groups via perceptual hashing.

    Images of the same jaguar with hamming distance <= threshold
    are clustered into the same group (connected components).
    This prevents near-duplicate leakage across folds.

    Falls back to per-image groups if imagehash is not installed.
    """
    try:
        import imagehash
    except ImportError:
        print("WARNING: imagehash not installed. Each image is its own group.")
        print("  Install with: pip install imagehash")
        return np.arange(len(df))

    print("Computing perceptual hashes for near-duplicate detection...")

    # Compute pHash for each image
    hashes = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Hashing"):
        img_path = os.path.join(img_dir, row[FILE_COL])
        try:
            img = Image.open(img_path).convert("RGB")
            h = imagehash.phash(img)
        except Exception:
            h = None
        hashes.append(h)

    # Union-Find for connected components
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare within each class (only same-jaguar images can be near-duplicates)
    labels = df['label'].values
    for cls in df['label'].unique():
        indices = np.where(labels == cls)[0]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                hi, hj = hashes[indices[i]], hashes[indices[j]]
                if hi is not None and hj is not None:
                    if hi - hj <= threshold:
                        union(indices[i], indices[j])

    # Convert to group IDs
    groups = np.array([find(i) for i in range(len(df))])

    n_images = len(df)
    n_groups = len(set(groups))
    n_grouped = n_images - n_groups
    print(f"  {n_groups} groups from {n_images} images "
          f"({n_grouped} images merged into existing groups)")

    return groups


def prepare_folds(csv_path, n_folds=N_FOLDS, seed=SEED):
    """
    Load training data, encode labels, assign groups, and create
    StratifiedGroupKFold splits.

    Returns DataFrame with added 'label' (int), 'group', and 'fold' columns.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Unique identities: {df[LABEL_COL].nunique()}")

    # Encode string jaguar names to integers
    le = LabelEncoder()
    df['label'] = le.fit_transform(df[LABEL_COL])
    num_classes = len(le.classes_)
    print(f"Num classes (after encoding): {num_classes}")

    # Print class distribution
    print("\nClass distribution:")
    for cls_name, cls_id in zip(le.classes_, range(num_classes)):
        count = (df['label'] == cls_id).sum()
        print(f"  {cls_name}: {count} images")

    # Assign groups for near-duplicate detection
    groups = assign_groups(df, TRAIN_IMG_DIR)
    df['group'] = groups

    # Create folds
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df['fold'] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(X=df, y=df['label'], groups=df['group'])
    ):
        df.loc[df.index[val_idx], 'fold'] = fold_idx

    # Log fold statistics
    print(f"\n{'='*55}")
    print(f"Fold Statistics ({n_folds}-Fold StratifiedGroupKFold)")
    print(f"{'='*55}")
    for fold_idx in range(n_folds):
        fold_df = df[df['fold'] == fold_idx]
        n_imgs = len(fold_df)
        n_cls = fold_df['label'].nunique()
        n_grps = fold_df['group'].nunique()
        print(f"  Fold {fold_idx}: {n_imgs:4d} images, "
              f"{n_cls:2d} classes, {n_grps:4d} groups")

    # Verify no group leakage
    for fold_idx in range(n_folds):
        val_groups = set(df[df['fold'] == fold_idx]['group'])
        train_groups = set(df[df['fold'] != fold_idx]['group'])
        leaked = val_groups & train_groups
        assert len(leaked) == 0, (
            f"Group leakage in fold {fold_idx}: {len(leaked)} groups "
            f"appear in both train and val!"
        )
    print("Group leakage check: PASSED\n")

    return df, num_classes


# ──────────────────────────────────────────────
# 4.  HELPERS
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    """
    Compute embeddings and return average cosine similarity
    between same-class pairs minus different-class pairs.
    """
    model.eval()
    all_embs   = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device)
        embs   = model(images, labels=None)
        all_embs.append(embs.cpu())
        all_labels.append(labels)

    all_embs   = torch.cat(all_embs,   dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    sim = all_embs @ all_embs.T

    N = sim.shape[0]
    same_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1))
    eye_mask  = torch.eye(N, dtype=torch.bool)
    same_mask = same_mask & ~eye_mask
    diff_mask = ~same_mask & ~eye_mask

    same_sim = sim[same_mask].mean().item() if same_mask.sum() > 0 else 0.0
    diff_sim = sim[diff_mask].mean().item() if diff_mask.sum() > 0 else 0.0
    gap      = same_sim - diff_sim

    return same_sim, diff_sim, gap


# ──────────────────────────────────────────────
# 5.  TRAINING LOOP
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images, labels=labels)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, labels=labels)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    avg_loss = total_loss / total
    acc      = correct / total
    return avg_loss, acc


def train_one_fold(fold_idx, df, num_classes, device):
    """Train and validate a single fold."""
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*60}")

    train_df = df[df['fold'] != fold_idx].reset_index(drop=True)
    val_df   = df[df['fold'] == fold_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    # Datasets & loaders
    train_ds = JaguarTrainDataset(train_df, TRAIN_IMG_DIR, transform=get_train_transform())
    val_ds   = JaguarTrainDataset(val_df,   TRAIN_IMG_DIR, transform=get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = JaguarReIDModel(num_classes, Config.EMBEDDING_DIM, pretrained=True).to(device)

    # Optimiser & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_gap   = -float('inf')
    best_epoch = 0
    save_path  = os.path.join(SAVE_DIR, f'fold{fold_idx}_best.pth')

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        same_sim, diff_sim, gap = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(
            f"Fold {fold_idx} Epoch [{epoch:03d}/{NUM_EPOCHS}]  "
            f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
            f"val_same={same_sim:.4f}  val_diff={diff_sim:.4f}  gap={gap:.4f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        if gap > best_gap:
            best_gap   = gap
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best gap={best_gap:.4f} — saved to {save_path}")

    print(f"Fold {fold_idx} done. Best gap {best_gap:.4f} at epoch {best_epoch}.")
    return best_gap


# ──────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Jaguar Re-ID Training with StratifiedGroupKFold")
    parser.add_argument('--fold', type=int, default=-1,
                        help='Train a specific fold (0-4). Default -1 = all folds.')
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"Device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Prepare folds
    df, num_classes = prepare_folds(TRAIN_CSV, n_folds=N_FOLDS, seed=SEED)

    # Save fold assignments for reproducibility
    df.to_csv(os.path.join(SAVE_DIR, 'folds.csv'), index=False)
    print(f"Fold assignments saved to {SAVE_DIR}/folds.csv")

    # Determine which folds to train
    if args.fold >= 0:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(N_FOLDS))

    # Train
    results = {}
    for fold_idx in folds_to_train:
        best_gap = train_one_fold(fold_idx, df, num_classes, device)
        results[fold_idx] = best_gap

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for fold_idx, gap in results.items():
        print(f"  Fold {fold_idx}: gap = {gap:.4f}")
    if len(results) > 1:
        mean_gap = np.mean(list(results.values()))
        std_gap  = np.std(list(results.values()))
        print(f"  Mean gap: {mean_gap:.4f} ± {std_gap:.4f}")

    print(f"\nCheckpoints saved in: {SAVE_DIR}/")


if __name__ == '__main__':
    main()
