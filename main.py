import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from models import JaguarReIDModel
from dataset import JaguarDataset, get_val_transform
from inference import compute_cosine_similarities, compute_rerank_similarities
from reranking import re_ranking


# ──────────────────────────────────────────────
# ENSEMBLE CONFIG
# ──────────────────────────────────────────────
CHECKPOINT_DIR = 'checkpoints'
N_FOLDS = 5
NUM_CLASSES = 31


def load_fold_models():
    """
    Load all 5 fold checkpoints and return a list of models.
    Raises FileNotFoundError if any checkpoint is missing
    (no more silent failures with random weights).
    """
    models = []
    for fold in range(N_FOLDS):
        path = os.path.join(CHECKPOINT_DIR, f'fold{fold}_best.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Make sure training has completed and checkpoints are saved in '{CHECKPOINT_DIR}/'."
            )
        model = JaguarReIDModel(NUM_CLASSES, Config.EMBEDDING_DIM, pretrained=False).to(Config.DEVICE)
        state_dict = torch.load(path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"  Loaded {path}")
    print(f"All {N_FOLDS} fold models loaded successfully.\n")
    return models


def extract_ensemble_embeddings(models, loader, device):
    """
    Extract embeddings from each fold model, average them,
    and L2-normalize the result.
    """
    # Extract embeddings from each model
    all_model_feats = []
    fnames_order = None

    for i, model in enumerate(models):
        model.eval()
        feats_list = []
        fnames_list = []

        with torch.no_grad():
            for images, filenames in tqdm(loader, desc=f"  Fold {i} embeddings", leave=False):
                images = images.to(device)
                feats = model(images, labels=None)
                feats = feats.cpu().numpy()

                for fname, feat in zip(filenames, feats):
                    feats_list.append(feat)
                    fnames_list.append(fname)

        all_model_feats.append(np.array(feats_list))

        # Verify filename order is consistent across models
        if fnames_order is None:
            fnames_order = fnames_list
        else:
            assert fnames_order == fnames_list, "Filename order mismatch between fold models!"

    # Average embeddings across all folds
    stacked = np.stack(all_model_feats, axis=0)  # (N_FOLDS, N_images, embed_dim)
    mean_feats = stacked.mean(axis=0)             # (N_images, embed_dim)

    # L2-normalize
    norms = np.linalg.norm(mean_feats, axis=1, keepdims=True) + 1e-8
    mean_feats = mean_feats / norms

    # Build dict for cosine similarity fallback
    embeddings_dict = {fname: mean_feats[i] for i, fname in enumerate(fnames_order)}

    print(f"Ensemble embeddings: {mean_feats.shape[0]} images, {mean_feats.shape[1]}-dim\n")
    return embeddings_dict, mean_feats, fnames_order


def build_test_loader():
    test_df_raw = pd.read_csv(Config.TEST_CSV)
    unique_test_images = sorted(list(set(test_df_raw['query_image']) | set(test_df_raw['gallery_image'])))
    test_images_df = pd.DataFrame({'filename': unique_test_images})

    val_transform = get_val_transform()
    test_dataset = JaguarDataset(test_images_df, Config.TEST_IMG_DIR, transform=val_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    return test_df_raw, test_loader


def main():
    print(f"Device: {Config.DEVICE}\n")

    # Load all fold models
    print(f"Loading {N_FOLDS}-fold ensemble from '{CHECKPOINT_DIR}/'...")
    models = load_fold_models()

    # Build data loader
    test_df_raw, test_loader = build_test_loader()

    # Extract ensemble embeddings
    print("Extracting ensemble embeddings...")
    embeddings_dict, feats_array, fnames_list = extract_ensemble_embeddings(
        models, test_loader, Config.DEVICE
    )
    fname_to_idx = {fname: i for i, fname in enumerate(fnames_list)}

    # Compute similarities
    if Config.USE_RERANKING:
        print("Applying k-Reciprocal Reranking...")
        rerank_sim_matrix = re_ranking(feats_array, len(feats_array), Config.K1, Config.K2, Config.LAMBDA_VALUE)
        print("Calculating final submission scores...")
        similarities = compute_rerank_similarities(test_df_raw, feats_array, fname_to_idx, rerank_sim_matrix)
    else:
        print("Applying Cosine Similarity...")
        similarities = compute_cosine_similarities(test_df_raw, embeddings_dict)

    # Save submission
    submission = test_df_raw[['row_id']].copy()
    submission['similarity'] = similarities
    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv")
    print("Done! Generated submission.csv for upload.")


if __name__ == '__main__':
    main()
