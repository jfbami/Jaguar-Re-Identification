import numpy as np
import torch
from tqdm.auto import tqdm


def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = {}
    feats_list = []
    fnames_list = []

    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Extracting Features"):
            images = images.to(device)
            feats = model(images, labels=None)
            feats = feats.cpu().numpy()

            for fname, feat in zip(filenames, feats):
                embeddings[fname] = feat
                feats_list.append(feat)
                fnames_list.append(fname)

    return embeddings, np.array(feats_list), fnames_list


def compute_cosine_similarities(test_df, embeddings_dict):
    similarities = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Cosine Similarity"):
        q_emb = embeddings_dict[row['query_image']]
        g_emb = embeddings_dict[row['gallery_image']]
        sim = np.dot(q_emb, g_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(g_emb) + 1e-6)
        sim = (sim + 1) / 2
        similarities.append(sim)
    return similarities


def compute_rerank_similarities(test_df, feats_array, fname_to_idx, rerank_sim_matrix):
    similarities = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Rerank Scores"):
        q_idx = fname_to_idx[row['query_image']]
        g_idx = fname_to_idx[row['gallery_image']]
        sim = rerank_sim_matrix[q_idx, g_idx]
        similarities.append(sim)
    return similarities
