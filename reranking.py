import numpy as np
import torch
from tqdm.auto import tqdm


def re_ranking(probFea, gallery_num, k1, k2, lambda_value):
    print("Computing Euclidean distance matrix...")
    query_num = probFea.shape[0]
    all_num = query_num
    feat = torch.from_numpy(probFea)

    distmat = (
        torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    )
    distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
    original_dist = distmat.cpu().numpy()

    del feat
    original_dist = np.maximum(original_dist, 0)
    original_dist = np.sqrt(original_dist)

    print("Starting k-reciprocal reranking calculation...")
    initial_rank = np.argsort(original_dist, axis=1)

    V = np.zeros((all_num, all_num), dtype=np.float32)
    initial_rank = initial_rank.astype(np.int32)

    print("Computing Jaccard distance...")
    for i in tqdm(range(all_num)):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.round(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.round(k1 / 2)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                candidate_k_reciprocal_index
            ):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    print("Finalizing Jaccard distance...")
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_sim = np.exp(-final_dist)
    return final_sim
