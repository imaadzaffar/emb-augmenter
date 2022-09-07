# open 2 normal embeddings, compare cosine difference
# open 1 normal and 1 augmented embedding, compare cosine difference

import os
from stringprep import b1_set
import numpy as np
import torch
import pandas as pd
import random

results_dir = "./3-downstream/results/sicapv2/combined_dagan_s1_lr1e-03_b1_20220109_173102/"

fishing_rod_results_dir = "/media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/"

labels_path = "./2-dagan/datasets_csv/labels.csv"

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

def load_patch_embs(slide_id):
    fpath = os.path.join(fishing_rod_results_dir, slide_id + ".pt")
    return torch.load(fpath)
    
def get_embs(patch_embs, patch_index):
    original = patch_embs[patch_index, 0, :]

    aug_index = np.random.choice([6, 7, 8, 9])
    aug = patch_embs[patch_index, aug_index, :]

    return original, aug

def get_emb_aug(patch_embs, patch_index, aug_index=6):
    # aug_index = np.random.choice([6, 7, 8, 9])
    return patch_embs[patch_index, aug_index, :]

if __name__ == "__main__":
    np.random.seed(1)

    df = pd.read_csv(labels_path)
    slide_ids = list(df["image_id"].values)

    avg_l1_original = 0.
    avg_l1_aug = 0.
    avg_l2_original = 0.
    avg_l2_aug = 0.
    avg_cos_original = 0.
    avg_cos_aug = 0.

    count = 0

    for slide_id in slide_ids:

        patch_embs = load_patch_embs(slide_id=slide_id)

        n = patch_embs.size(0)
        # n = 10

        # Something weird, lots of duplicates when comparing the two originals, maybe look into fishing rod patch generation code
        for patch_index in range(n):
            print("patch_index:", patch_index)
            a, a_aug = get_embs(patch_embs, patch_index)

            rand_index = np.random.choice([i for i in range(n) if i != patch_index])
            b, b_aug = get_embs(patch_embs, rand_index)
            avg_cos_original += cos(a, b).item()
            avg_cos_aug += cos(a, a_aug).item()

            # print("a, b:", cos(a, b).item())

            # print("a, a_aug:", cos(a, a_aug).item())
            # print("b, b_aug:", cos(b, b_aug).item())
            # print(a)
            # print(b)
            # print(a_aug)
            # print(b_aug)
            l1_loss = torch.nn.L1Loss()
            l2_loss = torch.nn.MSELoss()

            avg_l1_original += l1_loss(a, b).item()
            avg_l1_aug += l1_loss(a, a_aug).item()

            avg_l2_original += l2_loss(a, b).item()
            avg_l2_aug += l2_loss(a, a_aug).item()

            # print(l1_loss(a, b))
            # print(l1_loss(a, a_aug))
            # print(l2_loss(a, b))
            # print(l2_loss(a, a_aug))

            # print("a_aug, b_aug:", cos(a_aug, b_aug))

            count += 1


    avg_l1_original /= count
    avg_l1_aug /= count
    avg_l2_original /= count
    avg_l2_aug /= count
    avg_cos_original /= count
    avg_cos_aug /= count

    print("total patch embeddings:          ", count)
    print("l1 for patch & random patch:     ", avg_l1_original)
    print("l1 for patch & its aug:          ", avg_l1_aug)
    print("l2 for patch & random patch:     ", avg_l2_original)
    print("l2 for patch & its aug:          ", avg_l2_aug)
    print("cos for patch & random patch:    ", avg_cos_original)
    print("cos for patch & its aug:         ", avg_cos_aug)
