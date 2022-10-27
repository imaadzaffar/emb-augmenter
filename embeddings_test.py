# open 2 normal embeddings, compare cosine difference
# open 1 normal and 1 augmented embedding, compare cosine difference

import os
import torch
import numpy as np

import cv2
import h5py
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

from test_aug_times import get_true_emb
ImageFile.LOAD_TRUNCATED_IMAGES = True

from augmentations.augmentations import aug_combined, aug_rotation, aug_saturation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.autograd.set_grad_enabled(False)

slide_dir = "/media/disk2/prostate/SICAPv2/wsis"
data_root = "/media/disk2/proj_embedding_aug"
extracted_dir = "extracted_mag40x_patch256_fp"
patches_path = os.path.join(data_root, extracted_dir, "patches")

PATCH_SIZE = 256

# ---- RESNET ----
from torchvision import transforms
from utils.resnet_custom import resnet50_baseline
def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return trnsfrms_val

resnet = resnet50_baseline(pretrained=True)
resnet.to(device)
resnet.eval()

roi_transforms = eval_transforms(pretrained=True)

# ---- GENERATOR ----
from generator import GeneratorMLP
dagan_run_code = "gan_mlp_s1_lr1e-03_None_b64_20221909_182052"
dagan_state_path = f"/home/guillaume/Documents/uda/project-augmented-embeddings/2-dagan/results/sicapv2/{dagan_run_code}/s_4_checkpoint.pt"
dagan_state_dict = torch.load(dagan_state_path)
n_tokens = 1024
dropout = 0.2
generator = GeneratorMLP(n_tokens, dropout)
generator.load_state_dict(dagan_state_dict["G_state_dict"])
generator.eval().to(device)
# print(generator)

def load_patch_img(slide_id, patch_index=0, patch_size=256):
    slide_path = os.path.join(slide_dir, slide_id + ".png")
    h5_path = os.path.join(patches_path, slide_id + "_patches.h5")

    with h5py.File(h5_path, "r") as f:
        patch_coords = f["coords"][()]
        x, y = patch_coords[patch_index]
        slide = Image.open(slide_path)
        slide.load()
        image = slide.crop((x, y, x + patch_size, y + patch_size))
        image.load()
        image = np.array(image)
        return image

def get_emb(img):
    img = roi_transforms(img).unsqueeze(0)
    img = img.to(device)
    emb = resnet(img)
    return emb

def get_aug_true_emb(img, aug):
    aug_img = aug(img)
    aug_img = roi_transforms(aug_img).unsqueeze(0)
    aug_img = aug_img.to(device)
    aug_emb = resnet(aug_img)
    return aug_emb

def get_aug_gen_emb(emb):
    with torch.no_grad():
        noise = torch.randn(emb.size(0), emb.size(1), requires_grad=False).to(device)
        aug_emb = generator.forward(emb, noise)
    return aug_emb

if __name__ == "__main__":
    np.random.seed(1)

    slide_id = "16B0003394"

    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    avg_l1_original = 0.
    avg_l1_aug = 0.
    avg_l1_gen_aug = 0.
    avg_l2_original = 0.
    avg_l2_aug = 0.
    avg_l2_gen_aug = 0.
    avg_cos_original = 0.
    avg_cos_aug = 0.
    avg_cos_gen_aug = 0.

    count = 0
    
    # TODO? Something weird, lots of duplicates when comparing the two originals, maybe look into fishing rod patch generation code

    augs = {
        "combined": aug_combined,
        "rotation": aug_rotation,
        "saturation": aug_saturation,
    }
    aug_key = input("enter aug: ")
    aug_fn = augs[aug_key]

    for i in range(50):
        print(i)
        img_a = load_patch_img(slide_id, i, patch_size=256)
        emb_a = get_emb(img_a)
        aug_true_a = get_aug_true_emb(img_a, aug_fn)
        aug_gen_a = get_aug_gen_emb(emb_a)

        img_b = load_patch_img(slide_id, i+1, patch_size=256)
        emb_b = get_emb(img_b)
        # aug_true_b = get_aug_true_emb(img_b, aug_fn)
        # print(emb_a)
        # print(emb_b)
        # print(aug_true_a)
        # print(aug_true_b)
        # print(emb_a.shape)
        # print(aug_true_a.shape)
        # print(cos(emb_a, emb_b))

        avg_l1_original += l1_loss(emb_a, emb_b).item()
        avg_l1_aug += l1_loss(emb_a, aug_true_a).item()
        avg_l1_gen_aug += l1_loss(aug_gen_a, aug_true_a).item()
        avg_l2_original += l2_loss(emb_a, emb_b).item()
        avg_l2_aug += l2_loss(emb_a, aug_true_a).item()
        avg_l2_gen_aug += l2_loss(aug_gen_a, aug_true_a).item()
        avg_cos_original += cos(emb_a, emb_b).item()
        avg_cos_aug += cos(emb_a, aug_true_a).item()
        avg_cos_gen_aug += cos(aug_gen_a, aug_true_a).item()

        # print(a)
        # print(b)
        # print(cos(a,b).item())
        # assert False

        # print(l1_loss(a, b))
        # print(l1_loss(a, a_aug))
        # print(l2_loss(a, b))
        # print(l2_loss(a, a_aug))

        # print("a_aug, b_aug:", cos(a_aug, b_aug))

        count += 1

    avg_l1_original /= count
    avg_l1_aug /= count
    avg_l1_gen_aug /= count
    avg_l2_original /= count
    avg_l2_aug /= count
    avg_l2_gen_aug /= count
    avg_cos_original /= count
    avg_cos_aug /= count
    avg_cos_gen_aug /= count

    print("norms between resnet50 features for:", aug_key)
    print("n:", count)
    print("l1 for patch & random patch:     ", avg_l1_original)
    print("l1 for patch & its aug:          ", avg_l1_aug)
    print("l1 for aug patch & gen aug:      ", avg_l1_gen_aug)
    print("l2 for patch & random patch:     ", avg_l2_original)
    print("l2 for patch & its aug:          ", avg_l2_aug)
    print("l2 for aug patch & gen aug:      ", avg_l2_gen_aug)
    print("cos for patch & random patch:    ", avg_cos_original)
    print("cos for patch & its aug:         ", avg_cos_aug)
    print("cos for aug patch & gen aug:     ", avg_cos_gen_aug)
