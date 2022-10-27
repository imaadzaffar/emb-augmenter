import os
import torch
import numpy as np

from augmentations.augmentations import aug_combined, aug_rotation


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.autograd.set_grad_enabled(False)

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

import augmentations.augmentations as A

# import generator
from generator import GeneratorMLP, GeneratorIndependent

# load model
# dagan_run_code = "gan_mlp_s1_lr1e-03_None_b64_20221909_182052" # old
# dagan_run_code = "gan_mlp_s1_lr1e-03_None_b64_20221909_183959" # MLP exp
dagan_run_code = "gan_independent_s1_lr1e-03_None_b64_20221909_222224" # MLP ind
# dagan_run_code = "gan_independent_fast_s1_lr1e-03_None_b64_20222009_004136" # MLP ind fast
dagan_state_path = f"/home/guillaume/Documents/uda/project-augmented-embeddings/2-dagan/results/sicapv2/{dagan_run_code}/s_4_checkpoint.pt"
dagan_state_dict = torch.load(dagan_state_path)

n_tokens = 1024
dropout = 0.2
# generator = GeneratorMLP(n_tokens, dropout)
generator = GeneratorIndependent()
generator.load_state_dict(dagan_state_dict["G_state_dict"])
generator.eval().to(device)
# print(generator)

def get_true_emb(img, b):
    true_emb = []
    for _ in range(b):
        aug_img = A.aug_rotation(img)
        aug_img = roi_transforms(aug_img).unsqueeze(0)
        aug_img = aug_img.to(device)
        emb = resnet(aug_img)
        true_emb.append(emb)
    
    true_emb = torch.stack(true_emb, dim=0).to(device)
    return true_emb

def get_gen_emb(b):
    with torch.no_grad():
        emb = torch.randn(b, 1024, requires_grad=False).to(device)
        noise = torch.randn(emb.size(0), emb.size(1), requires_grad=False).to(device)
        aug_embs = generator.forward(emb, noise)
    
    return aug_embs

def time_func(f):
    st = time.time()
    f()
    et = time.time()
    res = et - st
    return res

if __name__ == "__main__":
    import time

    n = int(input("Enter number of times to run: "))
    b = int(input("Enter batch size: "))
    
    time_true = 0
    time_gen = 0
    for i in range(n):
        img = np.random.randint(low=0, high=255, size=(256,256,3), dtype='uint8')
        t_true = time_func(lambda: get_true_emb(img, b))
        time_true += t_true
        t_gen = time_func(lambda: get_gen_emb(b))
        time_gen += t_gen
        print(f"{i}/{n}")
        print(t_true)
        print(t_gen)
        
    time_true /= n
    time_gen /= n
    gain = time_true / time_gen

    print("true:        ", time_true)
    print("gen (mlp):   ", time_gen)
    print("gain:        ", gain)
