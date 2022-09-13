"""
A Modified Implementation of Deep Attention MIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .generator import GeneratorMLP


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """
        Attention Network without Gating (2 fc layers)
        args:
            L: input feature dimension
            D: hidden layer dimension
            dropout: whether to use dropout (p = 0.25)
            n_classes: number of classes (experimental usage for multiclass MIL)
        """
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L: input feature dimension
            D: hidden layer dimension
            dropout: whether to use dropout (p = 0.25)
            n_classes: number of classes (experimental usage for multiclass MIL)
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class SingleTaskAttentionMILClassifier(nn.Module):
    def __init__(self, gate=True, size_arg="big", dropout=False, n_classes=6):
        super(SingleTaskAttentionMILClassifier, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            device_ids = [0, 1, 2]
            self.attention_net = nn.DataParallel(
                self.attention_net, device_ids=device_ids
            ).to("cuda")

        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if return_features:
            results_dict.update({"features": M})

        return logits, Y_prob, Y_hat, A_raw, results_dict


if __name__ == "__main__":

    test = True
    if test:
        net = SingleTaskAttentionMILClassifier().cuda()
        generator = GeneratorMLP().cuda()

        patch_embs = torch.randn(67, 1024).cuda()
        print(patch_embs.size())

        noise = torch.randn(67, 1024)

        aug_embs = generator.forward(patch_embs, noise)
        print(aug_embs.size())

        out = net(patch_embs)
        print("Out shape:", out)
