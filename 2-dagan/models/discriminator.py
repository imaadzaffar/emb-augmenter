from cgi import test
import math

from torch import nn, Tensor
import torch
import torch.nn as nn
from torch.nn import ReLU, ELU


class DiscriminatorMLP(nn.Module):
    def __init__(self, n_tokens=1024, dropout=0.0) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_tokens * 2, n_tokens),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens, n_tokens // 2),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens // 2, n_tokens // 4),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens // 4, 1),
        )

    def forward(self, x, x_aug):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: augmented feature [B x 1024 x1]
            Returns:
                - is_real: if it's real of x [B x 1024]
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("x device:",x.device)
        # print("x_aug device:",x_aug.device)
        is_real = torch.cat([x, x_aug], dim=1).squeeze()
        is_real = self.net(is_real)
        return is_real


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class DiscriminatorTransformer(nn.Module):
    def __init__(
        self,
        n_tokens=1024,
        n_heads=1,
        emb_dim=8,
        dropout=0.0,
    ):
        super(DiscriminatorTransformer, self).__init__()

        self.augment_channel = nn.Linear(2, emb_dim)
        self.pos_encoding = PositionalEncoding(
            d_model=emb_dim, dropout=dropout, max_len=n_tokens
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            dim_feedforward=emb_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.reduce_channel = nn.Linear(emb_dim, 1)

    def forward(self, x, x_aug):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: augmented feature [B x 1024 x1]
            Returns:
                - is_real: if it's real of x [B x 1024]
        """
        x, x_aug = x.unsqueeze(2), x_aug.unsqueeze(2)
        is_real = torch.cat(
            [x, x_aug], dim=2
        )  # concat original and noise: B x seq_len x 2
        is_real = torch.permute(is_real, (1, 0, 2))  # seq len first: seq_len x B x 2
        is_real = self.augment_channel(is_real)  # increase channel: seq_len x B x 8
        is_real = self.pos_encoding(is_real)  # position encoding: seq_len x B x 8
        is_real = self.transformer(is_real)  # apply transformer: seq_len x B x 8
        is_real = torch.permute(is_real, (1, 0, 2))  # batch first: B x seq_len x 8
        is_real = torch.mean(is_real, dim=1)  # spatial pooling: B x 8
        is_real = self.reduce_channel(is_real)  # reduce channel: B x 1

        return is_real


class DiscriminatorIndependent(nn.Module):
    def __init__(
        self,
    ):
        super(DiscriminatorIndependent, self).__init__()

        self.all_mlps = []
        for _ in range(1024):
            mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
            # mlp = nn.Sequential(
            #     nn.Linear(2, 1),
            # )
            self.all_mlps.append(mlp)
        self.all_mlps = nn.ModuleList(self.all_mlps)

    def forward(self, x, x_aug):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: augmented feature [B x 1024 x1]
            Returns:
                - is_real: if it's real of x [B x 1024]
        """
        x, x_aug = x.unsqueeze(2), x_aug.unsqueeze(2)
        data = torch.cat([x, x_aug], dim=2)  # concat original and noise: B x 1024 x 2
        data = torch.permute(data, (1, 0, 2))  # feature first
        all_outputs = []
        for i in range(1024):
            o = self.all_mlps[i](data[i, :, :])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, dim=1)
        is_real = torch.mean(all_outputs, dim=1)
        return is_real


class DiscriminatorIndependentFast(nn.Module):
    def __init__(
        self,
    ):
        super(DiscriminatorIndependentFast, self).__init__()

        self.all_mlps = []
        for _ in range(1):
            mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
            # mlp = nn.Sequential(
            #     nn.Linear(2, 1),
            # )
            self.all_mlps.append(mlp)
        self.all_mlps = nn.ModuleList(self.all_mlps)

    def forward(self, x, x_aug):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: augmented feature [B x 1024 x1]
            Returns:
                - is_real: if it's real of x [B x 1024]
        """
        x, x_aug = x.unsqueeze(2), x_aug.unsqueeze(2)
        data = torch.cat([x, x_aug], dim=2)  # concat original and noise: B x 1024 x 2
        data = torch.permute(data, (1, 0, 2))  # feature first
        all_outputs = []
        for i in range(1024):
            o = self.all_mlps[0](data[i, :, :])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, dim=1)
        is_real = torch.mean(all_outputs, dim=1)
        return is_real


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    test_independent = True
    if test_independent:
        net = DiscriminatorIndependent().cuda()
        x = torch.randn(32, 1024).cuda()
        z = torch.randn(32, 1024).cuda()
        out = net(x, z)
        print("Out shape:", out.shape)

    test_mlp = False
    if test_mlp:
        net = DiscriminatorMLP().cuda()
        # print(net)
        print("Number of parameters:", count_parameters(net))
        x = torch.randn(32, 1024).cuda()
        z = torch.randn(32, 1024).cuda()
        out = net(x, z)
        print("Out shape:", out.shape)

    test_transformer = False
    if test_transformer:
        net = DiscriminatorTransformer(n_heads=4, emb_dim=64).cuda()
        # print("Net:", net)
        print("Number of parameters:", count_parameters(net))
        x = torch.randn(32, 1024).cuda()
        z = torch.randn(32, 1024).cuda()
        out = net(x, z)
        print("Out shape:", out.shape)
