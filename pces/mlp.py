from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        act_layer=nn.GELU,
        l2_normalize: bool = True
    ):
        super().__init__()
        mid = (input_dim + hidden_dim) // 2
        self.l2_normalize = l2_normalize

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mid),
            act_layer(),
            nn.Linear(mid, mid),
            act_layer(),
            nn.Linear(mid, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.l2_normalize:
            x = F.normalize(x, dim=1)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        d = F.pairwise_distance(output1, output2)
        return torch.mean((1 - label) * d.pow(2) + label * torch.clamp(self.margin - d, min=0.0).pow(2))


def train(
    train_feats_4d: torch.Tensor,
    max_train_pixels: int,
    max_pairs: int,
    mlp_hidden: int,
    mlp_embed: int,
    mlp_lr: float,
    mlp_epochs: int,
    device: torch.device
) -> Tuple[MLPEncoder, torch.Tensor, float, float, np.ndarray]:
    B, C, Hf, Wf = train_feats_4d.shape
    from vit.vit_backbone import flatten_BCHW_to_NC
    feats_all = flatten_BCHW_to_NC(train_feats_4d)  # (N,C)

    N = feats_all.shape[0]
    if N > max_train_pixels:
        idx = torch.randperm(N)[:max_train_pixels]
        feats_all = feats_all[idx]; N = feats_all.shape[0]

    num_pairs = min(max_pairs, N)
    idx1 = torch.randint(0, N, (num_pairs,))
    idx2 = torch.randint(0, N, (num_pairs,))
    y = torch.zeros(num_pairs, dtype=torch.float32)
    loader = DataLoader(TensorDataset(feats_all[idx1], feats_all[idx2], y), batch_size=512, shuffle=True, pin_memory=torch.cuda.is_available())

    mlp = MLPEncoder(input_dim=C, hidden_dim=mlp_hidden, embedding_dim=mlp_embed).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=mlp_lr)
    criterion = ContrastiveLoss(margin=1.0).to(device)

    mlp.train()
    for epoch in range(mlp_epochs):
        loss_sum, nb = 0.0, 0
        for a, b, lab in loader:
            a = a.to(device, non_blocking=True); b = b.to(device, non_blocking=True); lab = lab.to(device)
            out1 = mlp(a); out2 = mlp(b)
            loss = criterion(out1, out2, lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        print(f"[MLP-GLOBAL] epoch {epoch+1}/{mlp_epochs} | loss={(loss_sum/max(1,nb)):.6f}")

    mlp.eval()
    with torch.no_grad():
        emb = []
        chunk = 200_000
        for s in range(0, N, chunk):
            emb.append(mlp(feats_all[s:s+chunk].to(device)).cpu())
        emb = torch.cat(emb, dim=0)
        center = emb.mean(dim=0, keepdim=True)
        d = torch.norm(emb - center, dim=1).numpy()
        mu, sigma = float(np.mean(d)), float(np.std(d))
    print(f"[MLP-GLOBAL] train distances: mu={mu:.6f}  sigma={sigma:.6f}")
    return mlp, center.to(device), mu, sigma, d
