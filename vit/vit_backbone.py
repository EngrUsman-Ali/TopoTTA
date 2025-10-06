from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import timm

def build_vit(name: str, layers_keep: int, device: torch.device):
    vit = timm.create_model(name, pretrained=True)
    vit.eval().to(device)
    if hasattr(vit, "blocks") and isinstance(vit.blocks, nn.Sequential):
        total = len(vit.blocks)
        k = min(max(1, layers_keep), total)
        if k < total:
            vit.blocks = nn.Sequential(*vit.blocks[:k])
            print(f"[ViT] Keeping first {k}/{total} transformer blocks.")
    return vit


@torch.no_grad()
def vit_spatial_features(vit, x: torch.Tensor) -> torch.Tensor:
    z = vit.forward_features(x)
    if isinstance(z, dict):
        if 'x' in z: z = z['x']
        elif 'last_hidden_state' in z: z = z['last_hidden_state']
        else:
            for v in z.values():
                if torch.is_tensor(v):
                    z = v
                    break
    if z.dim() == 3:  # (B, tokens, C)
        B, T, C = z.shape
        if int(np.sqrt(T))**2 != T:  # likely has cls token
            z = z[:, 1:, :]
            T = z.shape[1]
        Hf = int(np.sqrt(T)); Wf = Hf
        z = z.transpose(1, 2).contiguous().view(B, C, Hf, Wf)
    elif z.dim() == 4:
        pass
    else:
        raise RuntimeError(f"Unsupported ViT features shape {tuple(z.shape)}")
    return z.float()

def flatten_BCHW_to_NC(t4: torch.Tensor) -> torch.Tensor:
    return t4.permute(0, 2, 3, 1).reshape(-1, t4.shape[1]).contiguous()
