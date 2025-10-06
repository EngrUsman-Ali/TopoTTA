from typing import List, Tuple
import numpy as np
import gudhi as gd

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    vmin, vmax = np.min(x), np.max(x)
    return (x - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(x, float)

def load_npy_slices(path: str):
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2:
        return [normalize01(arr)], [""]
    if arr.ndim == 3:
        if arr.shape[0] == arr.shape[1] and arr.shape[2] not in (arr.shape[0], arr.shape[1]):
            arr = np.moveaxis(arr, -1, 0)
        slices = [normalize01(arr[i]) for i in range(arr.shape[0])]
        tags   = [f"_idx{i:03d}" for i in range(arr.shape[0])]
        return slices, tags
    raise ValueError(f"Unsupported npy shape {arr.shape}; expected 2D or 3D.")

def persistence_pairs(img01: np.ndarray):
    # sublevel
    cc_sub = gd.CubicalComplex(dimensions=img01.shape, top_dimensional_cells=img01.flatten(order="F"))
    cc_sub.compute_persistence()
    pairs_sub = cc_sub.persistence()
    # superlevel on inverted image
    inv = 1.0 - img01
    cc_sup = gd.CubicalComplex(dimensions=inv.shape, top_dimensional_cells=inv.flatten(order="F"))
    cc_sup.compute_persistence()
    pairs_super = cc_sup.persistence()
    return pairs_sub, pairs_super

def build_features(pairs_sub, pairs_super, k_sub: int, k_sup: int, persistence_min: float):
    sub_feats, super_feats = [], []
    for dim, (b, d) in pairs_sub:
        if d == float('inf'): continue
        if dim == 1: sub_feats.append(('H1', b, d, d - b, 'sublevel'))
    for dim, (b_inv, d_inv) in pairs_super:
        if d_inv == float('inf'): continue
        if dim == 0:
            b0 = 1.0 - d_inv; d0 = 1.0 - b_inv
            super_feats.append(('H0', b0, d0, d0 - b0, 'superlevel'))
    sub_feats   = [f for f in sub_feats   if f[3] > persistence_min]
    super_feats = [f for f in super_feats if f[3] > persistence_min]
    sub_feats.sort(key=lambda x: x[3], reverse=True)
    super_feats.sort(key=lambda x: x[3], reverse=True)
    return sub_feats[:k_sub] + super_feats[:k_sup]

def mask(img01: np.ndarray, homology: str, birth: float, death: float, filtration: str, sub_h1_delta: float, sup_h0_delta: float) -> np.ndarray:
    thr = max(0.0, death - (sub_h1_delta if filtration == 'sublevel' else sup_h0_delta))
    return (img01 >= thr)

def combine(masks: List[np.ndarray], ref_shape: Tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(ref_shape, dtype=bool)
    out = np.zeros(ref_shape, dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out
