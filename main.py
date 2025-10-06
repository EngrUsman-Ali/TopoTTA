# main.py
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import build_parser
from tda.tda_utils import (
    load_npy_slices, persistence_pairs, build_features,
    mask, combine
)
from vit.vit_backbone import build_vit, vit_spatial_features
from pces.mlp import train
from utils.metrics import metrics, mean
from utils.plots import save_plot
from utils.mvtec import MVTecDataset

def safe_label_to_int(label):
    try:
        if hasattr(label, 'item'): return int(label.item())
        return int(label)
    except Exception:
        import numpy as np
        return int(np.asarray(label).tolist())

def main():
    parser = build_parser()
    args = parser.parse_args()

    CLASS_NAME = args.class_name.lower()
    ANOMALY_NPY_SUBDIR = os.path.join(args.anomaly_npy_dir, CLASS_NAME)
    SAVE_BASE = None
    if args.save_masks_dir:
        SAVE_BASE = os.path.join(args.save_masks_dir, CLASS_NAME, 'RESULTS')
        os.makedirs(SAVE_BASE, exist_ok=True)

    test_dataset = MVTecDataset(dataset_path=args.dataset_path, class_name=CLASS_NAME, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataset = MVTecDataset(dataset_path=args.dataset_path, class_name=CLASS_NAME, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

    USE_CUDA = torch.cuda.is_available()
    VIT_DEVICE = torch.device(args.vit_device if (args.vit_device == 'cpu' or USE_CUDA) else 'cpu')
    DEVICE = VIT_DEVICE

    vit = build_vit(args.rgb_backbone_name, args.layers_keep, VIT_DEVICE)

    if args.mlp_train_fraction > 0.0:
        from vit.vit_backbone import vit_spatial_features as _vit_feats
        n_take = max(1, int(np.ceil(args.mlp_train_fraction * len(train_dataset))))
        feats = []
        taken = 0
        rng = np.random.default_rng(args.mlp_sample_seed)
        selected = set(rng.choice(len(train_dataset), size=n_take, replace=False).tolist())
        for i, (x, *_ ) in enumerate(tqdm(train_loader, desc='Collect train feats')):
            if i not in selected: continue
            x = x.to(VIT_DEVICE, non_blocking=True)
            feats.append(_vit_feats(vit, x).cpu())
            taken += 1
            if taken >= n_take: break
        train_feats_4d = torch.cat(feats, dim=0)
    else:
        feats = []
        for i, (x, *_ ) in enumerate(tqdm(train_loader, desc='Collect few-shot')):
            x = x.to(VIT_DEVICE, non_blocking=True)
            feats.append(vit_spatial_features(vit, x).cpu())
            if (i+1) >= args.mlp_few_shot: break
        train_feats_4d = torch.cat(feats, dim=0)

    GLOBAL_MLP, _, _, _, _ = train(
        train_feats_4d=train_feats_4d,
        max_train_pixels=args.mlp_max_train_pixels,
        max_pairs=args.mlp_max_pairs,
        mlp_hidden=args.mlp_hidden,
        mlp_embed=args.mlp_embed,
        mlp_lr=args.mlp_lr,
        mlp_epochs=args.mlp_epochs,
        device=DEVICE
    )

    # Only MLP metrics are computed and saved
    per_sample_records_mlp = []

    for idx, (image, anomaly_label, gt_mask, defect_type) in enumerate(tqdm(test_loader, desc=f'Processing {CLASS_NAME}')):
        label_int = safe_label_to_int(anomaly_label)
        if label_int == 0: continue

        npy_filename = f"{CLASS_NAME}{idx}.npy"
        anomaly_map_path = os.path.join(ANOMALY_NPY_SUBDIR, npy_filename)
        if not os.path.exists(anomaly_map_path):
            print('[WARN] Missing', anomaly_map_path)
            continue

        slices, _ = load_npy_slices(anomaly_map_path)
        img01 = slices[0]
        H, W = img01.shape

        gt_np = gt_mask.squeeze().cpu().numpy()
        if gt_np.shape != img01.shape:
            from PIL import Image
            gt_img = Image.fromarray((gt_np*255).astype('uint8'))
            gt_img = gt_img.resize((W, H), resample=Image.NEAREST)
            gt_np = (np.array(gt_img) > 127).astype(np.uint8)
        gt_bool = (gt_np.astype(np.uint8) > 0)

        pairs_sub, pairs_super = persistence_pairs(img01)
        picked = build_features(pairs_sub, pairs_super, args.k_sub, args.k_sup, args.persistence_min)

        sub_masks, super_masks = [], []
        for hom, b, d, _, filt in picked:
            m = mask(img01, hom, b, d, filt, args.sub_h1_delta, args.sup_h0_delta)
            (sub_masks if filt == 'sublevel' else super_masks).append(m)

        sub_comb = combine(sub_masks, (H, W))
        super_comb = combine(super_masks, (H, W))

        image = image.to(VIT_DEVICE, non_blocking=True)
        feat_tensor = vit_spatial_features(vit, image)[0]
        Ht, Wt = np.logical_and(sub_comb, super_comb).shape
        if feat_tensor.shape[1:] != (Ht, Wt):
            feat_resized = F.interpolate(feat_tensor.unsqueeze(0), size=(Ht, Wt), mode='bilinear', align_corners=False).squeeze(0)
        else:
            feat_resized = feat_tensor

        feats_np = np.transpose(feat_resized.cpu().numpy(), (1,2,0)).reshape(-1, feat_resized.shape[0])
        labs_np = np.logical_and(sub_comb, super_comb).reshape(-1).astype(np.float32)
        feats = torch.from_numpy(feats_np).float()
        labs = torch.from_numpy(labs_np).float()

        GLOBAL_MLP.eval()
        with torch.no_grad():
            emb_all = GLOBAL_MLP(feats.to(DEVICE))
            labs_dev = labs.to(DEVICE)
            bg = emb_all[labs_dev == 0]
            fg = emb_all[labs_dev == 1]
            if bg.numel() == 0: bg = emb_all
            if fg.numel() == 0: fg = emb_all
            proto_bg = F.normalize(bg.mean(dim=0, keepdim=True), dim=1).squeeze(0)
            proto_fg = F.normalize(fg.mean(dim=0, keepdim=True), dim=1).squeeze(0)
            d_bg = torch.norm(emb_all - proto_bg[None, :], dim=1)
            d_fg = torch.norm(emb_all - proto_fg[None, :], dim=1)
            pred_flat_mlp = (d_fg < d_bg).cpu().numpy().astype(np.uint8)

        mlp_mask = pred_flat_mlp.reshape(Ht, Wt)
        rec_mlp = metrics(mlp_mask, gt_bool)   # only MLP metrics computed
        per_sample_records_mlp.append({'idx': idx, **rec_mlp})

        if SAVE_BASE:
            save_combined_plot(img01, gt_bool, mlp_mask.astype(np.uint8), os.path.join(SAVE_BASE, f'combined_mlp_{idx}.png'))

    macro_mlp = mean(per_sample_records_mlp)

    print('\n[SUMMARY - MACRO (MLP)]')
    print(macro_mlp)

    if SAVE_BASE:
        quantitative_dir = os.path.join(SAVE_BASE, 'quantitative')
        os.makedirs(quantitative_dir, exist_ok=True)
        csv_path_m = os.path.join(quantitative_dir, f'per_sample_metrics_mlp_{CLASS_NAME}_anom_only.csv')
        with open(csv_path_m, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['idx','tp','fp','fn','precision','recall','f1','iou'])
            for r in per_sample_records_mlp:
                writer.writerow([r['idx'], r['tp'], r['fp'], r['fn'], f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}", f"{r['iou']:.6f}"])

if __name__ == '__main__':
    main()
