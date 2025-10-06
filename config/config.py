import argparse
from html import parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TopoTTA."
    )
    parser.add_argument('class_name', type=str)
    parser.add_argument('--anomaly_npy_dir', type=str, default='anomaly_scores_mvtec')
    parser.add_argument('--dataset_path', type=str, default='')

    # TDA
    parser.add_argument('--k_sub', type=int, default=1)
    parser.add_argument('--k_sup', type=int, default=1)
    parser.add_argument('--persistence_min', type=float, default=0.0)
    parser.add_argument('--sub_h1_delta', type=float, default=0.47, help="Δ for sublevel H1")
    parser.add_argument('--sup_h0_delta', type=float, default=0.30, help="Δ for superlevel H0")

    # ViT backbone
    parser.add_argument('--rgb_backbone_name', type=str, default='vit_base_patch8_224.dino')
    parser.add_argument('--layers_keep', type=int, default=12)
    parser.add_argument('--vit_device', type=str, default='cuda', choices=['cuda','cpu'])

    # MLP hyperparameters
    parser.add_argument('--mlp_hidden', type=int, default=1024)
    parser.add_argument('--mlp_embed',  type=int, default=768)
    parser.add_argument('--mlp_lr',     type=float, default=1e-3)
    parser.add_argument('--mlp_epochs', type=int, default=30)
    parser.add_argument('--save_mlp_csv', action='store_true')

    # Global one-class controls
    parser.add_argument('--save_masks_dir', type=str, default='./1-RESULTS-MLP-SHOTS-10')
    parser.add_argument(
        '--mlp_few_shot',
        type=int,
        default=10,
        choices=[1, 3, 5, 7, 10],
        help=(
            "Number of few-shot samples per class used for MLP training. "
            "Ignored if --mlp_train_fraction > 0."
        )
    )
    parser.add_argument(
        '--mlp_train_fraction',
        type=float,
        default=0,
        help=(
            "Fraction of the available training data to use for training the MLP. "
            "If set to a value > 0, this overrides and ignores the --mlp_few_shot setting. "
            "For example, 0.2 means use 20% of the training data."
        )
    )
    parser.add_argument('--mlp_sample_seed', type=int, default=42)
    parser.add_argument('--mlp_max_train_pixels', type=int, default=300_000)
    parser.add_argument('--mlp_max_pairs', type=int, default=200_000)

    return parser


if __name__ == '__main__':
    p = build_parser()
    _ = p.parse_args()
