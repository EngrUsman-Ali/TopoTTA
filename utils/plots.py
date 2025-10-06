import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def save_plot(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # RGB
    img = rgb.copy()
    if img.max() <= 1.0:
        img = (img * 255).astype('uint8')
    axes[0].imshow(img)
    axes[0].set_title('RGB')
    axes[0].axis('off')

    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('GT')
    axes[1].axis('off')

    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
