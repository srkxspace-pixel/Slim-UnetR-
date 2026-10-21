"""Show Slim UNETR predictions on validation volumes: FLAIR | ground truth | prediction.

usage: python visualize.py [num_cases]   -> writes figures/pred_case_<i>.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
import yaml
from easydict import EasyDict
from matplotlib.colors import ListedColormap
from monai.utils import ensure_tuple_rep

from src.loader import get_dataloader
from src.SlimUNETR.SlimUNETR import SlimUNETR

# 0 background, 1 edema, 2 necrotic/non-enhancing core, 3 enhancing tumor
CMAP = ListedColormap([(0, 0, 0, 0), (0.1, 0.8, 0.2, 0.5), (1, 0.85, 0, 0.6), (1, 0.1, 0.1, 0.7)])


def to_label_map(m):
    """(3, ...) TC/WT/ET masks -> single map with nested regions."""
    out = np.zeros(m.shape[1:], np.uint8)
    out[m[1] > 0] = 1
    out[m[0] > 0] = 2
    out[m[2] > 0] = 3
    return out


if __name__ == "__main__":
    num_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    config = EasyDict(yaml.load(open("config.yml", encoding="utf-8"), Loader=yaml.FullLoader))
    # ponytail: brain datasets only (3 nested classes); hepatic vessel needs its own colormap
    data_flag = "brats2021" if config.is_brats2021 else "brain2019"
    config = config[data_flag]
    config.trainer.num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SlimUNETR(**config.slim_unetr).to(device).eval()
    ckpt = f"model_store/{config.finetune.checkpoint}/best/pytorch_model.bin"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(config.trainer.image_size, 3), overlap=0.5
    )
    dice = monai.metrics.DiceMetric(include_background=True, reduction="none")

    _, val_loader = get_dataloader(config, data_flag)
    os.makedirs("figures", exist_ok=True)
    for i, batch in enumerate(val_loader):
        if i >= num_cases:
            break
        image, label = batch["image"].to(device), batch["label"].to(device)
        with torch.no_grad():
            pred = (torch.sigmoid(inferer(image, model)) > 0.5).float()
        d = dice(pred, label)[0].cpu().numpy()  # TC, WT, ET

        img, gt, pr = image[0, 0].cpu().numpy(), label[0].cpu().numpy(), pred[0].cpu().numpy()
        z = int(gt[1].sum(axis=(0, 1)).argmax())  # axial slice with the most tumor
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        for ax, title, overlay in [
            (axes[0], "FLAIR", None),
            (axes[1], "Ground truth", to_label_map(gt[..., z])),
            (axes[2], "Slim UNETR prediction", to_label_map(pr[..., z])),
        ]:
            ax.imshow(np.rot90(img[..., z]), cmap="gray")
            if overlay is not None:
                ax.imshow(np.rot90(overlay), cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Val case {i}  |  Dice  TC {d[0]:.3f}  WT {d[1]:.3f}  ET {d[2]:.3f}  (slice {z})")
        fig.tight_layout()
        out = f"figures/pred_case_{i}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out, f"TC={d[0]:.3f} WT={d[1]:.3f} ET={d[2]:.3f}")
