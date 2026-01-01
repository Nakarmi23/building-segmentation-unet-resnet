from pathlib import Path

import torch
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from src.datasets.whu_dataset import WHUDataset, build_transforms

def denormalize(img_t: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = img_t * std + mean
    return x.clamp(0, 1)

def show_sample(img_t: torch.Tensor, mask_t: torch.Tensor, title: str = ""):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    img = denormalize(img_t.cpu(), mean, std).permute(1, 2, 0).numpy()
    mask = mask_t.squeeze().cpu().numpy()

    overlay = img.copy()

    overlay[mask > 0.5] = np.clip(overlay[mask > 0.5] * 0.5 + np.array([1.0, 0, 0]) * 0.5, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].axis("off")
                 
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def check_split(root_dir: str, split: str, image_size=(512, 512), batch_size=2):
    print(f"\n=== Checking {split} split ===")

    tfm = build_transforms(train=(split=="train"), image_size=image_size)
    ds = WHUDataset(root_dir=root_dir, split=split, transform=tfm, return_meta=True)

    print(f"Dataset size ({split}): {len(ds)} samples")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    
    if len(batch) == 3:
        imgs, masks, metas = batch
        sample_name = Path(metas["image_path"][0]).name if isinstance(metas, dict) else "sample"
    else:
        imgs, masks = batch
        sample_name = "sample"

    print (f"Batch image tensor:", imgs.shape, imgs.dtype)
    print (f"Batch mask tensor:", masks.shape, masks.dtype)

    unique_vals = torch.unique(masks[0]).cpu().numpy()
    print(f"Unique values in the first mask of the batch: {unique_vals}")

    assert imgs.ndim == 4 and imgs.shape[1] == 3, "Image tensor should have shape (B, 3, H, W)"
    assert masks.ndim == 4 and masks.shape[1] == 1, "Mask tensor should have shape (B, 1, H, W)"
    assert masks.min() >= 0 and masks.max() <= 1, "Masks should be in [0,1]"

    show_sample(imgs[0], masks[0], title=f"{split} | {Path(sample_name)}")


if __name__ == "__main__":
    # Update if your project root differs
    ROOT_DIR = "data"

    # Try 512 first; if you get GPU/memory issues later you can reduce to 256
    IMAGE_SIZE = (512, 512)

    check_split(ROOT_DIR, "train", image_size=IMAGE_SIZE, batch_size=2)
    check_split(ROOT_DIR, "val", image_size=IMAGE_SIZE, batch_size=2)
