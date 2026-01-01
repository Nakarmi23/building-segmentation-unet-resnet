from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.whu_dataset import WHUDataset, build_transforms
from src.models.unet import UNet

def save_vis(img, gt, pred, out_path: Path, title:str):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = (img.cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()

    gt = gt.cpu().squeeze(0).numpy()
    pred = pred.cpu().squeeze(0).numpy()

    overlay = img.copy()
    mask_pred = (pred > 0.5)

    red = np.array([1.0, 0.0, 0.0], dtype=overlay.dtype)
    overlay[mask_pred] = overlay[mask_pred] * 0.5 + red * 0.5

    fig, ax = plt.subplots(1,4, figsize=(16,4))
    ax[0].imshow(img); ax[0].set_title('Image'); ax[0].axis('off')
    ax[1].imshow(gt, cmap='gray'); ax[1].set_title('GT'); ax[1].axis('off')
    ax[2].imshow(pred, cmap='gray'); ax[2].set_title('Pred'); ax[2].axis('off')
    ax[3].imshow(overlay); ax[3].set_title('Overlay'); ax[3].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tfm = build_transforms(train=False, image_size=(256,256))
    ds = WHUDataset(root_dir='data/', split='test', transform=tfm, return_meta=True)
    out_dir = Path('outputs/preds/unet')
    out_dir.mkdir(parents=True, exist_ok=True)

    model = UNet(features=(32, 64, 128, 256)).to(device)
    ckpt = torch.load('checkpoints/unet/best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    num_samples = 10  # how many images to visualize
    random_seed = 42  # optional, for reproducibility

    random.seed(random_seed)
    indices = random.sample(range(len(ds)), num_samples)

    for i in indices:
        img, mask, meta = ds[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).cpu().squeeze(0)
        
        save_vis(img, mask, prob, out_dir / f"{i:02d}_{meta['stem']}.png", meta["stem"])

    print("Saved prediction visuals to:", out_dir)

if __name__ == "__main__":
    main()