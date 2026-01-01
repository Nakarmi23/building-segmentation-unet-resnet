from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.whu_dataset import WHUDataset, build_transforms
from src.models.unet import UNet
from src.losses.combined_loss import BCEDiceLoss
from src.metrics.segmentation import dice_score, iou_score, pixel_accuracy
from src.utils.metric_logger import MetricLogger

def train():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    image_size = (256, 256)
    batch_size = 4
    lr = 1e-4
    epochs = 20
    num_workers = 4

    ckpt_dir = Path("checkpoints/unet")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_tf = build_transforms(train=False, image_size=image_size)
    val_tf = build_transforms(train=False, image_size=image_size)

    train_ds = WHUDataset(
        root_dir="data",
        split="train",
        transform=train_tf
    )


    val_ds = WHUDataset(
        root_dir="data",
        split="val",
        transform=val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = UNet(features=(32, 64, 128, 256)).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0

    logger = MetricLogger("outputs/metrics/unet_metrics.csv")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device,non_blocking=True)
            masks = masks.to(device,non_blocking=True)

            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device,non_blocking=True)
                masks = masks.to(device,non_blocking=True)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item()
                val_dice += dice_score(logits, masks)
                val_iou += iou_score(logits, masks)
                val_acc += pixel_accuracy(logits, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc /= len(val_loader)

        logger.log(epoch, train_loss, val_loss, val_dice, val_iou, val_acc)

        print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Val IoU: {val_iou:.4f} | "
        f"Val Acc: {val_acc:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice
            }, ckpt_dir / "best.pt")

            print(f"Saved best model with IoU: {best_val_iou:.4f}")
            


if __name__ == "__main__":
    train()


