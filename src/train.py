import torch
from torch.utils.data import DataLoader

from src.datasets.whu_dataset import WHUDataset, build_transforms
from src.models.unet import UNet
from src.losses.combined_loss import BCEDiceLoss

def train_one_batch_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = build_transforms(train=True, image_size=(512, 512))

    dataset = WHUDataset(
        root_dir="data",
        split="train",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)

    model = UNet().to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for step in range(500):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(f"Step [{step}/500], Loss: {loss.item():.4f}")

    print("Overfitting on single batch completed.")


if __name__ == "__main__":
    train_one_batch_overfit()


