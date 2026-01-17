import torch
import torch.nn as nn

from src.models.resunet_blocks import ResidualConv, Down, Up, OutConv


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()

        # Encoder
        self.inc = ResidualConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        self.bottleneck = ResidualConv(features[3], features[3] * 2)

        # Decoder (FIXED)
        self.up3 = Up(in_channels=features[3] * 2, skip_channels=features[3], out_channels=features[3])
        self.up2 = Up(in_channels=features[3],     skip_channels=features[2], out_channels=features[2])
        self.up1 = Up(in_channels=features[2],     skip_channels=features[1], out_channels=features[1])
        self.up0 = Up(in_channels=features[1],     skip_channels=features[0], out_channels=features[0])

        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512

        x_bottleneck = self.bottleneck(x4)  # 1024

        x = self.up3(x_bottleneck, x4)  # -> 512
        x = self.up2(x, x3)             # -> 256
        x = self.up1(x, x2)             # -> 128
        x = self.up0(x, x1)             # -> 64

        return self.outc(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResUNet().to(device)
    x = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
