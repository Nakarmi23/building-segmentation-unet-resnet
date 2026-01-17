import torch
import torch.nn as nn

from models.unet_blocks import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        self.up3 = Up(features[3]*2 + features[3], features[3])
        self.up2 = Up(features[3] + features[2], features[2])
        self.up1 = Up(features[2] + features[1], features[1])
        self.up0 = Up(features[1] + features[0], features[0])

        self.outc = OutConv(features[0], out_channels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_bottleneck = self.bottleneck(x4)

        x = self.up3(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)

        output = self.outc(x)
        return output
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet().to(device)
    x = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)