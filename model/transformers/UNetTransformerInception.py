import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=5, padding=2)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        pool = self.pool(x)
        out = torch.cat([conv1, conv3, conv5, pool], dim=1)
        out = self.bn(out)
        out = F.relu(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = nn.Conv3d(in_channels + skip_channels,
                              out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class UNetTransformerInception(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.rescale = nn.Sequential(nn.BatchNorm3d(
            in_channels), nn.ReLU(), nn.Conv3d(in_channels, 3, kernel_size=1))

        # UNet encoding path with Inception blocks
        self.inc1 = InceptionBlock(3, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.inc2 = InceptionBlock(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.inc3 = InceptionBlock(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.inc4 = InceptionBlock(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Transformer
        resnet = resnet18(pretrained=True)
        self.transformer = nn.Sequential(*list(resnet.children())[:-2])

        # UNet decoding path with skip connections and transformer output
        self.up5 = UpsampleBlock(256, 128, 256)
        self.up6 = UpsampleBlock(256, 64, 128)
        self.up7 = UpsampleBlock(128, 32, 64)
        self.up8 = UpsampleBlock(64, 32, 32)

        # Output layer with sigmoid activation for multilabel segmentation
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Rescale input to 3 channels
        x = self.rescale(x)

        # Encoding path
        x1 = self.inc1(x)
        x = self.pool1(x1)
        x2 = self.inc2(x)
        x = self.pool2(x2)
        x3 = self.inc3(x)
        x = self.pool3(x3)
        x4 = self.inc4(x)
        x = self.pool4(x4)

        # Transformer
        x = self.transformer(x)

        # Decoding path with skip connections
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)

        # Output layer
        x = self.out(x)
        x = self.sigmoid(x)

        return x
