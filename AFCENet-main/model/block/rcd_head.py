import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv2d(x)

class UpsampleConvLayer(nn.Module):
    """Transpose convolution layer to upsample the feature maps"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv2d(x)

class ResidualBlock(nn.Module):
    """Residual 3x3 conv block used after each upsampling"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = out + residual
        return out

class ResidualConvDecoderHead(nn.Module):
    """
    Residual Convolutional Decoder head.
    - Inputs: four-scale fused features (p2, p3, p4, p5) with the same channel size.
    - Align p3/p4/p5 to p2's spatial size, concatenate, 1x1 reduce channels.
    - Two stages of TransposedConv upsampling, each followed by a residual block.
    - Final 3x3 Conv to num_classes.
    """
    def __init__(self, in_channels=128, num_classes=2, embed_channels=None):
        super(ResidualConvDecoderHead, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels or in_channels
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, self.embed_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.embed_channels),
            nn.ReLU(inplace=True),
        )
        self.up2x_1 = UpsampleConvLayer(self.embed_channels, self.embed_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(self.embed_channels)
        self.up2x_2 = UpsampleConvLayer(self.embed_channels, self.embed_channels, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualBlock(self.embed_channels)
        self.classifier = ConvLayer(self.embed_channels, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, f2, f3, f4, f5):
        # align to f2 spatial size
        size = f2.shape[-2:]
        f3_up = F.interpolate(f3, size=size, mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=size, mode='bilinear', align_corners=False)
        f5_up = F.interpolate(f5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([f2, f3_up, f4_up, f5_up], dim=1)
        x = self.reduce(x)
        x = self.up2x_1(x)
        x = self.res1(x)
        x = self.up2x_2(x)
        x = self.res2(x)
        out = self.classifier(x)
        return out
