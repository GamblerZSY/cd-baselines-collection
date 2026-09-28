import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CrossScaleGatedAggregationModule(nn.Module):

    def __init__(self, channel_L: int, channel_H: int):
        super(CrossScaleGatedAggregationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_H, out_channels=channel_L, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)

        self.gate = nn.Sequential(
            nn.Conv2d(channel_L, channel_L, kernel_size=3, stride=1, padding=1, groups=channel_L, bias=False),
            nn.BatchNorm2d(channel_L),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_L, channel_L, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode='bilinear', align_corners=False)
        f_high = self.relu(self.bn(self.conv1(f_high)))

        f_cat = f_high + f_low
        gate = self.gate(f_cat)
        out = f_low * gate + f_high * (1 - gate)
        out = out + self.residual_weight * f_low
        return out

class ExplicitEdgeAwareModule(nn.Module):

    def __init__(self, mid_d: int):
        super(ExplicitEdgeAwareModule, self).__init__()
        self.cbam = CBAM(channel=mid_d)
        self.edge_conv = nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=1, groups=mid_d, bias=False)#定义深度可分离卷积
        lap = torch.tensor([[0.0, -1.0, 0.0],
                            [-1.0, 4.0, -1.0],
                            [0.0, -1.0, 0.0]], dtype=torch.float32)#定义拉普拉斯算子核
        with torch.no_grad():
            self.edge_conv.weight.zero_()
            for i in range(mid_d):
                self.edge_conv.weight[i, 0].copy_(lap)#复制到每一个通道的卷积核里
        self.edge_conv.weight.requires_grad_(False)#冻结参数
        self.edge_scale = nn.Parameter(torch.zeros(1))#定义可学习的缩放系数
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.cbam(x)
        edge = self.edge_conv(context).abs().mean(dim=1, keepdim=True)
        edge_gate = torch.sigmoid(self.edge_scale * edge)
        context = context * (1 + edge_gate)
        x_out = self.conv2(context)
        return x_out
