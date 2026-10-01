import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.HRNet2 import hrnet32,hrnet48
import random


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class DynamicGatingFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, 2, 1),
            nn.Sigmoid()
        )
        
        self.transform = self._make_layer(ResBlock, dim * 2, dim, 2, stride=1)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x1,x2):

        
        concat_feat = torch.cat([x1, x2], dim=1)
        gates = self.gate_conv(concat_feat)  # B, 2, H, W
        
        gated_feature1 = x1 * gates[:, 0:1, :, :]
        gated_feature2 = x2 * gates[:, 1:2, :, :]
        
        fused = gated_feature1 + gated_feature2
        transformed = self.transform(concat_feat)
        
        return fused + transformed
    

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GUIDE(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(GUIDE, self).__init__()

        #self.FCN = FCN(in_channels, pretrained=True)
        self.hrnet = hrnet48(pretrained=True)
        self.classifier1 = nn.Conv2d(96, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(96, num_classes, kernel_size=1)
        self.conv1 = nn.Sequential(nn.Conv2d(336, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(144, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(256, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU())
        # self.conv4 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU())
        self.resCD1 = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        # self.resCD2 = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        self.resCD2_ = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        # self.resCD3 = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        self.resCD3_ = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        # self.resCD4 = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        # self.resCD4_ = self._make_layer(ResBlock, 192, 96, 2, stride=1)
        self.dgf2 = DynamicGatingFusion(96)
        self.dgf3 = DynamicGatingFusion(96)

        self.proj = nn.Sequential(nn.Conv2d(96, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU(), nn.Conv2d(96, 96, kernel_size=1))
        self.classifierCD = nn.Sequential(nn.Conv2d(96, 48, kernel_size=1), nn.BatchNorm2d(48), nn.ReLU(), nn.Conv2d(48, 1, kernel_size=1))    
        initialize_weights(self.classifier1, self.classifier2, self.resCD1, self.dgf2, self.resCD2_, self.dgf3, self.resCD3_, self.classifierCD,self.conv1,self.conv2,self.conv3,self.proj)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        
        # x = self.FCN.layer0(x) #size:1/4
        # x = self.FCN.maxpool(x) #size:1/4
        # x = self.FCN.layer1(x) #size:1/4
        # x = self.FCN.layer2(x) #size:1/8
        # x = self.FCN.layer3(x) #size:1/16
        # x = self.FCN.layer4(x)
        # x = self.FCN.head(x)
        x3_1,x3_2,x3_3,x2_1,x2_2,x1_1,x0_1 = self.hrnet(x)
        """ print("x1_size:",x1.size())#128
        print("x2_size:",x2.size())#64 c=128
        print("x3_size:",x3.size())#32
        print("x4_size:",x4.size())#16   """ 
        return x3_1,x3_2,x3_3,x2_1,x2_2,x1_1,x0_1
    
    def CD_forward(self, x1, x2, x1_bcd_2, x2_bcd_2, x1_bcd_3, x2_bcd_3):
        
        #SFCM
        # 1) stage-1
        x = self.resCD1(torch.cat([x1, x2], dim=1))

        # 2) stage-2
        x_bcd_2 = self.resCD2_(torch.cat([x1_bcd_2, x2_bcd_2], dim=1))
        x = self.dgf2(x,x_bcd_2)

        # 3) stage-3
        x_bcd_3 = self.resCD3_(torch.cat([x1_bcd_3, x2_bcd_3], dim=1))
        x = self.dgf3(x,x_bcd_3)

        # 4) stage-4 (关键：用目标分支尺寸做对齐)
        # x_bcd_4 = self.resCD4_(torch.cat([x1_bcd_4, x2_bcd_4], dim=1))
        # x = F.interpolate(x, size=x_bcd_4.shape[-2:], mode="bilinear", align_corners=False)
        # x = self.resCD4(torch.cat([x, x_bcd_4], dim=1))

        change = self.classifierCD(x)
        return change

    
    def forward(self, x1, x2,label = None):
        x_size = x1.size()
        H, W = x_size[2], x_size[3]

        x1_1,x1_2,x1_3, x1_4,x1_5, x1_6, x1_7 = self.base_forward(x1)
        x2_1,x2_2,x2_3, x2_4,x2_5, x2_6, x2_7 = self.base_forward(x2)

        target_size = x1_1.shape[-2:]  

        x1_2 = F.interpolate(x1_2, size=target_size, mode="bilinear", align_corners=False)
        x1_3 = F.interpolate(x1_3, size=target_size, mode="bilinear", align_corners=False)
        x2_2 = F.interpolate(x2_2, size=target_size, mode="bilinear", align_corners=False)
        x2_3 = F.interpolate(x2_3, size=target_size, mode="bilinear", align_corners=False)

        x1 = self.conv1(torch.cat([x1_1, x1_2, x1_3], 1))
        x2 = self.conv1(torch.cat([x2_1, x2_2, x2_3], 1))  

        x1_5 = F.interpolate(x1_5, size=target_size, mode="bilinear", align_corners=False)
        x2_5 = F.interpolate(x2_5, size=target_size, mode="bilinear", align_corners=False)

        x1_bcd_2 = self.conv2(torch.cat([x1_4, x1_5], 1))
        x2_bcd_2 = self.conv2(torch.cat([x2_4, x2_5], 1))

        x1_bcd_3 = self.conv3(x1_6)
        x2_bcd_3 = self.conv3(x2_6)

        # x1_bcd_4 = self.conv4(x1_7)
        # x2_bcd_4 = self.conv4(x2_7)


        proj1 = self.proj(x1)
        proj2 = self.proj(x2)


        change = self.CD_forward(x1, x2, x1_bcd_2, x2_bcd_2, x1_bcd_3, x2_bcd_3)

        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)

        return (
            F.interpolate(change, size=x_size[2:], mode="bilinear", align_corners=False),out1, out2 ,proj1,proj2
        )
