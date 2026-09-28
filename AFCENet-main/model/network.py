import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbone.mobilenetv2 import mobilenet_v2
from .block.afem import AdjacentFeatureEnhancementModule
from .block.eeam import CrossScaleGatedAggregationModule, ExplicitEdgeAwareModule
from .block.convs import ConvBnRelu, DsBnRelu
from .utils import init_method
from .block.heads import FCNHead, GatedResidualUpHead
from .block.de import AdaptiveWeightFeatureFusionModule
from .block.rcd_head import ResidualConvDecoderHead


def get_backbone(backbone_name):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == 'resnet18d':
        backbone = timm.create_model('resnet18d', pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


def get_afem(afem_name, in_channels, out_channels, deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv'):
    if afem_name == 'afem':
        afem = AdjacentFeatureEnhancementModule(in_channels, out_channels, deform_groups, gamma_mode, beta_mode)
    else:
        raise NotImplementedError("AdjacentFeatureEnhancementModule [%s] is not implemented!\n" % afem_name)
    return afem


class Detector(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', afem_name='afem', afem_channels=128,
                 deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv',
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal'):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        self.afem = get_afem(afem_name, in_channels=self.backbone.channels[-4:], out_channels=afem_channels,
                             deform_groups=deform_groups, gamma_mode=gamma_mode, beta_mode=beta_mode)

        # Replace MCAM (VerticalFusion) with ACFM+RM chain
        self.eeam_p5 = ExplicitEdgeAwareModule(afem_channels)
        self.eeam_p4 = ExplicitEdgeAwareModule(afem_channels)
        self.eeam_p3 = ExplicitEdgeAwareModule(afem_channels)
        self.eeam_p2 = ExplicitEdgeAwareModule(afem_channels)

        self.csga_54 = CrossScaleGatedAggregationModule(channel_L=afem_channels, channel_H=afem_channels)
        self.csga_43 = CrossScaleGatedAggregationModule(channel_L=afem_channels, channel_H=afem_channels)
        self.csga_32 = CrossScaleGatedAggregationModule(channel_L=afem_channels, channel_H=afem_channels)

        # Difference Enhancement (CDEM) with SSIM
        # All layers use SSIM attention (三路并行)
        self.diff_adaptive_weight_feature_fusion_p2 = AdaptiveWeightFeatureFusionModule(128, 128)  # H/4 浅层
        self.diff_adaptive_weight_feature_fusion_p3 = AdaptiveWeightFeatureFusionModule(128, 128)  # H/8 中浅层
        self.diff_adaptive_weight_feature_fusion_p4 = AdaptiveWeightFeatureFusionModule(128, 128)  # H/16 深层
        self.diff_adaptive_weight_feature_fusion_p5 = AdaptiveWeightFeatureFusionModule(128, 128)  # H/32 最深层
        
        self.p5_head = nn.Conv2d(afem_channels, 2, 1)
        self.p4_head = nn.Conv2d(afem_channels, 2, 1)
        self.p3_head = nn.Conv2d(afem_channels, 2, 1)
        self.p2_head = nn.Conv2d(afem_channels, 2, 1)
        self.project = nn.Sequential(nn.Conv2d(afem_channels*4, afem_channels, 1, bias=False),
                                     nn.BatchNorm2d(afem_channels),
                                     nn.ReLU(True)
                                     )
        self.head = ResidualConvDecoderHead(in_channels=afem_channels, num_classes=2)
        # Informative print to confirm decoder head type and TransposedConv usage
        try:
            num_transconv = sum(1 for m in self.head.modules() if isinstance(m, nn.ConvTranspose2d))
            print(f"[DMFANet] Decoder head: {self.head.__class__.__name__} | transposed_convs={num_transconv}")
        except Exception:
            pass
        # init_method(self.afem, self.p5_to_p4, self.p4_to_p3, self.p3_to_p2, self.p5_head, self.p4_head,
        #             self.p3_head, self.p2_head, init_type=init_type)

    def forward(self, x1, x2):
        ### Extract backbone features
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1) # [16, 24, 64, 64], [16, 32, 32, 32]，[16, 96, 16, 16]，[16, 320, 8, 8]
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2) # [16, 24, 64, 64]，[16, 32, 32, 32]，[16, 96, 16, 16]，[16, 320, 8, 8]
        t1_p2, t1_p3, t1_p4, t1_p5 = self.afem([t1_c2, t1_c3, t1_c4, t1_c5]) # [16, 128, 64, 64]，[16, 128, 32, 32]，[16, 128, 16, 16]，[16, 128, 8, 8]
        t2_p2, t2_p3, t2_p4, t2_p5 = self.afem([t2_c2, t2_c3, t2_c4, t2_c5]) # [16, 128, 64, 64]，[16, 128, 32, 32]，[16, 128, 16, 16]，[16, 128, 8, 8]

        diff_p2 = self.diff_adaptive_weight_feature_fusion_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_adaptive_weight_feature_fusion_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_adaptive_weight_feature_fusion_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_adaptive_weight_feature_fusion_p5(t1_p5, t2_p5)
        
        fea_p5 = self.eeam_p5(diff_p5)#Di(enc)
        pred_p5 = self.p5_head(fea_p5)#辅助预测图，由Di(enc)调整通道得到
        fea_p4 = self.eeam_p4(self.csga_54(diff_p4, fea_p5))
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.eeam_p3(self.csga_43(diff_p3, fea_p4))
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.eeam_p2(self.csga_32(diff_p2, fea_p3))
        pred_p2 = self.p2_head(fea_p2)
        pred = self.head(fea_p2, fea_p3, fea_p4, fea_p5)

        pred_p2 = F.interpolate(pred_p2, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p3 = F.interpolate(pred_p3, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p4 = F.interpolate(pred_p4, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p5 = F.interpolate(pred_p5, size=(256, 256), mode='bilinear', align_corners=False)

        return pred, pred_p2, pred_p3, pred_p4, pred_p5


if __name__ == '__main__':
    x1 = torch.randn((32, 512, 8, 8))
    x2 = torch.randn((32, 512, 8, 8))
    model = Detector(512, 512)
    out = model(x1, x2)
    print(out.shape)
