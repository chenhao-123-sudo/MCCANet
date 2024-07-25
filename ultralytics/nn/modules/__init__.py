# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, C2f_DCN, C2f_SE, C2f_Swin, C2f_SAT, MCSPFF,LWPF, C2f_CondConv, CARAFE,
                    C2f_LSKB_DCN, C2f_SAT_GS, C2f_GS,C2fCIB,PSA,MC2fCIB,
                    BasicStage, PatchEmbed_FasterNet, PatchMerging_FasterNet,
                    VoVGSCSP, VoVGSCSPC, SPPF_SE, CST, SPPF_SAT, SPPF_LSKB, SimCSPSPPF, ESimSPP2FE,PESimSPP2FE, PESimSPP2FE2,PESimSPP2FE3,CAPCE,CSAT, CST, MCS,
                    C2fSTR, C3STR,ResNetLayer,Silence,CBLinear,CBFuse,
                    SwinTransformerBlock, CC2f, MobileNetV3)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   MyConcat4,
                   GhostConv, GSConv, LightConv, RepConv, SpatialAttention, SPDConv, Shift8Conv, ChenConv, PConv,SCDown)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment, PDetect,v10Detect
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .VanillaNet import Block as VanillaBlock

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'GSConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'PDetect','v10Detect', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'C2f_DCN',
           'C2f_SAT', 'MCSPFF','LWPF', 'C2f_CondConv', 'CARAFE', 'C2f_LSKB_DCN', 'C2f_SE', 'C2f_Swin','C2fCIB','PSA','MC2fCIB',
           'SPPF_SE', 'SPPF_SAT', 'SPPF_LSKB', 'SimCSPSPPF', 'ESimSPP2FE','PESimSPP2FE','PESimSPP2FE2', 'PESimSPP2FE3','CAPCE','CST', 'MyConcat4', 'C2fSTR', 'C3STR',
           'SwinTransformerBlock', 'CC2f', 'VanillaBlock', 'SPDConv', 'ChenConv', 'PConv', 'Shift8Conv',
           'MobileNetV3', 'BasicStage', 'PatchEmbed_FasterNet', 'PatchMerging_FasterNet','ResNetLayer','SCDown','Silence','CBFuse','CBLinear')
