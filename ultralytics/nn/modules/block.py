# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""
import math
import warnings
from typing import Optional
import torchvision.models as models
from timm.models.layers import DropPath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from utils.general import make_divisible
from utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DWConv, GhostConv, GSConv, LightConv, RepConv, autopad, PConv, PConv1
from .transformer import TransformerBlock



__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'C2f_DCN', 'C2f_SAT', 'MCSPFF','LWPF','CARAFE','C2f_CondConv',
           'C2f_LSKB_DCN', 'C2f_SAT_GS', 'C2f_GS','CC2f','BasicStage', 'PatchEmbed_FasterNet', 'PatchMerging_FasterNet',
           'C2f_SE', 'C2f_Swin', 'VoVGSCSP', 'VoVGSCSPC','SwinTransformerBlock','C2fCIB','MC2fCIB',
           'SPPF_SE', 'CST', 'SPPF_SAT', 'SPPF_LSKB', 'SimCSPSPPF', 'ESimSPP2FE','PESimSPP2FE','PESimSPP2FE2','PESimSPP2FE3','CAPCE','MobileNetV3','CSAT', 'MCS','C2fSTR','C3STR',
           'Silence','CBFuse','CBLinear')

from ..ShuffleAttention import ShuffleAttention
# from ..SwinTransformer import SwinTransformerLayer
from ..LSKBlock import LSKBlock
from ..evc_blocks import EVCBlock,LightMLPBlock


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x



class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))

class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPF_SAT(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # 将普通Conv换成SimConv
        self.cv1 = SimConv(c1, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, c2, 1, 1)
        # self.cv1 = GSConv(c1, c_, 1, 1)
        # self.cv2 = GSConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # Create an instance of ShuffleAttention
        self.sa = ShuffleAttention(channel=c_)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            # Use ShuffleAttention on x
            x = self.sa(x)
            y3 = self.sa(y3)
            y4 = torch.cat((x, y1, y2, y3), 1)
            return self.cv2(y4)


# class SPPF_SAT(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
#
#     def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 5, c2, 1, 1)
#         #将普通Conv换成SimConv
#         # self.cv1 = SimConv(c1, c_, 1, 1)
#         # self.cv2 = SimConv(c_ * 4, c2, 1, 1)
#         # self.cv1 = GSConv(c1, c_, 1, 1)
#         # self.cv2 = GSConv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         # Create an instance of ShuffleAttention
#         self.sa = ShuffleAttention(channel=c_)
#
#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         x = self.cv1(x)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             y1 = self.m(x)
#             y2 = self.m(y1)
#             y3 = self.m(y2)
#             # Use ShuffleAttention on x
#             x = self.sa(x)
#             sa1 = self.sa(y3)
#             y4 = torch.cat((x, y1, y2, y3,sa1), 1)
#             return self.cv2(y4)

class SPPF_LSKB(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # 将普通Conv换成SimConv
        # self.cv1 = SimConv(c1, c_, 1, 1)
        # self.cv2 = SimConv(c_ * 4, c2, 1, 1)
        # self.cv1 = GSConv(c1, c_, 1, 1)
        # self.cv2 = GSConv(c_ * 4, c2, 1, 1)
        self.m = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # Create an instance of ShuffleAttention
        self.lskb = LSKBlock(c_)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            # Use ShuffleAttention on x
            x = self.lskb(x)
            y3 = self.lskb(y3)
            y4 = torch.cat((x, y1, y2, y3), 1)
            return self.cv2(y4)


class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ESimSPP2FE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(ESimSPP2FE, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = SimConv(c1, c_, 1, 1)
        self.cv2 = SimConv(c1, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(3 * c_, c2, 1, 1)
        self.cv8 = SimConv(c_,c2,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        x4 = self.m(x3)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))
        s1 = self.sigmoid(y1)
        y2 = self.cv2(x)
        y3 = y2 * s1
        result = y3 + y2
        return self.cv8(result)



class PESimSPP2FE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(PESimSPP2FE, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = SimConv(c1, c_, 1, 1)
        self.cv2 = SimConv(c1, c_, 1, 1)
        self.cv3 = PConv(c_, 3, 4, 'split_cat')
        self.cv4 = SimConv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = PConv(c_, 3, 4, 'split_cat')
        # self.cv7 = SimConv(3 * c_, c2, 1, 1)
        self.cv8 = SimConv(c_,c2,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        x4 = self.m(x3)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))
        s1 = self.sigmoid(y1)
        y2 = self.cv2(x)
        y3 = y2 * s1
        result = y3 + y2
        return self.cv8(result)


class PESimSPP2FE2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(PESimSPP2FE2, self).__init__()
        c_ = c1 // 4
        self.cv1 = Conv(c1 // 4,c2 // 4, 1, 1)
        self.cv2 = Conv(c_,c_, 3, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = Conv(c1,c_,1,1)
        self.cv4 = PConv(c_, 3, 4, 'split_cat')
        self.cv5 = Conv(2*c_,4*c_,1,1)

    def forward(self, x):
        Xsplit = x.chunk(4,1)
        x0 = Xsplit[0]
        x1 = Xsplit[1]
        x2 = Xsplit[2]
        x3 = Xsplit[3]
    #  分支一
        xx0 = self.cv2(self.cv1(x0))
        xx01 = self.cv1(self.cv2(self.cv1(xx0)))
        xx02 = xx0 + xx01
        xx03 = self.cv1(self.cv2(self.cv1(xx02)))
        xx04 = xx02 + xx03
    #  分支二
        xx1 = self.cv1(self.cv2(self.cv1(x1)))
        z1 = xx04 + xx1
    #  分支三
        xx2 = self.cv1(self.cv2(self.cv1(x2)))
    #分支四
        xx3 = self.cv2(self.cv1(x3))
        xx31 = self.m(xx3)
        xx32 = self.m(xx31)
        xx33 = self.m(xx32)
        xx34 = self.cv4(self.cv3(torch.cat((xx3, xx31, xx32, xx33), 1)))
        z2 = xx34 + xx2
        return self.cv5(torch.cat((z1,z2),1))

class PESimSPP2FE3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(PESimSPP2FE3, self).__init__()
        c_ = c1 // 4
        self.cv1 = Conv(c1 // 4,c2 // 4, 1, 1)
        self.cv2 = Conv(c_,c_, 3, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = Conv(c1,c_,1,1)
        self.cv4 = PConv(c_, 3, 4, 'split_cat')
        self.cv5 = Conv(2*c_,4*c_,1,1)

    def forward(self, x):
        Xsplit = x.chunk(4,1)
        x0 = Xsplit[0]
        x1 = Xsplit[1]
        x2 = Xsplit[2]
        x3 = Xsplit[3]
    #  分支一
        xx0 = self.cv4(self.cv1(x0))
        xx01 = self.cv1(self.cv2(self.cv1(xx0)))
        xx02 = xx0 + xx01
        xx03 = self.cv1(self.cv2(self.cv1(xx02)))
        xx04 = xx02 + xx03
    #  分支二
        xx1 = self.cv1(self.cv4(self.cv1(x1)))
        z1 = xx04 + xx1
    #  分支三
        xx2 = self.cv1(self.cv4(self.cv1(x2)))
    #分支四
        xx3 = self.cv4(self.cv1(x3))
        xx31 = self.m(xx3)
        xx32 = self.m(xx31)
        xx33 = self.m(xx32)
        xx34 = self.cv4(self.cv3(torch.cat((xx3, xx31, xx32, xx33), 1)))
        z2 = xx34 + xx2
        return self.cv5(torch.cat((z1,z2),1))

class CAPCE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(CAPCE, self).__init__()
        c_ = c1 // 4
        self.cv1 = Conv(c_,c_, 1, 1)
        self.cv2 = Conv(c_,c_, 3, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = Conv(c1,c_,1,1)
        self.cv5 = Conv(2*c_,4*c_,1,1)

    def forward(self, x):
        Xsplit = x.chunk(4,1)
        x0 = Xsplit[0]
        x1 = Xsplit[1]
        x2 = Xsplit[2]
        x3 = Xsplit[3]
    #  分支一
        xx0 = self.cv2(self.cv1(x0))
        xx01 = self.cv1(self.cv2(self.cv1(xx0)))
        xx02 = xx0 + xx01
        xx03 = self.cv1(self.cv2(self.cv1(xx02)))
        xx04 = xx02 + xx03
    #  分支二
        xx1 = self.cv1(self.cv2(self.cv1(x1)))
        z1 = xx04 + xx1
    #  分支三
        xx2 = self.cv1(self.cv2(self.cv1(x2)))
    #分支四
        xx3 = self.cv2(self.cv1(x3))
        xx31 = self.m(xx3)
        xx32 = self.m(xx31)
        xx33 = self.m(xx32)
        xx34 = self.cv2(self.cv3(torch.cat((xx3, xx31, xx32, xx33), 1)))
        z2 = xx34 + xx2
        return self.cv5(torch.cat((z1,z2),1))


class MobileNetV3(nn.Module):
    def __init__(self, slice):
        super(MobileNetV3, self).__init__()
        self.model  = None
        if slice == 1:
            self.model = models.mobilenet_v3_small(pretrained=True).features[:2]
        elif slice == 2:
            self.model = models.mobilenet_v3_small(pretrained=True).features[2:4]
        elif slice == 3:
            self.model = models.mobilenet_v3_small(pretrained=True).features[4:9]
        else:
            self.model = models.mobilenet_v3_small(pretrained=True).features[9:]
    def forward(self,x):
        return self.model(x);

class SimCSPSPPF(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SimCSPSPPF, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = SimConv(c1, c_, 1, 1)
        self.cv2 = SimConv(c1, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        x4 = self.m(x3)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class CST(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = SwinTransformerBlock(c_, c_, c_ // 32, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF_SE(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF_SE, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.se = SELayer(c_ * 4, reduction=16)  # Adding SENet's SE module

    def forward(self, x):
        """Forward pass through SPPF layer."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        x = self.se(x)  # Applying SENet's SE module
        y3 = self.se(self.m(y2))
        out = torch.cat((x, y1, y2, y3), 1)
        return self.cv2(out)


class CSAT(nn.Module):  # CNN-ShuffleAttention
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        # self.m = SwinTransformerBlock(c_, c_, c_//32,1)
        self.swin = ShuffleAttention(channel=c_)

    def forward(self, x):
        return self.cv3(torch.cat((self.swin(self.cv1(x)), self.cv2(x)), dim=1))


class DPUP(nn.Module):
    def __init__(self, c1, kernal=(16, 16), scale=1.25):
        super().__init__()
        self.dp = nn.AdaptiveAvgPool2d(kernal[0])
        self.kernal = kernal
        self.conv = Conv(c1, 256, k=1)
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear')

    def forward(self, x):
        x = self.dp(x)
        x = self.conv(x)
        x = self.up(x)
        return x


class DPUP(nn.Module):
    def __init__(self, c1, kernal=(16, 16), scale=1.25):
        super().__init__()
        self.dp = nn.AdaptiveAvgPool2d(kernal[0])
        self.kernal = kernal
        self.conv = Conv(c1, 256, k=1)
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear')

    def forward(self, x):
        x = self.dp(x)
        x = self.conv(x)
        x = self.up(x)
        return x


class MCS(nn.Module):
    def __init__(self, c1, c2=0, k=0):
        super().__init__()
        self.dp1 = DPUP(c1, kernal=(16, 16), scale=1.25)
        self.dp2 = DPUP(c1, kernal=(10, 10), scale=2)
        self.dp3 = DPUP(c1, kernal=(8, 8), scale=2.5)
        self.dp4 = DPUP(c1, kernal=(5, 5), scale=4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv6 = Conv(256, c2, k=1)

    def forward(self, x):
        residual = x
        dep1 = self.dp1(x)
        dep2 = self.dp2(x)
        dep3 = self.dp3(x)
        dep4 = self.dp4(x)
        out = torch.cat([dep1, dep2, dep3, dep4], dim=1)
        y = self.avg_pool(out)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)
        out2 = x * y.expand_as(x)

        result = out2[:, :256, ...] + out2[:, 256:512, ...] + out2[:, 512:768, ...] + out2[:, 768:1024, ...]
        result = self.conv6(result) + residual
        return result


# class MCS(nn.Module):
#     def __init__(self,c1,c2=0,k=0):
#         super().__init__()
#         # 关闭确定性算法
#         # torch.use_deterministic_algorithms(False)
#         self.dp1 = DPUP(c1,kernal=(16,16),scale=1.25)
#         self.dp2 = DPUP(c1,kernal=(10,10),scale=2)
#         self.dp3 = DPUP(c1,kernal=(8,8),scale=2.5)
#         self.dp4 = DPUP(c1,kernal=(5,5),scale=4)
#
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#         #用于调整输入图片的规格
#         self.dp = nn.AdaptiveMaxPool2d((20,20))
#         self.reform = Conv(c1,1024,k=1)
#         self.reform1 = Conv(1024,512,k=1)
#
#         self.conv6 = Conv(1024, 576, k=1)
#         # self.sa = ShuffleAttention(c1)
#
#     def forward(self,x):
#         residual = x
#         # x = self.reform(x)
#         dep1 = self.dp1(x)
#         dep2 = self.dp2(x)
#         dep3 = self.dp3(x)
#         dep4 = self.dp4(x)
#         out = torch.cat([dep1,dep2,dep3,dep4], dim=1)
#         y = self.avg_pool(out)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         y = self.sigmoid(y)
#         print("y.shape1:",y.shape)
#
#         x = self.dp(x)
#         y = self.dp(y)
#         x = self.reform(x)
#         # y = self.reform1(y)
#         print("x.shape:",x.shape)
#         print("y.shape2:",y.shape)
#         # y = self.reform1(y)
#         y = y.expand_as(x)
#
#         out2 = x * y
#         print("out2:",out2.shape)
#         # result = self.sa(out2)
#         # result = out2[:,:256,...] + out2[:,256:512,...] + out2[:,512:768,...]+ out2[:,768:1024,...]
#         result = self.conv6(out2)
#         # result = self.conv6(result) + residual
#         print("result:",result.shape)
#         return result


class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels_p1, out_channels_p2_1, out_channels_p2_2, out_channels_final):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels_p1, kernel_size=1)
        self.conv2_1 = nn.Conv2d(out_channels_p1, out_channels_p2_1, kernel_size=1)
        self.conv2_2 = nn.Conv2d(out_channels_p1, out_channels_p2_2, kernel_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=8)
        self.upsample1 = nn.Upsample(scale_factor=1.25, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = nn.Concat(dim=1)
        self.final_conv = nn.Conv2d(out_channels_p2_1 + out_channels_p2_2 + out_channels_p1, out_channels_final,
                                    kernel_size=1)

    def forward(self, x):
        p1 = self.conv1(x)

        p1_1, p1_2 = p1.chunk(2, dim=1)

        p2_1 = self.maxpool1(p1_1)
        p2_1 = self.conv2_1(p2_1)
        p2_1 = self.upsample1(p2_1)

        p2_2 = self.maxpool2(p1_2)
        p2_2 = self.conv2_2(p2_2)
        p2_2 = self.upsample2(p2_2)

        p2 = self.concat([p2_1, p2_2])

        p3 = self.maxpool1(p2)
        output = self.concat([p3, p1])
        output = self.final_conv(output)

        return output


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,
                                                       shift_size=0 if (i % 2 == 0) else self.shift_size) for i in
                                  range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.tr(x)
        return x


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(ERFFE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class CC2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))
class C2f_Swin(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.swin = SwinTransformerBlock(self.c, self.c, self.c // 32, 1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        return self.cv2(self.swin(y))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        return self.cv2(self.swin(y))


class C2f_SE(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        # Add SENet SE Module
        self.se = SELayer((2 + n) * self.c)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.se(y)  # Apply SENet SE Module
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.se(y)  # Apply SENet SE Module
        return self.cv2(y)


class SELayer(nn.Module):
    """SENet SE Module."""

    def __init__(self, channels=512, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class C2f_SAT(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.shuffle_attention = ShuffleAttention((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)


class MCSPFF(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.shuffle_attention = ShuffleAttention((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ERFFE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)


class LWPF(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.shuffle_attention = ShuffleAttention((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(LWB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)


class ERFFE(nn.Module):
    # Standard bottleneck with DCN
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        #******************可变形卷积操作开始********************
        if k[0] == 3:
            self.cv1 = DCNv2(c1, c_, k[0], 1)
        else:
            self.cv1 = Conv(c1, c_, k[0], 1)
        if k[1] == 3:
            self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # ******************可变形卷积操作结束********************
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class LWB(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = PConv(c1, 3, 4, 'split_cat')
        self.cv2 = PConv(c_, 3, 4, 'split_cat')
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


'''
    https://arxiv.org/abs/1905.02188
    c:输入通道数
    scale:上采样扩大尺寸倍数，h*w -> (h*scale)*(w*scale)
'''
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X




class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class C2f_LSKB_DCN(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.lskb = LSKBlock((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ERFFE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.lskb(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.lskb(y)  # Apply ShuffleAttention
        return self.cv2(y)


class C2f_SAT_GS(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.shuffle_attention = ShuffleAttention((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GSBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        # self.gsb = GSBottleneck(c_, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)


class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)


class C2f_GS(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # self.shuffle_attention = ShuffleAttention((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GSBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        # y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        # y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()





class C2f_DCN(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ERFFE(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fSTR(C2f):
    # C2f module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)

class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4 * c, bias=False)
        self.fc2 = nn.Linear(4 * c, c, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x_ = self.ln1(x)
        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x
        x_ = self.ln2(x)
        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath1(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath1, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerLayer(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads > 10:
            drop_path = 0.1
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)
        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(in_features=c, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = ((0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.tensor(-100.0)).masked_fill(attn_mask == 0,
                                                                                            torch.tensor(0.0))
        return attn_mask

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()  # [b,h,w,c]

        attn_mask = self.create_mask(x, h, w)  # [nW, Mh*Mw, Mh*Mw]
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            # print(f"shift size: {self.shift_size}")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 2, 1).contiguous()
        return x  # (b, self.c2, w, h)


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)


class CondConv2D(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size) # 3 -> (3,3)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)

# class C3_CondConv(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, num_experts=3, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = CondConv2D(c1, c_, num_experts=num_experts, kernel_size=1, stride=1)
#         self.cv2 = CondConv2D(c1, c_, num_experts=num_experts, kernel_size=1, stride=1)
#         self.cv3 = CondConv2D(2 * c_, c2, num_experts=num_experts, kernel_size=1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# class C2f_CondConv(nn.Module):
#     """CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2,num_experts, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = CondConv2D(c1, 2 * self.c,num_experts=num_experts, kernel_size=1, stride=1)
#         self.cv2 = CondConv2D((2 + n) * self.c, c2, num_experts=num_experts, kernel_size=1, stride=1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         # y.extend(m(y[-1]) for m in self.m)
#         for m in self.m:
#             y.append(m(y[-1]))
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         # y.extend(m(y[-1]) for m in self.m)
#         for m in self.m:
#             y.append(m(y[-1]))
#         return self.cv2(torch.cat(y, 1))

class C2f_CondConv(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    # c2f with ShuffleAttention

    def __init__(self, c1, c2,num_experts, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CondConv2D(c1, 2 * self.c,num_experts=num_experts, kernel_size=1, stride=1)
        self.shuffle_attention = ShuffleAttention((2 + n) * self.c)  # Instantiate ShuffleAttention module
        self.cv2 = CondConv2D((2 + n) * self.c, c2, num_experts=num_experts, kernel_size=1, stride=1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ERFFE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.shuffle_attention(y)  # Apply ShuffleAttention
        return self.cv2(y)



class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 depth=1,
                 n_div=4,
                 mlp_ratio=2,
                 layer_scale_init_value=0,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        dpr = [x.item()
               for x in torch.linspace(0, 0.0, sum([1, 2, 8, 2]))]
        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.blocks(x)
        return x


class PatchEmbed_FasterNet(nn.Module):

    def __init__(self, in_chans, embed_dim, patch_size, patch_stride, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x

    def fuseforward(self, x):
        x = self.proj(x)
        return x


class PatchMerging_FasterNet(nn.Module):

    def __init__(self, dim, out_dim, k, patch_stride2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=k, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(out_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.reduction(x))
        return x

    def fuseforward(self, x):
        x = self.reduction(x)
        return x

# -----------------------------------
class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class MC2fCIB(MCSPFF):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))



class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)


class PCIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
            PConv(c2, 3, 4, 'split_cat')
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert (c1 == c2)    #c1=c2=1024
        self.c = int(c1 * e)    #self.c=512
        # self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv1 = PConv(c1,3, 4, 'split_cat')
        # self.cv2 = Conv(2 * self.c, c1, 1)
        self.cv2 = PConv(c1,3, 4, 'split_cat')

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):
        return x


##### CBNet #####

class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

# class CBFuse(nn.Module):
#     def __init__(self, idx):
#         super(CBFuse, self).__init__()
#         self.idx = idx
#
#     def forward(self, xs):
#         target_size = xs[-1].shape[2:]
#         res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
#         out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
#         return out


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        target_channels = xs[-1].shape[1]
        res = [torch.nn.functional.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        res = [torch.nn.functional.pad(r, (0, 0, 0, 0, 0, target_channels - r.shape[1])) for r in res]  # pad to match the channels
        out = torch.sum(torch.stack(res + [xs[-1]]), dim=0)
        # print(out.shape)
        return out


#################
