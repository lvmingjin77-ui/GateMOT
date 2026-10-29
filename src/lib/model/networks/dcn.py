"""Checkpoint-compatible modulated DCN using torchvision's packaged operator.

The parameter names and offset layout follow the original DCNv2 implementation.
"""
import math
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class TorchvisionDCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, deformable_groups=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels,in_channels,*self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        bound = 1 / math.sqrt(in_channels*self.kernel_size[0]*self.kernel_size[1])
        nn.init.uniform_(self.weight,-bound,bound)
        self.conv_offset_mask = nn.Conv2d(in_channels,
            deformable_groups*3*self.kernel_size[0]*self.kernel_size[1],
            self.kernel_size,self.stride,self.padding)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x):
        o1,o2,mask = self.conv_offset_mask(x).chunk(3,dim=1)
        return deform_conv2d(x,torch.cat((o1,o2),dim=1),self.weight,self.bias,
                             self.stride,self.padding,self.dilation,mask.sigmoid())


DCN = TorchvisionDCN
