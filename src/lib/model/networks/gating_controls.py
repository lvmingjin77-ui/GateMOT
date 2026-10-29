"""Explicit control definitions. These are not claimed completed experiments.

Both controls use BatchNorm after gating. GLU uses 1x1 projections;
gated_conv uses 3x3 projections. They have different parameter/FLOP counts.
"""
from torch import nn


class GatingControl(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.feature = nn.Conv2d(in_channels,out_channels,kernel_size,padding=kernel_size//2)
        self.gate = nn.Conv2d(in_channels,out_channels,kernel_size,padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.feature(x)*self.gate(x).sigmoid())
