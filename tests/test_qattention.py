"""Shape and path checks for Q-Attention. No dataset and no DCNv2."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "lib"))
from model.networks.base_model import QAttention


def main():
    layer = QAttention(64, 256)
    features = torch.randn(2, 64, 20, 24)
    output = layer(features)
    assert output.shape == (2, 256, 20, 24), output.shape

    value = layer.conv1(features)
    key = layer.conv2(features)
    gate = torch.sigmoid(layer.conv3(features))
    pooled = torch.nn.functional.max_pool2d(key * gate, kernel_size=3, stride=1, padding=1)
    fused = layer.bn4(layer.conv4(torch.cat([value, pooled], dim=1)))
    assert torch.allclose(output, fused)

    layer.conv1.weight.data.zero_()
    without_value = layer(features)
    layer.conv1.weight.data.normal_()
    with_value = layer(features)
    assert not torch.allclose(without_value, with_value), "Value must change the output"
    print("QAttention ok", tuple(output.shape))


if __name__ == "__main__":
    main()
