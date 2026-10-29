from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from loguru import logger
from .gating_controls import GatingControl
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class QAttention(nn.Module):
  """Q-Gated Attention for one dense task head.

  Three 1x1 projections implement the paper's W_V, W_K and W_Q.
  Attribute names conv1, conv2, conv3 and conv4 are the checkpoint keys:
  conv1 is Value, conv2 is Key, conv3 is Query, conv4+bn4 is the fusion.
  Query is a sigmoid gate on Key only. Value does not pass through the gate.
  Gated Key is max-pooled over a 3x3 neighborhood and concatenated with Value.
  """
  def __init__(self, ich, och, kernel_size=1, stride=1, padding=0):
    super(QAttention, self).__init__()
    self.conv1 = nn.Conv2d(ich, och, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.conv2 = nn.Conv2d(ich, och, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.conv3 = nn.Conv2d(ich, och, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    self.prob = nn.Sigmoid()

    self.conv4 = nn.Conv2d(2 * och, och, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    self.bn4 = nn.BatchNorm2d(och)

    self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

  def _select(self, x):
    value = self.conv1(x)
    key = self.conv2(x)
    query = self.conv3(x)
    gate = self.prob(query)
    key = self.pool(key * gate)
    return value, key

  def forward(self, x):
    value, key = self._select(x)
    fused = torch.cat([value, key], dim=1)
    return self.bn4(self.conv4(fused))

  def fuseforward(self, x):
    # Used after fuse_model folds bn4 into conv4. Value stays in the concat.
    value, key = self._select(x)
    fused = torch.cat([value, key], dim=1)
    return self.conv4(fused)


BFL = QAttention 
             

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          if opt.local_rank in [-1,0]:
            logger.info('Using head kernel: {}'.format(opt.head_kernel))
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks# 1
        self.heads = heads#{'hm': 5, 'reg': 2, 'tracking': 2, 'ltrb_amodal': 4, 'reid': 128}
        for head in self.heads:
            classes = self.heads[head]# num_classes
            head_conv = head_convs[head]# 256
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)# 256 num_classes 1 1
              kind = opt.head_overrides.get(head, opt.head_type)
              if kind == 'qattn':
                conv = QAttention(last_channel, head_conv[0])
              elif kind in ('glu', 'gated_conv'):
                conv = GatingControl(last_channel, head_conv[0], 1 if kind == 'glu' else 3)
              elif kind in ('conv', 'conv_bn'):
                # Published convolutional baseline: 3x3 conv with bias and no BatchNorm.
                # conv_bn is the explicit matched-normalization control.
                conv = nn.Conv2d(last_channel, head_conv[0],
                                kernel_size=head_kernel,
                                padding=head_kernel // 2, bias=True)
                if kind == 'conv_bn':
                  conv = nn.Sequential(conv, nn.BatchNorm2d(head_conv[0]))
              else:
                raise ValueError('Unknown head type {}'.format(kind))
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out