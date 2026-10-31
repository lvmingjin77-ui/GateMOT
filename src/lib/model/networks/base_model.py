from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


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


class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        head_kernel = opt.head_kernel
        self.num_stacks = num_stacks# 1
        self.heads = heads#{'hm': 5, 'reg': 2, 'tracking': 2, 'ltrb_amodal': 4, 'reid': 128}
        for head in self.heads:
            classes = self.heads[head]# num_classes
            head_conv = head_convs[head]# 256
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)# 256 num_classes 1 1
              if opt.use_bfl:
                conv = QAttention(last_channel, head_conv[0])
              else:
                conv = nn.Conv2d(last_channel, head_conv[0],
                                 kernel_size=head_kernel,
                                 padding=head_kernel // 2, bias=True)
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
              fc = nn.Conv2d(last_channel, classes, kernel_size=1,
                             stride=1, padding=0, bias=True)
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
