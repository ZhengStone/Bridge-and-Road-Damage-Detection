import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['SE']


class SE(nn.Module):
    """
    Squeeze-and-Excitation Module.
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 with_bn=True,
                 act_func='sigmoid'):
        super(SE, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // ratio, 1)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(int(channels / ratio), channels, 1)

        assert act_func in ['sigmoid', 'hsigmoid']
        self.act_func = act_func

        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(channels // ratio)
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        if self.with_bn:
            out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        if self.with_bn:
            out = self.bn2(out)
        if self.act_func == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.act_func == 'hsigmoid':
            out = F.relu6(x + 3, inplace=True) / 6
        return x * out
