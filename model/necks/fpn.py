import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FPN']

class FPN(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=256):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for channel in self.in_channels:
            l_conv = nn.Conv2d(channel, out_channels, 1)
            self.lateral_convs.append(l_conv)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_convs.append(fpn_conv)

    def forward(self, features):
        assert len(features) == len(self.in_channels)
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs
