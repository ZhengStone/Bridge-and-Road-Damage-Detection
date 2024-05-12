import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["effifchead"]

class effifchead(nn.Module):
    def __init__(
        self,
        lateral_channel=256,
    ):
        super(effifchead, self).__init__()
        self.lateral_convs = nn.ModuleList()
        for i in range(4):
            if i == 0:
                l_conv = nn.Conv2d(
                    lateral_channel, lateral_channel // 4, 3, padding=1, bias=True
                )
            else:
                l_conv = nn.Sequential(
                    nn.Conv2d(
                        lateral_channel, lateral_channel // 4, 3, padding=1, bias=True
                    ),
                    nn.UpsamplingNearest2d(scale_factor=2 ** i),
                )
            self.lateral_convs.append(l_conv)

        self.conv_fc = nn.Sequential(
            nn.Conv2d(lateral_channel, lateral_channel // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(lateral_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                lateral_channel // 4, lateral_channel // 4, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(lateral_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(lateral_channel // 4, 1, 3, padding=1, bias=True),
            nn.Flatten(),
            nn.Linear(320*320, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, features):
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        fuse = torch.cat(laterals, 1)
        x_1 = self.conv_fc(fuse)  # 1/1 [-wuqiong, +wuqiong]
        return x_1