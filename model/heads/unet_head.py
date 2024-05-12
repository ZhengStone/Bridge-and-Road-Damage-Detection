import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["unethead"]

class unethead(nn.Module):
    def __init__(
        self,
        lateral_channel=256,
    ):
        super(unethead, self).__init__()
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
            nn.Linear(640*640, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, features):
        x_1 = self.conv_fc(features)  # 1/1 [-wuqiong, +wuqiong]
        return x_1