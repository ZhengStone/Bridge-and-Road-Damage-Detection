# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Up', 'Down', 'CatAndConv', 'DoubleConv', 'InConv', 'OutConv', 'UpSample', 'DownSample', 'conv1x1', 'conv3x3']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Up(nn.Module):
    def __init__(self, in_ch, use_bilinear=True, align_corners=True):
        super(Up, self).__init__()
        if use_bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.down(x)
        return x


class CatAndConv(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, kernel_size=3, norm=nn.BatchNorm2d, usr_res=True, is_size_pad=False):
        super(CatAndConv, self).__init__()
        self.is_size_pad = is_size_pad  # padding 自适应尺寸
        self.conv = DoubleConv(in_ch_1 + in_ch_2, out_ch, kernel_size, norm, usr_res)

    def forward(self, x1, x2):
        # follow x2 size
        if self.is_size_pad:
            diff_h = x2.size()[2] - x1.size()[2]
            diff_w = x2.size()[3] - x1.size()[3]

            if diff_h > 0:
                x1 = F.pad(x1, (0, 0, diff_h // 2, diff_h - diff_h // 2))
            else:
                x1 = x1[..., -diff_h // 2:-diff_h // 2 + x2.size(2), :]

            if diff_w > 0:
                x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2, 0, 0))
            else:
                x1 = x1[..., -diff_w // 2:-diff_w // 2 + x2.size(3)]

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, kernel_size=3, norm=nn.BatchNorm2d, usr_res=True):
        super(DoubleConv, self).__init__()
        self.usr_res = usr_res
        padding = (kernel_size - 1) // 2
        if usr_res:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
            self.bn1 = norm(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
            self.bn2 = norm(out_ch)
            if in_ch != out_ch:
                self.downsample = nn.Conv2d(in_ch, out_ch, 1)
            else:
                self.downsample = None
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                norm(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
                norm(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.usr_res:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(identity)
            out += identity
            x = self.relu(out)
        else:
            x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, norm=nn.BatchNorm2d, usr_res=True):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, kernel_size, norm, usr_res)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, norm=nn.BatchNorm2d, usr_res=True):
        super(DownSample, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, kernel_size, norm, usr_res))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_bilinear=True, align_corners=True, norm=nn.BatchNorm2d, usr_res=True):
        super(UpSample, self).__init__()
        if use_bilinear:
            self.UpConv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners),
                DoubleConv(in_channels, out_channels, kernel_size, norm, usr_res))
        else:
            self.UpConv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
                DoubleConv(in_channels, out_channels, kernel_size, norm, usr_res))

    def forward(self, x):
        x = self.UpConv(x)
        return x
