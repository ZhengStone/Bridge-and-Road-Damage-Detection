# -*- coding:utf-8 -*-
import torch.nn as nn
from ..bricks import InConv, DownSample, UpSample, CatAndConv, OutConv

__all__ = ['UnetInstance']


class UnetInstance(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, kernel_size=3, norm=nn.BatchNorm2d, usr_res=False):
        super(UnetInstance, self).__init__()
        self.inc = InConv(in_channels, 64, kernel_size, norm, usr_res=usr_res)

        self.down1 = DownSample(64, 128, kernel_size, norm, usr_res=usr_res)
        self.down2 = DownSample(128, 256, kernel_size, norm, usr_res=usr_res)
        self.down3 = DownSample(256, 512, kernel_size, norm, usr_res=usr_res)
        self.down4 = DownSample(512, 1024, kernel_size, norm, usr_res=usr_res)

        self.up1 = UpSample(1024, 512, kernel_size, use_bilinear=True, align_corners=True, norm=nn.BatchNorm2d, usr_res=usr_res)
        self.up2 = UpSample(512, 256, kernel_size, use_bilinear=True, align_corners=True, norm=nn.BatchNorm2d, usr_res=usr_res)
        self.up3 = UpSample(256, 128, kernel_size, use_bilinear=True, align_corners=True, norm=nn.BatchNorm2d, usr_res=usr_res)
        self.up4 = UpSample(128, 64, kernel_size, use_bilinear=True, align_corners=True, norm=nn.BatchNorm2d, usr_res=usr_res)

        self.CAC1 = CatAndConv(in_ch_1=512, in_ch_2=512, out_ch=512, kernel_size=3, norm=nn.BatchNorm2d, usr_res=usr_res, is_size_pad=False)
        self.CAC2 = CatAndConv(in_ch_1=256, in_ch_2=256, out_ch=256, kernel_size=3, norm=nn.BatchNorm2d, usr_res=usr_res, is_size_pad=False)
        self.CAC3 = CatAndConv(in_ch_1=128, in_ch_2=128, out_ch=128, kernel_size=3, norm=nn.BatchNorm2d, usr_res=usr_res, is_size_pad=False)
        self.CAC4 = CatAndConv(in_ch_1=64, in_ch_2=64, out_ch=64, kernel_size=3, norm=nn.BatchNorm2d, usr_res=usr_res, is_size_pad=False)

        self.output_conv = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.CAC1(x, x4)
        x = self.up2(x)
        x = self.CAC2(x, x3)
        x = self.up3(x)
        x = self.CAC3(x, x2)
        x = self.up4(x)
        x = self.CAC4(x, x1)

        x = self.output_conv(x)
        return x
