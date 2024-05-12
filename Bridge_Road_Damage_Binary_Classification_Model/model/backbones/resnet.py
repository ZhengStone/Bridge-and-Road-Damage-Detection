# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from ..blocks import BasicBlock, BottleNeck
from ..bricks import conv1x1

__all__ = ['ResNet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    #The pre-trained weights of ResNet-50 are trained on the ImageNet dataset(IMAGENET1K_V2).
    'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# from torchvision.models.resnet import resnet34


class ResNet(nn.Module):
    def __init__(self,
                 depth,
                 stage_blocks=None,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(False, False, False, False),
                 out_indices=(0, 1, 2, 3),
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 norm_layer=None,
                 pretrained=True,
                 load_fc=False,
                 is_extra=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.zero_init_residual = zero_init_residual
        self.depth = depth
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        if stage_blocks is None:
            if depth == 18:
                stage_blocks = (BasicBlock, (2, 2, 2, 2))
            elif depth == 34:
                stage_blocks = (BasicBlock, (3, 4, 6, 3))
            elif depth == 50:
                stage_blocks = (BottleNeck, (3, 4, 6, 3))
            elif depth == 101:
                stage_blocks = (BottleNeck, (3, 4, 23, 3))
            elif depth == 152:
                stage_blocks = (BottleNeck, (3, 8, 36, 3))
            else:
                raise ValueError('请手动输入，暂时还没有默认定义！')

        block, stage_blocks = stage_blocks
        self.stage_blocks = stage_blocks[:num_stages]

        self.is_extra = is_extra
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # stem层
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer层
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            res_layer = self._make_layer(block, planes, num_blocks, stride=strides[i], dilate=dilations[i])
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.init_weights(pretrained, load_fc)

    def init_weights(self, pretrained=None, load_fc=False):
        if isinstance(pretrained, bool):
            # resnet系列有默认预训练权重，自动导入
            if pretrained is True:
                model_name = 'resnet' + str(self.depth)
                state_dict = model_zoo.load_url(model_urls[model_name], progress=True)
                if load_fc is False:
                    state_dict.pop('fc.weight')
                    state_dict.pop('fc.bias')
                self.load_state_dict(state_dict)
                print('load pretrained from {}'.format(model_urls[model_name]))
        elif isinstance(pretrained, str):
            if pretrained.startswith(('http://', 'https://')):
                # url
                state_dict = model_zoo.load_url(pretrained, progress=True)
                if load_fc is False:
                    state_dict.pop('fc.weight')
                    state_dict.pop('fc.bias')
                self.load_state_dict(state_dict)
                print('load pretrained from {}'.format(pretrained))
            else:
                # 本地路径
                state_dict = torch.load(pretrained, map_location='cpu')
                self.load_state_dict(state_dict)
                print('load pretrained from {}'.format(pretrained))
        else:
            # 初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BottleNeck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.is_extra:
            outs = [x]
        else:
            outs = []
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
