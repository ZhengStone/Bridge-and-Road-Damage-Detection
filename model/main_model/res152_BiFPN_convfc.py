import torch
import torch.nn as nn
from ..backbones import ResNet
from ..necks import BiFPN
from ..heads import FCHead
from .Intergrated_Model import IntergratedModel

backbone_train = ResNet(depth=152, 
                  stage_blocks=None,
                  in_channels=3, num_stages=4, strides=(1, 2, 2, 2),
                  dilations=(False, False, False, False),
                  out_indices=(0, 1, 2, 3),
                  zero_init_residual=False,
                  groups=1,
                  width_per_group=64,
                  norm_layer=None,
                  pretrained=True,
                  load_fc=False,
                  is_extra=False)

backbone_test = ResNet(depth=152, 
                  stage_blocks=None,
                  in_channels=3, num_stages=4, strides=(1, 2, 2, 2),
                  dilations=(False, False, False, False),
                  out_indices=(0, 1, 2, 3),
                  zero_init_residual=False,
                  groups=1,
                  width_per_group=64,
                  norm_layer=None,
                  pretrained=False,
                  load_fc=False,
                  is_extra=False)

neck = BiFPN(in_channels=[256, 512, 1024, 2048], out_channels=256)

head = FCHead(lateral_channel=256)

res152_BiFPN_convfc_train = IntergratedModel(backbone_train, neck, head)
res152_BiFPN_convfc_train = res152_BiFPN_convfc_train.cuda()

res152_BiFPN_convfc_test = IntergratedModel(backbone_test, neck, head)
res152_BiFPN_convfc_test = res152_BiFPN_convfc_test.cuda()