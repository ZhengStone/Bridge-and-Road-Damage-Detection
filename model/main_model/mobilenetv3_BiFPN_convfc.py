import torch
import torch.nn as nn
from ..backbones import MobileNetV3
from ..necks import BiFPN
from ..heads import FCHead
from .Intergrated_Model import IntergratedModel

backbone = MobileNetV3(is_large=True, in_channels=3, out_indices=(0, 1, 2, 3))

neck = BiFPN(in_channels=[24, 40, 160, 960], out_channels=256)

head = FCHead(lateral_channel=256)

mobilev3_BiFPN_convfc_train = IntergratedModel(backbone, neck, head)
mobilev3_BiFPN_convfc_train = mobilev3_BiFPN_convfc_train.cuda()

mobilev3_BiFPN_convfc_test = IntergratedModel(backbone, neck, head)
mobilev3_BiFPN_convfc_test = mobilev3_BiFPN_convfc_test.cuda()