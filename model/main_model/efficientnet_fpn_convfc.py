import torch
import torch.nn as nn
from ..backbones import EfficientNet
from ..necks import FPN
from ..heads import effifchead
from .Intergrated_Model import IntergratedModel

backbone_train = EfficientNet(model_name='efficientnet-b1', pretrained=True, in_channels=3, out_indices=(0, 1, 2, 3))

backbone_test = EfficientNet(model_name='efficientnet-b1', pretrained=False, in_channels=3, out_indices=(0, 1, 2, 3))

neck = FPN(in_channels=[16, 24, 40, 112], out_channels=256)

head = effifchead(lateral_channel=256)

effib1_fpn_convfc_train = IntergratedModel(backbone_train, neck, head)
effib1_fpn_convfc_train = effib1_fpn_convfc_train.cuda()

effib1_fpn_convfc_test = IntergratedModel(backbone_test, neck, head)
effib1_fpn_convfc_test = effib1_fpn_convfc_test.cuda()