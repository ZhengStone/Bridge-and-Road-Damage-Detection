import torch
import torch.nn as nn
from ..backbones import UnetInstance
from ..necks import IdentityNeck
from ..heads import unethead
from .Intergrated_Model import IntergratedModel

backbone = UnetInstance(in_channels=3, out_channels=64, kernel_size=3, norm=nn.BatchNorm2d, usr_res=True)

neck = IdentityNeck()

head = unethead(lateral_channel=64)

unet_iden_convfc_train = IntergratedModel(backbone, neck, head)
unet_iden_convfc_train = unet_iden_convfc_train.cuda()

unet_iden_convfc_test = IntergratedModel(backbone, neck, head)
unet_iden_convfc_test = unet_iden_convfc_test.cuda()