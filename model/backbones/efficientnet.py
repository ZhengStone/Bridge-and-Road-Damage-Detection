# -*- coding:utf-8 -*-
import torch.nn as nn
from efficientnet_pytorch import EfficientNet as EffNet
from efficientnet_pytorch.utils import url_map
import torch.utils.model_zoo as model_zoo



class EfficientNet(nn.Module):
    def __init__(self, model_name, pretrained=True, in_channels=3, out_indices=(0, 1, 2, 3, 4)):
        """
        https://arxiv.org/pdf/1905.11946.pdf
        Args:
            model_name:
                VALID_MODELS = (
                        'efficientnet-b0',
                        'efficientnet-b1',
                        'efficientnet-b2',
                        'efficientnet-b3',
                        'efficientnet-b4',
                        'efficientnet-b5',
                        'efficientnet-b6',
                        'efficientnet-b7',
                        'efficientnet-b8',
                        # Support the construction of 'efficientnet-l2' without pretrained weights
                        'efficientnet-l2'
                    )
            pretrained:
                True or False
            in_channels: 3 in default
            out_indices: (0, 1, 2, 3, 4)
        """
        super(EfficientNet, self).__init__()
        self.eff_net = EffNet.from_name(model_name=model_name, in_channels=in_channels, num_classes=1000)
        if pretrained and model_name != 'efficientnet-l2':
            weights_path = url_map[model_name]
            state_dict = model_zoo.load_url(weights_path, progress=True)
            self.eff_net.load_state_dict(state_dict)
        self.out_indices = out_indices

    def forward(self, x):
        '''
        输入： (4, 3, 256, 256)
        注意resnet-50的参数量是 26M
        模型：      精度对标     模型参数量
        b0: <====> resnet-50      5.3M
            torch.Size([4, 16, 128, 128])
            torch.Size([4, 24, 64, 64])
            torch.Size([4, 40, 32, 32])
            torch.Size([4, 112, 16, 16])
            torch.Size([4, 1280, 8, 8])
        b1: <====> resnet152      7.8M
            torch.Size([4, 16, 128, 128])
            torch.Size([4, 24, 64, 64])
            torch.Size([4, 40, 32, 32])
            torch.Size([4, 112, 16, 16])
            torch.Size([4, 1280, 8, 8])
        b2: <====> Inception-v4    9.2M
            torch.Size([4, 16, 128, 128])
            torch.Size([4, 24, 64, 64])
            torch.Size([4, 48, 32, 32])
            torch.Size([4, 120, 16, 16])
            torch.Size([4, 1408, 8, 8])
        b3: <====> ResNeXt-101     12M
            torch.Size([4, 24, 128, 128])
            torch.Size([4, 32, 64, 64])
            torch.Size([4, 48, 32, 32])
            torch.Size([4, 136, 16, 16])
            torch.Size([4, 1536, 8, 8])
        b4: <====> SENet           19M
            torch.Size([4, 24, 128, 128])
            torch.Size([4, 32, 64, 64])
            torch.Size([4, 56, 32, 32])
            torch.Size([4, 160, 16, 16])
            torch.Size([4, 1792, 8, 8])
        b5: <====> AmoebaNet-C      30M
            torch.Size([4, 24, 128, 128])
            torch.Size([4, 40, 64, 64])
            torch.Size([4, 64, 32, 32])
            torch.Size([4, 176, 16, 16])
            torch.Size([4, 2048, 8, 8])
        b6: <====>                  43M
            torch.Size([4, 32, 128, 128])
            torch.Size([4, 40, 64, 64])
            torch.Size([4, 72, 32, 32])
            torch.Size([4, 200, 16, 16])
            torch.Size([4, 2304, 8, 8])
        b7: <====> GPipe            66M
            torch.Size([4, 32, 128, 128])
            torch.Size([4, 48, 64, 64])
            torch.Size([4, 80, 32, 32])
            torch.Size([4, 224, 16, 16])
            torch.Size([4, 2560, 8, 8])
        b8: <====>
            torch.Size([4, 32, 128, 128])
            torch.Size([4, 56, 64, 64])
            torch.Size([4, 88, 32, 32])
            torch.Size([4, 248, 16, 16])
            torch.Size([4, 2816, 8, 8])
        l2: <====>
            torch.Size([4, 72, 128, 128])
            torch.Size([4, 104, 64, 64])
            torch.Size([4, 176, 32, 32])
            torch.Size([4, 480, 16, 16])
            torch.Size([4, 5504, 8, 8])
        '''
        features = self.eff_net.extract_endpoints(x)
        return [list(features.values())[i] for i in self.out_indices]
