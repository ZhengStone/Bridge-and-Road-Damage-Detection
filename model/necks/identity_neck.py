import torch.nn as nn


__all__ = ['IdentityNeck']


class IdentityNeck(nn.Module):
    def __init__(self):
        super(IdentityNeck, self).__init__()

    def forward(self, input_tensor):
        return input_tensor
