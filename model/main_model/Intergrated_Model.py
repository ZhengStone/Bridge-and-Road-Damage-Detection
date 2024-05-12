import torch
import torch.nn as nn

class IntergratedModel(nn.Module):
    def __init__(self, backbone, neck, head):
        super(IntergratedModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x