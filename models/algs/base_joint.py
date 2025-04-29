from models.algs.base import Base
import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_joint(Base):
    def __init__(self, args):
        super(Base_joint, self).__init__(args)
        self.args = args
        self.head2 = self.get_head(args, num_classes=3)
        
    def forward(self, x):
        x = self.backbone(x)
        x1 = self.head(x)
        x2 = self.head2(x)
        return x1, x2