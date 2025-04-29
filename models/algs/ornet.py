from models.algs.base import Base
import torch
import torch.nn as nn
import torch.nn.functional as F


class ORNet(Base):
    def __init__(self, args):
        super(ORNet, self).__init__(args)
        self.args = args
        self.reg_head = self.get_reg_head()

    def get_reg_head(self):
        reghead = nn.Linear(self.args.num_classes, 1)
        return reghead
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        r = self.reg_head(x)
        return x, r 