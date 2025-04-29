from models.algs.base import Base
from models.module import OrdinalRegressor
import torch
import torch.nn as nn


class OrdinalReg(Base):
    def __init__(self, args):
        super(OrdinalReg, self).__init__(args)
        self.score2prob = OrdinalRegressor(args.num_classes)

    def get_head(self,args):
        head = nn.Linear(self.num_features, 1)
        return head

    def forward(self, x):
        x = self.backbone(x)
        x = self.score2prob(self.head(x))
        return x