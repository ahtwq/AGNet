from models.algs.base_joint import Base_joint
from models.module import OrdinalRegressor
import torch
import torch.nn as nn


class OrdinalReg_joint(Base_joint):
    def __init__(self, args):
        super(OrdinalReg_joint, self).__init__(args)
        self.score2prob = OrdinalRegressor(args.num_classes)
        self.score2prob2 = OrdinalRegressor(3)

    def get_head(self, args, num_classes=None):
        head = nn.Linear(self.num_features, 1)
        return head

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.score2prob(self.head(x))
        x2 = self.score2prob2(self.head2(x))
        return x1, x2