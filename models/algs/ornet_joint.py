from models.algs.base_joint import Base_joint
import torch
import torch.nn as nn
import torch.nn.functional as F


class ORNet_joint(Base_joint):
    def __init__(self, args):
        super(ORNet_joint, self).__init__(args)
        self.args = args
        self.reg_head = self.get_reg_head(num_classes=args.num_classes)
        self.reg_head2 = self.get_reg_head(num_classes=3)

    def get_reg_head(self, num_classes):
        reghead = nn.Linear(num_classes, 1)
        return reghead
    
    def forward(self, x):
        x = self.backbone(x)
        x1 = self.head(x1)
        r1 = self.reg_head(x1)
        
        x2 = self.head2(x)
        r2 = self.reg_head2(x2)
        return x1, x2, r1, r2