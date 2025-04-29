from models.algs.base import Base
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggdNet(Base):
    def __init__(self, args):
        super(AggdNet, self).__init__(args)
        if args.var_type == 'two_layer':
            self.predict_var = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.Dropout(),
                nn.ReLU(True),
                nn.Linear(512, 2)
            )
        elif args.var_type == 'one_layer':
            self.predict_var = nn.Linear(self.num_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        sigma = self.predict_var(x)
        sigma = torch.clamp(F.softplus(sigma), max=50)
        x = self.head(x)
        return x, sigma.detach()