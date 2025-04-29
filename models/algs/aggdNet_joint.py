from models.algs.base_joint import Base_joint
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggdNet_joint(Base_joint):
    def __init__(self, args):
        super(AggdNet_joint, self).__init__(args)
        if args.var_type == 'two_layer':
            self.predict_var = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.Dropout(),
                nn.ReLU(True),
                nn.Linear(512, 2)
            )
            self.predict_var2 = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.Dropout(),
                nn.ReLU(True),
                nn.Linear(512, 2)
            )
        elif args.var_type == 'one_layer':
            self.predict_var = nn.Linear(self.num_features, 2)
            self.predict_var2 = nn.Linear(self.num_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        sigma = torch.clamp(F.softplus(self.predict_var(x)), max=50)
        sigma2 = torch.clamp(F.softplus(self.predict_var2(x)), max=50)
        return self.head(x), self.head2(x), sigma.detach(), sigma2.detach()