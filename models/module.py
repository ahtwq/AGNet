import torch
import torch.nn as nn



class OrdinalRegressor(nn.Module):
    def __init__(self, num_class, train_cutpoints=False, scale=5.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred):
        sigmoids = torch.sigmoid(self.cutpoints - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)
        return likelihoods