import timm
import torch
import torch.nn as nn
from collections import OrderedDict
import os


backbone2nf = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'efficientnet_b1': 1280,
            'vit_base_patch16_224': 768}


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.num_features = backbone2nf[args.backbone]
        self.backbone = self.get_backbone(args)
        self.head = self.get_head(args)
        
    def get_backbone(self, args):
        # backbone_name = 'resnet'
        backbone = timm.create_model(args.backbone, num_classes=0, pretrained=True)
        if args.pretrained_dir is None:
            print(f'loading params from timm.')
            return backbone

        path = args.pretrained_dir
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                key = k[7:] if k[0:7] == 'module.' else k
                new_state_dict[key] = v
            model_state = backbone.state_dict()
            filter_state_dict = {k: v for k,v in new_state_dict.items() if k in model_state}
            print(f'valid params ratio: {len(filter_state_dict)}/{len(new_state_dict)}')

            model_state.update(filter_state_dict)
            backbone.load_state_dict(model_state)
            print(f'loading params from {path} on pretrained on kaggle DR 2015.')
        else:
            print('!!! offered pretrained dir is not exists.')
        return backbone

    def get_head(self, args, num_classes=None):
        if args.head_type == 'one_layer':
            if num_classes is not None:
                head = nn.Linear(self.num_features, num_classes)
            else:
                head = nn.Linear(self.num_features, args.num_classes)
        elif args.head_type == 'two_layer':
            if num_classes is not None:
                head = nn.Linear(self.num_features, num_classes)
            else:
                head = nn.Sequential(
                    nn.Linear(self.num_features, 512),
                    nn.Dropout(),
                    nn.ReLU(True),
                    nn.Linear(512, args.num_classes)
                )
        return head
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

