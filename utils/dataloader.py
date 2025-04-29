import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from .dataset import IDRID, IDRID_noVal, IDRID_joint, Messidor, KNEEOA 
from .aug import Rotate, Cutout


def get_loader(args):
    if args.dataset == 'idrid_noval':
        img_size = args.img_size
        up_size = int(args.img_size * 256 / 224)
        train_transform = transforms.Compose([transforms.Resize((up_size, up_size)),
                                    transforms.RandomCrop((img_size, img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        val_transform = transforms.Compose([transforms.Resize((up_size, up_size)),
                                    transforms.CenterCrop((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = IDRID_noVal(root=args.data_root, is_train=True, transform=train_transform)
        valset = IDRID_noVal(root=args.data_root, is_train=False, transform=val_transform)
        testset = None

    elif args.dataset == 'idrid':
        img_size = args.img_size
        up_size = int(args.img_size * 256 / 224)
        train_transform = transforms.Compose([
            transforms.Resize((up_size, up_size)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if args.task == 'joint':
            train_transform = transforms.Compose([
                transforms.Resize((up_size, up_size)),
                transforms.RandomCrop((img_size, img_size)),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                Rotate(),
                transforms.RandomHorizontalFlip(),
                Cutout(n_holes=2, length=100),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        val_transform = transforms.Compose([
            transforms.Resize((up_size, up_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if args.task == 'single':
            trainset = IDRID(root=args.data_root, mode='train', transform=train_transform)
            valset = IDRID(root=args.data_root, mode='valid', transform=val_transform)
            testset = IDRID(root=args.data_root, mode='test', transform=val_transform)
        elif args.task == 'joint':
            trainset = IDRID_joint(root=args.data_root, mode='train', transform=train_transform)
            valset = IDRID_joint(root=args.data_root, mode='valid', transform=val_transform)
            testset = IDRID_joint(root=args.data_root, mode='test', transform=val_transform)

    elif args.dataset == 'messidor':
        n_fold = str(args.n_fold)
        img_size = args.img_size
        up_size = int(args.img_size * 256 / 224)
        train_transform = transforms.Compose([
            transforms.Resize((up_size, up_size)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((up_size, up_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Messidor(root=args.data_root, mode='train'+n_fold, transform=train_transform)
        valset = Messidor(root=args.data_root, mode='valid'+n_fold, transform=val_transform)
        testset = Messidor(root=args.data_root, mode='test'+n_fold, transform=val_transform)

    elif args.dataset == 'kneeoa':
        pixel_mean, pixel_std = 0.66133188, 0.21229856
        data_transform = {
            'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
            ])
        }

        trainset = KNEEOA(root=args.data_root, mode='train', transform=data_transform['train'])
        valset = KNEEOA(root=args.data_root, mode='val', transform=data_transform['val'])
        testset = KNEEOA(root=args.data_root, mode='test', transform=data_transform['test'])

    train_sampler = RandomSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    if testset is not None:
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
