import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob


class IDRID_noVal():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        train_file = os.path.join(self.root, '2.Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
        test_file = os.path.join(self.root, '2.Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')
        if self.is_train:
            with open(train_file) as f:
                lines = f.readlines()
            self.train_img_paths = [os.path.join(self.root, '1.OriginalImages/a.TrainingSet', f.split(',')[0] + '.jpg') for f in lines[1:]]
            self.train_label = [int(f.split(',')[1]) for f in lines[1:]]

        if not self.is_train:
            with open(test_file) as f:
                lines = f.readlines()
            self.test_img_paths = [os.path.join(self.root, '1.OriginalImages/b.TestingSet', f.split(',')[0] + '.jpg') for f in lines[1:]]
            self.test_label = [int(f.split(',')[1]) for f in lines[1:]]

    def __getitem__(self, index):
        if self.is_train:
            img_path, target = self.train_img_paths[index], self.train_label[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img_path, target = self.test_img_paths[index], self.test_label[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class IDRID():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.transform = transform
        mode_file = os.path.join(self.root, mode+'.txt')
        with open(mode_file) as f:
            lines = f.readlines()
        head = 'b.TestingSet' if mode == 'test' else 'a.TrainingSet'
        self.mode_img_paths = [os.path.join(self.root, f'1.OriginalImages/{head}.gray', f.split(',')[0] + '.png') for f in lines]
        self.mode_labels = [int(f.split(',')[1]) for f in lines]

    def __getitem__(self, index):
        img_path, target = self.mode_img_paths[index], self.mode_labels[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.mode_img_paths)


class IDRID_joint():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.transform = transform
        mode_file = os.path.join(self.root, mode+'.txt')
        with open(mode_file) as f:
            lines = f.readlines()
        head = 'b.TestingSet' if mode == 'test' else 'a.TrainingSet'
        self.mode_img_paths = [os.path.join(self.root, f'1.OriginalImages/{head}.gray', f.split(',')[0] + '.png') for f in lines]
        self.mode_labels = [int(f.split(',')[1]) for f in lines]
        self.mode_labels2 = [int(f.split(',')[2]) for f in lines]

    def __getitem__(self, index):
        img_path, target = self.mode_img_paths[index], self.mode_labels[index]
        target2 = self.mode_labels2[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, target2

    def __len__(self):
        return len(self.mode_img_paths)


class Messidor():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.transform = transform
        mode_file = os.path.join(self.root, 'splits', mode+'.txt')
        with open(mode_file) as f:
            lines = f.readlines()
        self.mode_img_paths = [os.path.join(self.root, 'images.gray', f.split(',')[0].replace('tif', 'png')) for f in lines]
        self.mode_labels = [int(f.split(',')[1]) for f in lines]

    def __getitem__(self, index):
        img_path, target = self.mode_img_paths[index], self.mode_labels[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.mode_img_paths)


class KNEEOA():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.transform = transform
        mode_root = os.path.join(self.root, mode)
        self.mode_img_paths = glob.glob(f'{mode_root}/*/*')
        
    def __getitem__(self, index):
        img_path = self.mode_img_paths[index]
        img = Image.open(img_path).convert('RGB')
        target = int(img_path.split('/')[-2])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.mode_img_paths)