import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image


# transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
# transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),


class Rotate():
    def __call__(self, img):
        angle = random.sample([0, 90, 180, 270, 360], 1)[0]
        return F.rotate(img, angle)


class Cutout():
    def __init__(self, p=0.5, n_holes=2, length=30):
        self.p = p
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if np.random.random() > self.p:
            return img
        
        img = np.array(img)
        mask_val = img.mean()
        h, w = img.shape[0:2]
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            img[y1: y2, x1: x2] = mask_val
        img = Image.fromarray(img)
        return img