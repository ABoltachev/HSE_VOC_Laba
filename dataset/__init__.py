import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']


class VocAnimalsDataset(Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = os.path.join(root, 'VOCdevkit', 'VOC2012')
        self.images_path = os.path.join(self.root, 'JPEGImages')
        self.labels_path = os.path.join(self.root, 'ImageSets', 'Main')
        self.labels = {}
        for i, label in enumerate(CLASSES):
            file_path = os.path.join(self.labels_path, label + f'_{set}.txt')
            for line in open(file_path):
                tmp = line.strip().split()
                target = self.labels.get(tmp[0])
                if target is None:
                    self.labels[tmp[0]] = np.array([int(tmp[1])])
                else:
                    self.labels[tmp[0]] = np.append(target, int(tmp[1]))
        self.labels = {key: value for key, value in self.labels.items() if 1 in value}

        self.transform = transform
        self.target_transform = target_transform
        self.images = list(self.labels.keys())

    def __getitem__(self, index):
        img_path = os.path.join(self.images_path, self.images[index] + '.jpg')
        img = Image.open(img_path).convert("RGB")

        target = self.labels[os.path.splitext(self.images[index])[0]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.labels)
