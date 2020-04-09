import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ['person', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']


class VocClassificationDataset(Dataset):
    """
    VOC dataset
    """
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = os.path.join(root, 'VOCdevkit', 'VOC2012')
        self.images_path = os.path.join(self.root, 'JPEGImages')
        self.labels_path = os.path.join(self.root, 'ImageSets', 'Main')
        self.labels = {}
        self.num_classes = len(CLASSES)
        for i, label in enumerate(CLASSES):
            file_path = os.path.join(self.labels_path, label + f'_{set}.txt')
            for line in open(file_path):
                tmp = line.strip().split()
                target = self.labels.get(tmp[0])
                value = 1. if int(tmp[1]) == 1 else 0.
                if target is None:
                    self.labels[tmp[0]] = np.array([value], dtype='float64')
                else:
                    self.labels[tmp[0]] = np.append(self.labels[tmp[0]], value)

        self.transform = transform
        self.target_transform = target_transform
        self.images = list(self.labels.keys())

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
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
