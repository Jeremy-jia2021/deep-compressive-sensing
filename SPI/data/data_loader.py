from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np


class public_dataset(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_root, image_size=128):
        """Initialize and preprocess the CelebA dataset."""
        self.image_root = image_root
        self.image_size = image_size
        self.test_dataset = []
        self.preprocess()

        transform = []
        # if mode == 'train':
        #     transform.append(T.RandomHorizontalFlip())
        transform.append(T.Resize(self.image_size))
        transform.append(T.Grayscale(1))
        transform.append(T.ToTensor())
        # transform_list += [transforms.Normalize((0.5,), (0.5,))]
        # transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
        # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(transform)

    def preprocess(self):
        self.image_dir = os.path.join(self.image_root)
        try:
            for file in os.listdir(self.image_dir):
                filename = os.path.join(self.image_dir, file)
                self.test_dataset.append(filename)
            self.num_images = len(self.test_dataset)
            print('dataset preprocessing done!')
        except:
            print(self.image_root + ' is not a directory')
            self.test_dataset = None
            self.num_images = 0

    def __getitem__(self, index):
        filename = self.test_dataset[index]
        image = Image.open(os.path.join(filename))
        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def create_dataset(image_dir):
    return public_dataset(image_dir)

def create_dataloader(dataset, batch_size=1, num_workers=1):
    """Build and return a data loader."""

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader