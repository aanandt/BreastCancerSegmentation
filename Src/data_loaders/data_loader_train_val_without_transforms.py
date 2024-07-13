import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pdb 
from tqdm import tqdm

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform_1=None, transform_2=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'Images')
        self.mask_dir = os.path.join(data_dir, 'Masks')

        self.images = os.listdir(self.image_dir)
        self.images.sort()
        self.masks = os.listdir(self.mask_dir)
        self.masks.sort()

        self.transform1 = transform_1
        self.transform2 = transform_2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)

        if self.transform1:
            image = self.transform1(image)
            mask = self.transform2(mask)

        return image, mask, img_name, mask_name

