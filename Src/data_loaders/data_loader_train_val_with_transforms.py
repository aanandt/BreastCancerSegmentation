import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pdb 
from tqdm import tqdm
import torchvision.transforms.functional as TF
import random

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

    def transform_seg_fun(self, image,seg,size = 256,local = False):
        # Resize
        resize = transforms.Resize(size=(int(256), int(256)))
        image = resize(image)
        seg = resize(seg)

        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(size, size))
        # image = TF.crop(image, i, j, h, w)
        # seg = TF.crop(seg, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            seg = TF.hflip(seg)

        seg = TF.to_grayscale(seg,num_output_channels=1)

        image_2 = image

        count = 0
        while(count < 3):

            # This is Identity
            if random.random() > 0.9:
                count = count + 1
            if(count > 2):
                break 

            # This is the gaussina_blur
            if random.random() > 0.9:
                image_2 = TF.gaussian_blur(image_2,(3,3))
                count = count + 1
            if(count > 2):
                break

            # This is the adjust_contrast
            if random.random() > 0.9:
                in_count = 0
                while(in_count == 0):
                    val = random.random()
                    if(val > 0.5 and val < 0.95):
                        in_count = 1
                image_2 = TF.adjust_contrast(image_2,val)
                count = count + 1
            if(count > 2):
                break

            # This is the adjust_brightness
            if random.random() > 0.9:
                in_count = 0
                while(in_count == 0):
                    val = random.random()
                    if(val > 0.5 and val < 0.95):
                        in_count = 1
                image_2 = TF.adjust_brightness(image_2,val)
                count = count + 1
            if(count > 2):
                break

            # # This is elastic
            if random.random() > 0.9:
                in_count = 0
                while(in_count == 0):
                    val = random.random() - 1
                    if(val > -0.5 and val < 0.5):
                        in_count = 1
                image_2 = TF.equalize(image_2)
                count = count + 1
            if(count > 2):
                break

        image = TF.to_tensor(image)
        image_2 = TF.to_tensor(image_2)
        seg = TF.to_tensor(seg)
        

        return image,image_2,seg

    def transform(self, image, mask,shape = 256):
        # Resize
        width,height = image.size

        resize = transforms.Resize(size=(int(shape),int(shape)))
        image = resize(image)
        mask = resize(mask)

        mask = TF.rgb_to_grayscale(mask,1)    
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = image.repeat(3,1,1)
        image = TF.normalize(image,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

        return image, mask

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)

        image,image_2,mask = self.transform_seg_fun(image,mask)

        return image,image_2, mask, img_name, mask_name
