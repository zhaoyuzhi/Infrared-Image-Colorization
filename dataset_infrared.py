# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:03:52 2018

@author: yzzhao2
"""

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class InfraredDataset(Dataset):
    def __init__(self, baseroot, lwirlist, visiblelist):                    # root: list ; transform: torch transform
        self.baseroot = baseroot
        self.lwirlist = lwirlist
        self.visiblelist = visiblelist
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        
        # get one image
        lwirpath = self.baseroot + self.lwirlist[index]                     # path of one image
        lwirimg = Image.open(lwirpath).convert('L')                         # read one image, and convert to grayscale
        lwirimg = lwirimg.resize((256, 256), Image.ANTIALIAS)               # this is [256, 256]
        img = self.transform1(lwirimg)
        
        # change RGB to Lab
        visiblepath = self.baseroot + self.visiblelist[index]               # path of one image
        visibleimg = Image.open(visiblepath).convert('RGB')                 # read one image, and convert to RGB
        visibleimg = visibleimg.resize((256, 256), Image.ANTIALIAS)         # this is [256, 256]
        target = self.transform2(visibleimg)

        return img, target
    
    def __len__(self):
        return len(self.lwirlist)
