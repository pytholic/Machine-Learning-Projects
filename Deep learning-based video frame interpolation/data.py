#!/usr/bin/env python
# coding: utf-8

# In[80]:

import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import os
import glob


# In[ ]:


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# In[209]:


class triplet_train(Dataset):
    """
    Reads all triplet set of frames in a directory.
    Each triplet set contains frame 1, 2, 3.
    Each image is named im1.png, im2.png, im3.png.
    Frame 1, 3 are the input and Frame 2 is the output.
    """

    def __init__(self, in_dir, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        self.sub_folders = [(in_dir + '/' + f) for f in listdir(in_dir)]
        self.input_triplets = []
        for folder in self.sub_folders:
            self.input_triplets += [(folder + '/' + f) for f in listdir(folder)]
        
        self.input_triplets = np.array(self.input_triplets)
        self.data_len = len(self.input_triplets)
        
    def __getitem__(self, index):
        frame1 = self.transform(Image.open(self.input_triplets[index] + "/im1.png"))
        frame2 = self.transform(Image.open(self.input_triplets[index] + "/im2.png"))
        frame3 = self.transform(Image.open(self.input_triplets[index] + "/im3.png"))

        return frame1, frame2, frame3

    def __len__(self):
        return self.data_len


# In[163]:


class triplet_test(Dataset):
    """
    Reads all triplet set of frames in test directory.
    Each triplet set contains frame 1, 2, 3.
    Each image is named im1.png, im2.png, im3.png.
    Frame 1, 3 are the input and Frame 2 is the ground truth.
    """

    def __init__(self, db_dir, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        #sub_folders_list = os.listdir(sub_folder)
        self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir)])
        self.data_len = len(self.triplet_list)

    def __getitem__(self, index):
        frame1 = self.transform(Image.open(self.triplet_list[index] + "/frame0.png"))
        gt = self.transform(Image.open(self.triplet_list[index] + "/frame1.png"))
        frame3 = self.transform(Image.open(self.triplet_list[index] + "/frame2.png"))
        
        return frame1, gt, frame3
    
    def __len__(self):
        return self.data_len

