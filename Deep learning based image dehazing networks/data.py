#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from utils import rgb_to_tensor


# In[ ]:


def augmentation(input_image, target_image):
    deg = random.choice([0, 90, 180, 270])
    
    if deg != 0:
        input_image = transforms.functional.rotate(input_image, deg)
        target_image = transforms.functional.rotate(target_image, deg)

    return input_image, target_image


# In[ ]:


'''def extract_patch(input_image, gt_image):
    width, height = input_image.size
    if width >= 2048 and height >= 2048:
        choice = random.choice([1, 2, 3])
        if choice == 1:
            px = py = 1024
        elif choice == 2:
            px = 1024
            py = 2048
        else:
            px = py = 2048
    else:
        px = py = 1024
    x = random.randrange(0, width - px + 1)
    y = random.randrange(0, height - py + 1)
    input_image = input_image.crop((x, y, x + px, y + py))
    gt_image = gt_image.crop((x, y, x + px, y + py))
    
    if px > 1024 or py > 1024:
        input_image = input_image.resize((1024, 1024))
        gt_image = gt_image.resize((1024, 1024))
    return input_image, gt_image'''

def image_large(image_path):
    width, height = Image.open(image_path).size
    return width >= 512 and height >= 512

def extract_patch(input_image, gt_image):
    width, height = input_image.size
    if width >= 1024 and height >= 1024:
        choice = random.choice([1, 2, 3])
        if choice == 1:
            px = py = 512
        elif choice == 2:
            px = 512
            py = 1024
        else:
            px = py = 1024
    else:
        px = py = 512
    x = random.randrange(0, width - px + 1)
    y = random.randrange(0, height - py + 1)
    input_image = input_image.crop((x, y, x + px, y + py))
    gt_image = gt_image.crop((x, y, x + px, y + py))
    
    if px > 512 or py > 512:
        input_image = input_image.resize((512, 512))
        gt_image = gt_image.resize((512, 512))
    return input_image, gt_image

def image_large(image_path):
    width, height = Image.open(image_path).size
    return width >= 512 and height >= 512

# In[ ]:


def get_paths(folder):
    file_paths = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if image_large(file_path):
            file_paths.append(file_path)
    file_paths = sorted(file_paths)
    return file_paths


# In[ ]:


class MyDataset(data.Dataset):
    def __init__(self, dataDir, is_train=False):
        super(MyDataset, self).__init__()
        self.is_train = is_train
        
        I_hazyDir = os.path.join(dataDir, 'IndoorTrainHazy')
        I_gtDir = os.path.join(dataDir, 'IndoorTrainGT')
        O_hazyDir = os.path.join(dataDir, 'OutdoorTrainHazy')
        O_gtDir = os.path.join(dataDir, 'OutdoorTrainHazy')
        
        self.input_paths = get_paths(I_hazyDir) + get_paths(O_hazyDir)
        self.gt_paths = get_paths(I_gtDir) + get_paths(O_gtDir)
        self.num_samples = len(self.input_paths)
    
    def form_image_pair(self, idx):
        input_image = Image.open(self.input_paths[idx]).convert('RGB')
        gt_image = Image.open(self.gt_paths[idx]).convert('RGB')

        return input_image, gt_image
    
    def __getitem__(self, idx):
        input_image, gt_image = self.form_image_pair(idx)

        if self.is_train:
            input_image, gt_image = extract_patch(input_image, gt_image)
            input_image, gt_image = augmentation(input_image, gt_image)

        input_image = rgb_to_tensor(input_image)
        gt_image = rgb_to_tensor(gt_image)

        return input_image, gt_image

    def __len__(self):
        return self.num_samples

