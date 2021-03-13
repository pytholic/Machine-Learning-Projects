#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Super-Resolution
#HW1
#EE838A
#by Raja Haseeb


# In[2]:


import torch
import torch.nn as nn

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from math import log10

from model import model
from data import DIV2K, Set5
from utils import *
import time


# In[2]:


#Check if gpu is available
torch.cuda.is_available()


# In[3]:


#Using easydict instead of argparser because I am using notebook

from easydict import EasyDict as edict

args = edict()

#Directory paths
args.dataDir = '/home/trojan/Desktop/Image restoration/Homeworks/HW1/Dataset/SR_data/train'   #dataset directory
args.saveDir = '/home/trojan/Desktop/Image restoration/Homeworks/HW1/result'   #datasave directory
args.HR_valDataroot = '/home/trojan/Desktop/Image restoration/Homeworks/HW1/Dataset/SR_data/benchmark/Set5/HR'
args.LR_valDataroot = '/home/trojan/Desktop/Image restoration/Homeworks/HW1/Dataset/SR_data/benchmark/Set5/LR_bicubic/X2'
args.valBatchSize = 5

#Basic options
args.load = 'NetFinal'   #save result
args.model_name = 'RDN_model'   #model to select
args.finetuning = False   #fintuning the training
args.need_patch = True   #get patch from image

#Network options
args.num_dense_layer = 3   #number of dense blocks
args.growth_rate = 32   #growth rate of dense net
args.num_features = 64   #number of feathure maps
args.num_channel = 3   #number of color maps to use
args.patch_size = 64   #patch size (GT)

args.nThreads = 8   #number of threads for data loading
args.batch_size = 16 #batch size for training
args.lr = 1e-3   #learning rate
args.epochs = 200   #number of training epochs
args.lr_decay = 100   #number of epochs to drop lr
args.decay_type = 'inv' #lr decay type
args.loss_type = 'L1'   #Loss type

args.period = 10   #period of evaluation
args.scale = 2   #scale output size /input size
args.gpu = True   #gpu index


# In[4]:


#Basic Settings
if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[5]:


#Check cuda device
device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
print (device)


# In[6]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

def get_dataset(args):
    data_train = DIV2K(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def get_testdataset(args):
    data_test = Set5(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def set_loss(args):
    loss_type = args.loss_type
    if loss_type == 'MSE':
        lossfunction = nn.MSELoss()
    elif loss_type == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction


def set_lr(args, epoch, optimizer):
    lr_decay = args.lr_decay
    decay_type = args.decay_type
    if decay_type == 'step':
        epoch_iter = (epoch + 1) // lr_decay
        lr = args.lr / 2 ** epoch_iter
    elif decay_type == 'exp':
        k = math.log(2) / lr_decay
        lr = args.lr * math.exp(-k * epoch)
    elif decay_type == 'inv':
        k = 1 / lr_decay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[101]:


import matplotlib.pyplot as plt
import cv2

def test(args, model, dataloader):

    avg_psnr = 0
    psnr_val = 0
    n = 0
    for batch, (im_lr, im_hr) in enumerate(dataloader):
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = model(im_lr)

        output = output.cpu()
        output = output.data.squeeze(0)
        
        # denormalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)
        
        #added code to save the resulting images
        output_new = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        RGB_output = cv2.cvtColor(output_new, cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.saveDir + '/image_{}.png'.format(n), RGB_output)
        n += 1
        #output = Image.fromarray(np.uint8(output[0]), mode='RGB')
        #plt.imshow(output)

        # =========== Target Image ===============
        im_hr = im_hr.cpu()
        im_hr = im_hr.data.squeeze(0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(im_hr, mean, std):
            t.mul_(s).add_(m)

        im_hr = im_hr.numpy()
        im_hr *= 255.0
        im_hr = im_hr.clip(0, 255)
        # im_hr = Image.fromarray(np.uint8(im_hr[0]), mode='RGB')

        mse = ((im_hr[:, 8:-8,8:-8] - output[:, 8:-8,8:-8]) ** 2).mean()
        psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        psnr_val = psnr
        avg_psnr += psnr
        
        

    return avg_psnr/args.valBatchSize


# In[8]:


def train(args):
    # Set a Model
    if args.model_name == 'RDN_model':
        my_model = model.RDN_model(args)
    my_model.apply(weights_init)
    my_model.cuda()

    save = saveData(args)

    Numparams = count_parameters(my_model)
    save.save_log(str(Numparams))

    last_epoch = 0
    # fine-tuning or retrain
    if args.finetuning:
        my_model, last_epoch = save.load_model(my_model)

    # load data
    dataloader = get_dataset(args) # [-1,1]
    testdataloader = get_testdataset(args)

    start_epoch = last_epoch
    lossfunction = set_loss(args)
    lossfunction.cuda()
    total_loss = 0
    total_time = 0
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        optimizer = optim.Adam(my_model.parameters()) # optimizer
        learning_rate = set_lr(args, epoch, optimizer)
        total_loss_ = 0
        loss_ = 0
        for batch, (im_lr, im_hr) in enumerate(dataloader):
            im_lr = Variable(im_lr.cuda())
            im_hr = Variable(im_hr.cuda())

            my_model.zero_grad()
            output = my_model(im_lr)
            loss = lossfunction(output, im_hr)
            total_loss = loss
            total_loss.backward()
            optimizer.step()

            loss_ += loss.data.cpu().numpy()
            total_loss_ += loss.data.cpu().numpy()
        loss_ = loss_ / (batch + 1)
        total_loss_ = total_loss_ / (batch + 1)

        end = time.time()
        epoch_time = (end - start)
        total_time = total_time + epoch_time
            
        if (epoch + 1) % args.period == 0:
            my_model.eval()
            avg_psnr = test(args, my_model, testdataloader)
            my_model.train()
            log = "[{} / {}] \tLearning_rate: {:.5f}\t Train total_loss: {:.4f}\t Train Loss: {:.4f} \t Val PSNR: {:.4f} Time: {:.4f}".format(epoch + 1,
                                                                                                                                              args.epochs, learning_rate, total_loss_, loss_, avg_psnr, total_time)
            print(log)
            save.save_log(log)
            save.save_model(my_model, epoch)
            total_time = 0


# In[9]:


if __name__ == '__main__':
    train(args)


# In[102]:


#Load the trained model

my_model = model.RDN_model(args)
my_model.load_state_dict(torch.load(args.saveDir + '/NetFinal/model/model_lastest.pt'))
my_model.cuda()


# In[103]:


#Test the model
test(args, my_model, dataloader = get_testdataset(args))


# In[ ]:




