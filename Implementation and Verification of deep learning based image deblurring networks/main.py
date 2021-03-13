#!/usr/bin/env python
# coding: utf-8

# In[32]:


'''Main Code'''

#Import libraries and scripts

import torch
import torch.nn as nn

from torch.nn import init
from torch.autograd import Variable
import argparse
import torch.optim as optim
import math
import ast
from PIL import Image
import numpy as np

from ssim import ssim as compare_ssim
from msssim import msssim as compare_mssim
from utils import compute_psnr as compare_psnr

from model import model
from data import Gopro
from utils import *

from tqdm.notebook import tqdm as tqdm_notebook
import time


# In[33]:


#Check if gpu is available
torch.cuda.is_available()


# # Experimental setup and argument parsing

# In[34]:


#Using easydict instead of argparser because I am using notebook

from easydict import EasyDict as edict

args = edict()

# Training data
args.data_dir = '/home/trojan/Desktop/Image restoration/Homeworks/HW2/dataset/GOPRO_Large/train'   # train dataset directory
args.save_dir = '/home/trojan/Desktop/Image restoration/Homeworks/HW2/result'   # directory to save data

# Model
args.exp_name = 'Net_single+skip'   # model to be selected
args.finetuning = False   # to finetune the training
args.load = 'NetFinal'

# Training
args.patch_size = 256   # training patch size
args.batch_size = 16   # input batch size

#Validation data
args.val_data = True
args.val_batch_size = 1   # batch size for validation data
args.n_threads = 8   # threads number for loading data

# Testing data
args.test_dir = '/home/trojan/Desktop/Image restoration/Homeworks/HW2/dataset/GOPRO_Large/test'   # test dataset directory
args.save = True
args.padding = 8

# Network
args.skip = True   # to use long skip connection
args.multi = False   # to use multi-scale model
args.num_features = 64   # number of feature maps
args.num_resblocks = 9   # number of residual blocks to use

# Optimization
args.lr = 1e-3   # learning rate for the optimizer
args.epochs = 50 #800   # number of training epochs
#args.lr_step_size = 600   # decay learning rate after N epochs
args.lr_gamma = 0.1   # learning rate decay factor
args.lr_decay = 30   #number of epochs to drop lr
args.decay_type = 'inv' #lr decay type
args.loss_type = 'MSE'   #Loss type

args.period = 1
args.gpu = True   # gpu index


# In[35]:


#Basic Settings
if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[36]:


#Check cuda device
device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
print (device)


# # Define helper function

# In[37]:


def get_dataset(args):
    data_train = Gopro(args.data_dir, patch_size=args.patch_size, is_train=True, multi=args.multi)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                                             drop_last=True, shuffle=True, num_workers=int(args.n_threads), pin_memory=False)
    return dataloader

def get_testdataset(args):
    data_test = Gopro(args.test_dir, patch_size=args.patch_size, is_train=False, multi=args.multi)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.n_threads), pin_memory=False)
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


# In[38]:


# This will be used to get validation results using test dataset

def validation(model, dataloader, multi):
    total_psnr = 0
    for batch, images in tqdm_notebook(enumerate(dataloader)):
        with torch.no_grad():
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            if multi:
                input_b2 = Variable(images['input_b2'].cuda())
                input_b3 = Variable(images['input_b3'].cuda())
                output_l1, _, _ = model((input_b1, input_b2, input_b3))
            else:
                output_l1 = model(input_b1)

        output_l1 = tensor_to_rgb(output_l1)
        target_s1 = tensor_to_rgb(target_s1)

        # compute psnr using function from utils
        psnr = compute_psnr(target_s1, output_l1)
        total_psnr += psnr

    return total_psnr / (batch + 1)


# # Training

# In[39]:


def train(args):
    print(args)
    if args.multi:
        net_model = model.MultiScaleNet(num_features=args.num_features, num_resblocks=args.num_resblocks, is_skip=args.skip)
    else:
        net_model = model.SingleScaleNet(num_features=args.num_features, num_resblocks=args.num_resblocks, is_skip=args.skip)
    net_model = net_model.cuda()
    loss_function = set_loss(args)
    loss_function.cuda()
    
    last_epoch = 0
    loss_values = []
    
    save = SaveData(args.save_dir, args.exp_name, args.finetuning)
    save.save_params(args)
    num_params = count_parameters(net_model)
    save.save_log(str(num_params))


    if args.finetuning:
        net_model, last_epoch = save.load_model(net_model)

    start_epoch = last_epoch
    total_loss = 0
    total_time = 0
    
    # load dataset
    dataloader = get_dataset(args)
    testdataloader = get_testdataset(args)

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        optimizer = optim.Adam(net_model.parameters(), lr=args.lr)
        learning_rate = set_lr(args, epoch, optimizer)
        print("Epoch {}/{}".format(epoch + 1, args.epochs))

        total_loss_ = 0
        loss = 0
        for batch, images in tqdm_notebook(enumerate(dataloader)):
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            if args.multi:
                input_b2 = Variable(images['input_b2'].cuda())
                target_s2 = Variable(images['target_s2'].cuda())
                input_b3 = Variable(images['input_b3'].cuda())
                target_s3 = Variable(images['target_s3'].cuda())
                output_l1, output_l2, output_l3 = net_model((input_b1, input_b2, input_b3))
                loss = (loss_function(output_l1, target_s1)
                        + loss_function(output_l2, target_s2)
                        + loss_function(output_l3, target_s3)) / 3
            else:
                output_l1 = net_model(input_b1)
                loss = loss_function(output_l1, target_s1)

            net_model.zero_grad()
            loss.backward()
            optimizer.step()
            #tqdm._instances.clear()
            total_loss_ += loss.data.cpu().numpy()

        total_loss = total_loss_ / (batch + 1)
        loss_values.append(total_loss)
        save.add_scalar('train/loss', total_loss, epoch)
        end = time.time()
        epoch_time = (end - start)
        total_time = total_time + epoch_time           
        
        if  (epoch + 1) % args.period == 0:

            if args.val_data:
                net_model.eval()
                psnr = validation(net_model, testdataloader, args.multi)
                net_model.train()

                log = "Epoch {}/{} \t Learning rate: {:.5f} \t Train total_loss: {:.5f} \t * Val_PSNR: {:.2f} \t Time: {:.4f}\n".format(
                    epoch + 1, args.epochs, learning_rate, total_loss, psnr, total_time)
                print(log)
                save.save_log(log)
                save.add_scalar('valid/psnr', psnr, epoch)
            save.save_model(net_model, epoch)
            total_time = 0
        
        else:
            log = "Epoch {}/{} \t Learning rate: {:.5f} \t Train total_loss: {:.5f} t Time: {:.4f} \t Total time: {:.4f}\n".format(epoch + 1, args.epochs,
                                                                                              learning_rate, total_loss, total_time)
            print(log)
            save.save_log(log)
            save.save_model(net_model, epoch)
            total_time = 0


# In[9]:


if __name__ == '__main__':
    train(args)


# # Testing

# In[40]:


def load_params(args):
    path_params_log = os.path.join(args.save_dir, args.exp_name, "params.txt")
    with open(path_params_log, 'r') as f:
        str_params = f.read().strip()
    return ast.literal_eval(str_params)


# In[41]:


def test(args):

    params = load_params(args)   
    if params['multi']:   # check multi parameter to see whether to use multi model or single
        test_model = model.MultiScaleNet(num_features=params['num_features'], num_resblocks=params['num_resblocks'],
                                       is_skip=params['skip'])
    else:
        test_model = model.SingleScaleNet(num_features=params['num_features'], num_resblocks=params['num_resblocks'],
                                       is_skip=params['skip'])
    
    # load the saved model
    test_model.load_state_dict(torch.load(os.path.join(args.save_dir, args.exp_name, 'model', 'model_lastest.pt')))
    test_model.cuda()
    test_model.eval()
    
    testdataloader = get_testdataset(args)
    
    if args.save:
        output_dir = os.path.join(args.save_dir, args.exp_name, 'test_output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    log_file = open(os.path.join(output_dir, 'test_logs.txt'), 'w')

    total_psnr, total_ssim, total_mssim, count = 0, 0, 0, 0
    for batch, images in enumerate(testdataloader):
        count += 1
        with torch.no_grad():
            input_b1 = Variable(images['input_b1'].cuda())
            target_s1 = Variable(images['target_s1'].cuda())

            if params['multi']:
                input_b2 = Variable(images['input_b2'].cuda())
                input_b3 = Variable(images['input_b3'].cuda())
                output_l1, _, _ = test_model((input_b1, input_b2, input_b3))
            else:
                output_l1 = test_model(input_b1)

        output_l1 = tensor_to_rgb(output_l1)
        target_s1 = tensor_to_rgb(target_s1)

        p = args.padding
        if p != 0:
            img1 = output_l1[:, p:-p, p:-p].squeeze()
            img2 = target_s1[:, p:-p, p:-p].squeeze()
        else:
            img1 = output_l1.squeeze()
            img2 = target_s1.squeeze()

        # Calculate psnr, ssim, mssim using libraries
        with torch.no_grad():
            mssim = compare_mssim(torch.from_numpy(img1[None]).cuda(),
                                  torch.from_numpy(img2[None]).cuda()).cpu().numpy()
            ssim = compare_ssim(torch.from_numpy(img1[None] / 255.0).cuda(),
                                torch.from_numpy(img2[None] / 255.0).cuda()).cpu().numpy()
        psnr = compare_psnr(img1, img2)

        total_psnr += psnr
        total_ssim += ssim
        total_mssim += mssim

        if args.save:
            out = Image.fromarray(np.uint8(output_l1.transpose(1, 2, 0)), mode='RGB')  # output of SRCNN
            out.save(os.path.join(output_dir, 'DB_{:04d}.png'.format(count)))

        log = 'Image {:04d} - PSNR {:.2f} - SSIM {:.4f} - MSSIM {:.4f}'.format(count, psnr, ssim, mssim)
        print(log)
        log_file.write(log + "\n")

    avg_psnr = total_psnr / (batch + 1)
    avg_ssim = total_ssim / (batch + 1)
    avg_mssim = total_mssim / (batch + 1)
    log = 'Average - PSNR {:.2f} dB - SSIM {:.4f} - MSSIM {:.4f}'.format(avg_psnr, avg_ssim, avg_mssim)
    print(log)
    log_file.write(log + "\n")
    log_file.close()

    if args.save:
        print('{:04d} images saved at {}'.format(count, output_dir))


# In[42]:


if __name__ == '__main__':
    test(args)


# In[ ]:




