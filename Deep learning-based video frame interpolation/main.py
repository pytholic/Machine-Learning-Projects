#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Main Code'''

#Import libraries and scripts

import torch
import torch.nn as nn
import io

from torch.nn import init
from torch.autograd import Variable
import argparse
import torch.optim as optim
import math
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from math import log10
from torchvision.utils import save_image as imwrite

from model import ConvNetSep
from torchvision import transforms
import os

import data
from data import triplet_train, triplet_test

import model
import warnings
warnings.filterwarnings("ignore")

from tqdm.notebook import tqdm as tqdm_notebook
import time


# In[2]:


#Check if gpu is available
torch.cuda.is_available()


# # Experimental setup and argument parsing

# In[9]:


#Using easydict instead of argparser because I am using notebook

from easydict import EasyDict as edict

args = edict()

# Training data
args.data_dir = '/home/trojan/Desktop/Image restoration/Homeworks/HW3/dataset/Dataet_VFI_HW3/Vimeo90K_HW3'  # training data 
args.save_dir = '/home/trojan/Desktop/Image restoration/Homeworks/HW3/result'   # results directory

# Model
args.exp_name = 'Net_SepConv'   # model to be selected
args.finetuning = False   # to finetune the training
args.load = None #'NetFinal'

#Validation data
args.val_data = True
args.val_batch_size = 1   # batch size for validation data
#args.n_threads = 8   # threads number for loading data'''

# Testing data
args.test_dir = '/home/trojan/Desktop/Image restoration/Homeworks/HW3/dataset/Dataet_VFI_HW3/ucf101_HW3'   # test dataset directory
args.save = True

# Training and Optimization
args.patch_size = 128
args.batch_size = 8
args.kernel_size = 25
args.lr = 1e-4   # learning rate for the optimizer
args.epochs = 200   # number of training epochs
#args.lr_step_size = 600   # decay learning rate after N epochs
#args.lr_gamma = 0.1   # learning rate decay factor
args.lr_decay = 100   #number of epochs to drop lr
args.decay_type = 'step' #lr decay type
args.loss_type = 'L1'   #Loss type

args.period = 5
args.gpu = True   # gpu index


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


# # Define helper function

# In[6]:


transform = transforms.Compose([transforms.ToTensor()])

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

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

def inc_epoch(self):
    self.epoch += 1


# In[11]:


def test(model, Dataset, save_dir=args.save_dir, output_name='output.png'):
        test_dir = args.test_dir
        patch_size = args.patch_size
         
        avg_psnr = 0    
        test_loader = DataLoader(dataset=Dataset, batch_size=1, shuffle=False, num_workers=0)

        total_batches = 0 
        for batch_index, (Frame1, Frame2, Frame3) in enumerate(test_loader):
            Frame1 = to_variable(Frame1)
            Frame2 = to_variable(Frame2)
            Frame3 = to_variable(Frame3)
            frame_out = model(Frame1, Frame3)
            gt = Frame2
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            avg_psnr += psnr
            imwrite(frame_out, save_dir + '/' + '{}'.format(batch_index) + output_name, range=(0, 1))
            
            msg = "Batch: {}\t PSNR: {:<20.4f}\n".format(batch_index, psnr)
            total_batches += 1
            #print(msg, end='')
            #logfile.write(msg)
            
        avg_psnr /= total_batches
        return avg_psnr


# # Training

# In[8]:


def train(args):
    
    train_dir = args.data_dir
    save_dir = args.save_dir
    test_dir = args.test_dir
    batch_size = args.batch_size
    total_epochs = args.epochs
    patch_size = args.patch_size
    
    loss_function = set_loss(args)
    loss_function.cuda()
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ckptDir = save_dir + '/chekpoint'
    resultDir = save_dir 
    
    if not os.path.exists(ckptDir):
        os.makedirs(ckptDir)
        
    dataset = triplet_train(train_dir, resize=(patch_size, patch_size))
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = triplet_test(test_dir, resize=None)
    test_resultDir = save_dir + '/test_result'
    
    test_output_dir = test_resultDir
    
    if not os.path.exists(test_resultDir):
        os.makedirs(test_resultDir)
    
    log_dir = save_dir + '/logging'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if os.path.exists(log_dir + '/log.txt'):
        logFile = open(log_dir + '/log.txt', 'a')
    else:
        logFile = open(log_dir + '/log.txt', 'w')
    
    logFile.write('Batch size: ' + str(batch_size) + '\n')
    
    if args.load is not None:
        checkpoint = torch.load(args.load)
        kernel_size = checkpoint['kernel_size']
        model = ConvNetSep(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = args.kernel_size
        model = ConvNetSep(kernel_size=kernel_size)

    logFile.write('Kernel size: ' + str(kernel_size) + '\n')
    num_params = count_parameters(model)
    logFile.write('Parameters: ' + str(num_params) + '\n')
    
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    start_epoch = 0
    epoch_time = 0
    total_time = 0
    max_step = train_loader.__len__()
    
    #model.eval()
    #test(model, test_dataset, start_epoch, test_output_dir, logfile, output_name = 'output.png')
    
    
    
    for epoch in range(start_epoch, total_epochs):
        start = time.time()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        learning_rate = set_lr(args, epoch, optimizer)
        print("Epoch {}/{}".format(epoch + 1, total_epochs))
        
        model.train()
        for batch_index, (Frame1, Frame2, Frame3) in enumerate(train_loader):
            Frame1 = to_variable(Frame1)
            Frame2 = to_variable(Frame2)
            Frame3 = to_variable(Frame3)
            output = model(Frame1, Frame3)
            loss = loss_function(output, Frame2)

            model.zero_grad()
            loss.backward()
            optimizer.step()  
            
        end = time.time()
        epoch_time = (end - start)
        total_time += epoch_time
        #model.inc_epoch()
        
        #if batch_index % 100 == 0:
        #if  (epoch + 1) % args.period == 0:
        log = "Epoch {}/{} \t Learning rate: {:.5f} \t Train Loss: {:.5f} \t Epoch Time: {:.4f} \t Total time: {:.4f}\n".format(epoch + 1, total_epochs,
                                                                                                                             learning_rate, loss, epoch_time, total_time)
        print(log)
        logFile.write(log)
                
        if  (epoch + 1) % args.period == 0:
                
            if args.val_data:
                torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, ckptDir + '/model_epoch' + str(model.epoch).zfill(3) + '.pth')
                model.eval()
                avg_psnr = test(model, test_dataset, epoch, test_output_dir, output_name = 'output.png')
                msg = 'Average PSNR: {:<20.4f}'.format(avg_psnr) + '\n'
                print(msg)
                logFile.write(msg)
                logFile.write('\n\n')
        logFile.flush()
        
    logFile.close()

       


# In[9]:


if __name__ == '__main__':
    train(args)


# # Testing

# In[12]:


print("Reading Test DB...")
test_dataset = triplet_test(args.test_dir, resize=None)

print("Loading the Model...")
ckptDir = args.save_dir + '/chekpoint'
checkpoint = torch.load(ckptDir + '/model_epoch010.pth', map_location=torch.device('cpu'))
model = ConvNetSep(kernel_size=args.kernel_size)
model = model.cuda()
model.load_state_dict(checkpoint['state_dict'])
model.epoch = checkpoint['epoch']


# In[13]:


test_resultDir_final = args.save_dir + '/test_result_final'
if not os.path.exists(test_resultDir_final):
        os.makedirs(test_resultDir_final)


# In[14]:


print("Test Start...")
test_psnr = test(model, test_dataset, test_resultDir_final, output_name='output.png')
test_log = "PSNR on test data: {}".format(test_psnr)
print(test_log)

log_dir = args.save_dir + '/logging'

if os.path.exists(log_dir + '/log_test.txt'):
    logfile = open(log_dir + '/log_test.txt', 'a')
else:
    logfile = open(log_dir + '/log_test.txt', 'w')
        
logfile.write(test_log)
logfile.close()


# In[ ]:




