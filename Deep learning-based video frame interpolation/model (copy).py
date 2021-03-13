#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
import sepconv
import sys
from torch.nn import functional as F


# In[ ]:


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
args.epochs = 10   # number of training epochs
#args.lr_step_size = 600   # decay learning rate after N epochs
#args.lr_gamma = 0.1   # learning rate decay factor
args.lr_decay = 100   #number of epochs to drop lr
args.decay_type = 'step' #lr decay type
args.loss_type = 'L1'   #Loss type

args.period = 1
args.gpu = True   # gpu index


# In[ ]:


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

# Default convolution layer to keep same kernel, padding and stride
def default_conv(in_channel, out_channel):
    return nn.Conv2d(
        in_channels=in_channel, out_channels=out_channel, 
        kernel_size=3, stride=1, padding=1
    )

def avg_pool():
    return nn.AvgPool2d(kernel_size=2, stride=2)


# In[ ]:


# Kernel estmation network
class Net(torch.nn.Module):
    def __init__(self, kernel_size):
        super(Net, self).__init__()
        self.kernel_size = args.kernel_size
        
        def SimpleConvBlock(in_channel, out_channel):
            return nn.Sequential(
            default_conv(in_channel, out_channel),
            nn.ReLU(inplace=False),
            default_conv(out_channel, out_channel),
            nn.ReLU(inplace=False),
            default_conv(out_channel, out_channel),
            nn.ReLU(inplace=False)
            )

        
        def Upsample(channel):
            return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            default_conv(in_channel=channel, out_channel=channel),
            nn.ReLU(inplace=False)
            )

            
        def SubNet(ks):
            return nn.Sequential(
            default_conv(in_channel=64, out_channel=64),
            nn.ReLU(inplace=False),
            default_conv(in_channel=64, out_channel=64),
            nn.ReLU(inplace=False),
            default_conv(in_channel=64, out_channel=ks),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            default_conv(in_channel=ks, out_channel=ks)
            )
                
        
        # Encoder 
        self.Conv1 = SimpleConvBlock(6, 32)
        self.Pool1 = avg_pool()
        self.Conv2 = SimpleConvBlock(32, 64)
        self.Pool2 = avg_pool()
        self.Conv3 = SimpleConvBlock(64, 128)
        self.Pool3 = avg_pool()
        self.Conv4 = SimpleConvBlock(128, 256)
        self.Pool4 = avg_pool()
        self.Conv5 = SimpleConvBlock(256, 512)
        self.Pool5 = avg_pool()
                
        # Decoder 
        self.DeConv5 = SimpleConvBlock(512, 512)
        self.Upsample5 = Upsample(512)
        self.DeConv4 = SimpleConvBlock(512, 256)
        self.Upsample4 = Upsample(256)
        self.DeConv3 = SimpleConvBlock(256, 128)
        self.Upsample3 = Upsample(128)
        self.DeConv2 = SimpleConvBlock(128, 64)
        self.Upsample2 = Upsample(64)
                
        # Subnets
        self.Vertical1 = SubNet(self.kernel_size)
        self.Vertical2 = SubNet(self.kernel_size)
        self.Horizontal1 = SubNet(self.kernel_size)
        self.Horizontal2 = SubNet(self.kernel_size)

                
    # Define forward method
    
    def forward(self, tensor1, tensor3):
        
        # Join the two tensors
        tensorCat = torch.cat([tensor1, tensor3], 1)
        
        # Encoder Pass
        OpConv1 = self.Conv1(tensorCat)
        OpPool1 = self.Pool1(OpConv1)
        
        OpConv2 = self.Conv2(OpPool1)
        OpPool2 = self.Pool2(OpConv2)
                
        OpConv3 = self.Conv3(OpPool2)
        OpPool3 = self.Pool3(OpConv3)
                
        OpConv4 = self.Conv4(OpPool3)
        OpPool4 = self.Pool4(OpConv4)
                
        OpConv5 = self.Conv5(OpPool4)
        OpPool5 = self.Pool5(OpConv5)
                
        # Decoder Pass
        OpDeconv5 = self.DeConv5(OpPool5)
        OpUpsample5 = self.Upsample5(OpDeconv5)
                
        tensorCombine = OpUpsample5 + OpConv5
                
        OpDeconv4 = self.DeConv4(tensorCombine)
        OpUpsample4 = self.Upsample4(OpDeconv4)
                
        tensorCombine = OpUpsample4 + OpConv4
                
        OpDeconv3 = self.DeConv3(tensorCombine)
        OpUpsample3 = self.Upsample3(OpDeconv3)
                
        tensorCombine = OpUpsample3 + OpConv3
                
        OpDeconv2 = self.DeConv2(tensorCombine)
        OpUpsample2 = self.Upsample2(OpDeconv2)
                
        tensorCombine = OpUpsample2 + OpConv2
                                 
        # Subnet Pass
        Ver1 = self.Vertical1(tensorCombine)
        Ver2 = self.Vertical2(tensorCombine)
        Hor1 = self.Horizontal1(tensorCombine)
        Hor2 = self.Horizontal2(tensorCombine)
        
        return Ver1, Ver2, Hor1, Hor2


# In[ ]:


class ConvNetSep(torch.nn.Module):
    def __init__(self, kernel_size):
        super(ConvNetSep, self).__init__()
        
        # Pass the arguments
        self.kernel_size = args.kernel_size
        self.kernel_padding = int(math.floor(kernel_size / 2.0))
        self.estimate_kernel = Net(self.kernel_size)
        self.epoch = args.epochs
        self.optimizer = optim.Adam(self.parameters(), lr = args.lr)
        self.criterion = set_loss(args)
        
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_padding, self.kernel_padding, 
                                                    self.kernel_padding, self.kernel_padding])
        
    
    def forward(self, Frame1, Frame3):
        h_1 = int(list(Frame1.size())[2])
        w_1 = int(list(Frame1.size())[3])
        h_3 = int(list(Frame3.size())[2])
        w_3 = int(list(Frame3.size())[3])
        
        # Make sure frame size is same
        if h_1 != h_3 or w_1 != w_3:
            sys.exit('Size mismatch')
    
        h_pad = False
        w_pad = False
        
        if w_1 % 32 != 0:
            pad_w = 32 - (w_1 % 32)
            Frame1 = F.pad(Frame1, (0, pad_w, 0, 0))
            Frame3 = F.pad(Frame3, (0, pad_w, 0, 0))
            w_pad = True
            
        if h_1 % 32 != 0:
            pad_h = 32 - (h_1 % 32)
            Frame1 = F.pad(Frame1, (0, 0, 0, pad_h))
            Frame3 = F.pad(Frame3, (0, 0, 0, pad_h))
            h_pad = True

        Ver1, Hor1, Ver2, Hor2 = self.estimate_kernel(Frame1, Frame3)
        
        tenDot1 = sepconv.FunctionSepconv()(self.modulePad(Frame1), Ver1, Hor1)
        tenDot2 = sepconv.FunctionSepconv()(self.modulePad(Frame3), Ver2, Hor2)
        
        Frame2 = tenDot1 + tenDot2
        
        if h_pad:
            Frame2 = Frame2[:, :, 0:h_1, :]
        if w_pad:
            Frame2 = Frame2[:, :, :, 0:w_1]
            
        return Frame2
    
    
