#Residual Dense Network Model
#Link to the original paper: https://arxiv.org/pdf/1802.08797.pdf
#Model implemented from figure in original paper

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class create_dense_layer(nn.Module):
  def __init__(self, num_channels, growth_rate, kernel_size=3):
    super(create_dense_layer, self).__init__()
    self.conv = nn.Conv2d(num_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)   #local feature fusion (LFF)
    return out


class conv_sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(conv_sub_pixel, self).__init__()
        layers = []
        layers.append(nn.PixelShuffle(scale))
        self.sub_pixel_layer = nn.Sequential(*layers)
    def forward(self, x):
        x = self.sub_pixel_layer(x)
        return x

    
# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, num_channels, num_dense_layer, growth_rate):
    super(RDB, self).__init__()
    num_channels_ = num_channels
    dense_layers = []
    for i in range(num_dense_layer):    
        dense_layers.append(create_dense_layer(num_channels_, growth_rate))
        num_channels_ += growth_rate 
    self.layer_dense = nn.Sequential(*dense_layers)
    self.conv_1x1 = nn.Conv2d(num_channels_, num_channels, kernel_size=1, padding=0, bias=False)   #local feature fusion (LFF)
  def forward(self, x):
    out = self.layer_dense(x)
    out = self.conv_1x1(out)
    out += x
    return out


# Residual Dense Network
class RDN_model(nn.Module):
    def __init__(self, args):
        super(RDN_model, self).__init__()
        num_channel = args.num_channel
        num_dense_layer = args.num_dense_layer
        num_features = args.num_features
        scale = args.scale
        growth_rate = args.growth_rate
        self.args = args

        #Initial convolution layers
        #Shallow feature extraction
        self.conv1 = nn.Conv2d(num_channel, num_features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)
        
        # Residual Dense Blocks
        self.RDB1 = RDB(num_features, num_dense_layer, growth_rate)
        self.RDB2 = RDB(num_features, num_dense_layer, growth_rate)
        self.RDB3 = RDB(num_features, num_dense_layer, growth_rate)
        
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(num_features*3, num_features, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)
        
        # Upsampler
        self.conv_up = nn.Conv2d(num_features, num_features*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = conv_sub_pixel(scale)
        
        # Final conv layer 
        self.conv3 = nn.Conv2d(num_features, num_channel, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):

        F_1  = self.conv1(x)
        F0 = self.conv2(F_1)
        F1 = self.RDB1(F0)
        F2 = self.RDB2(F1)
        F3 = self.RDB3(F2)     
        F = torch.cat((F1, F2, F3), 1)
        FdLF = self.GFF_1x1(F)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_1
        upscale = self.conv_up(FDF)
        upscale = self.upsample(upscale)

        output = self.conv3(upscale)


        return output



