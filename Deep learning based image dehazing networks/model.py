#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


def ConvInitial(in_channles, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1),
        nn.ReLU(inplace=True)
    )
        
def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True)
    )

def MaxPool():
    return nn.MaxPool2d(2) 

def Upsample(channel):
            return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
    
def DiscConv(in_channels, out_channels):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, padding=1),
    nn.LeakyReLU(0.2, inplace=True)
    )


# In[ ]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        #self.ConvInitial = ConvInitial(3, 64)
        
        # Encoder 
        self.Conv1 = ConvBlock(3, 64)
        self.Pool1 = MaxPool()
        self.Conv2 = ConvBlock(64, 128)
        self.Pool2 = MaxPool()
        self.Conv3 = ConvBlock(128, 256)
        self.Pool3 = MaxPool()
        self.Conv4 = ConvBlock(256, 512)
        self.Pool4 = MaxPool()
        self.Conv5 = ConvBlock(512, 1024)
        self.Pool5 = MaxPool()
                
        # Decoder 
        self.DeConv5 = ConvBlock(1024, 1024)
        self.Upsample5 = Upsample(1024)
        self.DeConv4 = ConvBlock(1024, 512)
        self.Upsample4 = Upsample(512)
        self.DeConv3 = ConvBlock(512, 256)
        self.Upsample3 = Upsample(256)
        self.DeConv2 = ConvBlock(256,128)
        self.Upsample2 = Upsample(128)
        self.DeConv1 = ConvBlock(128,64)
        self.Upsample1 = Upsample(64)
        
        self.ConvFinal = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        #InitialConv = self.ConvInitial(x)
        
        # Encoder Pass
        OpConv1 = self.Conv1(x)
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
        #tensorCombine = torch.cat([OpUpsample5, OpConv5], dim=1)  
        
        OpDeconv4 = self.DeConv4(tensorCombine)
        OpUpsample4 = self.Upsample4(OpDeconv4)
                
        tensorCombine = OpUpsample4 + OpConv4
        #tensorCombine = torch.cat([OpUpsample4, OpConv4], dim=1) 
        
        OpDeconv3 = self.DeConv3(tensorCombine)
        OpUpsample3 = self.Upsample3(OpDeconv3)
                
        tensorCombine = OpUpsample3 + OpConv3
        #tensorCombine = torch.cat([OpUpsample3, OpConv3], dim=1) 
        
        OpDeconv2 = self.DeConv2(tensorCombine)
        OpUpsample2 = self.Upsample2(OpDeconv2)
        
        tensorCombine = OpUpsample2 + OpConv2
        #tensorCombine = torch.cat([OpUpsample2, OpConv2], dim=1) 
        
        OpDeconv1 = self.DeConv1(tensorCombine)
        OpUpsample1 = self.Upsample1(OpDeconv1)
        
        tensorCombine = OpUpsample1 + OpConv1
        #tensorCombine = torch.cat([OpUpsample1, OpConv1], dim=1) 
        
        out = self.ConvFinal(tensorCombine)
        
        dehaze = self.tanh(out)
        
        return dehaze


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, in_channel=3, num_filters=48):
        super(Discriminator, self).__init__()
        
        self.Pass1 = DiscConv(in_channel, num_filters)
        self.Pass2 = DiscConv(num_filters, num_filters*2)
        self.Pass3 = DiscConv(num_filters*2, num_filters*4)
        self.Pass4 = DiscConv(num_filters*4, num_filters*8)
        
        self.PassFinal = nn.Conv2d(num_filters*8, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        pass1 = self.Pass1(x)
        pass2 = self.Pass2(pass1)
        pass3 = self.Pass3(pass2)
        pass4 = self.Pass4(pass3)
        
        out = self.PassFinal(pass4)
        out = self.sigmoid(out)
        
        return out
        
    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req

