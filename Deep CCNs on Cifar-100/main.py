#!/usr/bin/env python
# coding: utf-8

# # AI502/KSE527, Homework 02

# In[1]:


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchsummary import summary
from tqdm import tqdm

from ptflops import get_model_complexity_info 

get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib.pyplot as plt
from matplotlib import style
## Reference of ptflops: https://github.com/sovrasov/flops-counter.pytorch


# # CIFAR100

# In[2]:


# Global Variable For training
# You just use the following hyper-parameters
BATCH_SIZE = 128
NUM_EPOCH = 100
LEARNING_RATE = 0.01
CRITERION = nn.CrossEntropyLoss()


# In[3]:


# CIFAR100 Dataset
train_dataset = dsets.CIFAR100(root='./data', train=True, 
                              transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]), download=True)
test_dataset = dsets.CIFAR100(root='./data', train=False,
                             transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[4]:


for i in train_loader:
    print (i[0][0].shape)
    break


# # Fit / Eval function

# In[5]:


def fit(model,train_loader):
    model.train()
    device = next(model.parameters()).device.index
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #losses = []
    #for epoch in range(NUM_EPOCH):
    for i, data in enumerate(train_loader):
            #for t in tqdm(train_loader):

        image = data[0].type(torch.FloatTensor).cuda(device)
        label = data[1].type(torch.LongTensor).cuda(device)

        pred_label = model(image)
        loss = CRITERION(pred_label, label)
            #losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            #tqdm._instances.clear()
    #print(f"Epoch: {epoch}. Loss: {loss}")
    #losses.append(loss)
    return (loss)
    #plt.plot(losses)
    #plt.show()
    #print(losses)
    avg_loss = sum(losses)/len(losses)   
    print (f"Avg Loss = {avg_loss}")
    


# In[6]:


def eval(model, test_loader):
    model.eval()
    device = next(model.parameters()).device.index
    pred_labels = []
    real_labels = []
    
    for i, data in enumerate(test_loader):
        image = data[0].type(torch.FloatTensor).cuda(device)
        label = data[1].type(torch.LongTensor).cuda(device)
        real_labels += list(label.cpu().detach().numpy())
        
        pred_label = model(image)
        pred_label = list(pred_label.cpu().detach().numpy())
        pred_labels += pred_label
    
    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.argmax(axis=1)
    acc = sum(real_labels==pred_labels)/len(real_labels)*100
    #accuracies.append(acc)
    #plt.plot(accuracies)
    #plt.show()
    return acc


# In[7]:


from datetime import datetime
def run(model):
    
    start_time = datetime.now()
    accuracies = []
    losses = []

    for epoch in range(NUM_EPOCH):
        x = fit(model, train_loader)
        y = eval(model, test_loader)
        #losses.append (loss)
        #accuracies.append (acc)
        #return x, y
        losses.append(x)
        accuracies.append(y)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    fig = plt.figure() 
    plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 12})

    #define axis, each axis is a graph
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(losses, label="training_loss")
    ax1.legend(loc=2)
    #ax1.plot(accuracies, label="test_accuracy")
    #ax1.legend(loc=2)
    ax2.plot(accuracies, label="test_accuracy")
    ax2.legend(loc=2)
    #ax2.legend(loc=2)
    plt.show() 
    #fig.savefig(('/home/trojan/Desktop/defected_area2.png'))
    
    return losses, accuracies


# # Construct the 18 layers network
# 
# All of PlainNet18, ResNet18, MobileNet18 have same network structure but they are consist of different convolution block (PlainBlock, ResidualBlock, MobileBlock). You have to utilize Net18() when you define the network.

# In[8]:


# Example

#plainnet_model = Net18(PlainBlock, [2, 2, 2, 2], 100).cuda()
# resnet_model = Net18(ResidualBlock, [2, 2, 2, 2], 100).cuda()
# mobilenet_model = Net18(MobileBlock, [2, 2, 2, 2], 100).cuda()


# In[9]:


class Net18(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(Net18, self).__init__()
        self.inp = 64
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, oup, num_block, stride=1):
        layers = []
        strides = [stride] + [1]*(num_block-1)
        for stride in strides:
            layers.append(block(self.inp, oup, stride))
            self.inp = oup
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn0(self.conv0(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# # Task1 : Implement 18-layer CNNs for CIFAR100

# In[10]:


class PlainBlock(nn.Module):
    #expansion = 1
    def __init__(self, inp, oup, stride=1):
        super(PlainBlock, self).__init__()
        
        #####################################
        
        # Write down your own code

        self.plain_function = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(oup)
        )

        
        
        #####################################
        
    def forward(self, x):
        
        #####################################
        
        # Write down your own code
        out = nn.ReLU(inplace=True)(self.plain_function(x))
        #####################################
        
        return out


# In[11]:


plainnet_model = Net18(PlainBlock, [2, 2, 2, 2], 100).cuda()


# In[12]:


print (plainnet_model)


# In[13]:


torch.cuda.is_available()


# In[14]:


device = torch.device("cuda:0")
device


# In[15]:


#.to(device) tensor to device)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


# In[16]:


run(plainnet_model)


# # Task2 : Implement ResNet18 for CIFAR100

# In[17]:


class ResidualBlock(nn.Module):
    """Residual Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup * ResidualBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(oup * ResidualBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or inp != ResidualBlock.expansion * oup:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup * ResidualBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup * ResidualBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# In[18]:


resnet_model = Net18(ResidualBlock, [2, 2, 2, 2], 100).cuda()


# In[19]:


print(resnet_model)


# In[20]:


run(resnet_model)


# # Task3 : Implement MobileNet for CIFAR100

# In[21]:


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin 
                         a network uniformly at each layer. For a given 
                         layer and width multiplier α, the number of 
                         input channels M becomes αM and the number of 
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
       super().__init__()

       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       self.fc = nn.Linear(int(1024 * alpha), class_num)
       self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)


# In[22]:


net = MobileNet().cuda()
print(net)


# In[23]:


run(net)


# In[37]:


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = MobileNet()
  macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[49]:


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(resnet_model, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[43]:


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(plainnet_model, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

