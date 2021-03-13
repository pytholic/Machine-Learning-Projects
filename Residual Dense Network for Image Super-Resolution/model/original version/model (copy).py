import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Net1(nn.Module):
    """ Example Net1 Code Format """
    def __init__(self, args):
        super(Net1, self).__init__()
        self.args = args
        n_feats = self.args.nFeat # 64
        kernel_size = 3
        reduction = 4
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(args.nChannel, n_feats, 7)] # k7n64s1
        modules_head.append(act) # ReLU

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size)) # k3n64s1

        # define tail module
        modules_tail = [conv(n_feats, n_feats*4, 3),
            nn.PixelShuffle(2), act,
            conv(n_feats, args.nChannel, 7)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x
