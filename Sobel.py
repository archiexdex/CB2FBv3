import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
#np.set_printoptions(threshold=np.nan)
from math import exp
import cv2
import matplotlib.pyplot as plt

def sobel(kernel_size):
    assert kernel_size in [3,5,7]
    ind = kernel_size//2
    matx = torch.zeros((kernel_size, kernel_size))
    maty = torch.zeros((kernel_size, kernel_size))
    for j in range(-ind,ind+1):
        for i in range(-ind,ind+1):
            matx[j][i] = i / (i*i+j*j) if i*i+j*j>0 else 0
            maty[j][i] = j / (i*i+j*j) if i*i+j*j>0 else 0

    if kernel_size==3:
        mult = 2
    elif kernel_size==5:
        mult = 20
    elif kernel_size==7:
        mult = 780

    matx *= mult
    maty *= mult

    return matx, maty

def create_window(kernel_size, channel):
    windowx, windowy = sobel(kernel_size)
    #windowx, windowy = map(lambda x: x.view(1,1,kernel_size,kernel_size), sobel(kernel_size)).repeat(channel,channel,1,1), sobel(kernel_size))
    # windowx [kernel_size, kernel_size]
    windowx, windowy = windowx.view(1,1,kernel_size,kernel_size), windowy.view(1,1,kernel_size,kernel_size)
    # windowx [1, 1, kernel_size, kernel_size]
    windowx, windowy = windowx.repeat(channel,channel,1,1), windowy.repeat(channel,channel,1,1)
    # windowx [out_channel, in_channel, kernel_size, kernel_size]
    return windowx,windowy

def gradient(img, windowx, windowy, padding):
    gradx = F.conv2d(img, windowx, padding=padding)
    grady = F.conv2d(img, windowy, padding=padding)
    return gradx, grady

class SobelGrad(torch.nn.Module):
    def __init__(self, kernel_size=3, channel=1, device='cuda'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding= kernel_size >> 1
        self.channel = channel
        self.windowx, self.windowy = map(lambda x: x.to(device), create_window(kernel_size, channel))

    def forward(self, x):
        gradx, grady = gradient(x, self.windowx, self.windowy, self.padding)
        return gradx, grady
