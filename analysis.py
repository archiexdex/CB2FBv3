import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_1d(img1, img2):

    """
        Assume type of img1, img2 are torch.Tensor()
        Dimesion of img1 = [batch, channel, row, column], so as img2
    """
    mu1 = img1.mean(dim=-1).mean(dim=-1).mean(dim=-1)
    mu2 = img2.mean(dim=-1).mean(dim=-1).mean(dim=-1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = img1.pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1) - mu1_sq
    sigma2_sq = img2.pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1) - mu2_sq
    sigma12 = (img1*img2).mean(dim=-1).mean(dim=-1).mean(dim=-1) - mu1*mu2
    sigma1 = sigma1_sq.sqrt()
    sigma2 = sigma2_sq.sqrt()

    C1 = 1e-2**2 
    C2 = 3e-2**2

    ssim_val = ((2 * mu1 * mu2 + C1)*(2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_val.mean()

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(1/mse)
