import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import json
from IQA_pytorch import SSIM, FSIM, DISTS, GMSD, MS_SSIM, LPIPSvgg, CW_SSIM, VIF
from torchvision.utils import save_image

from model import *
from dataset import *
from utils import *
from argparse import Namespace

opt = parse()
load_mode = opt.load_mode
sample_mode = opt.sample_mode
with open(f"{opt.cfg_dir}/{opt.no}.json", "r") as fp:
    opt = json.load(fp)
opt = Namespace(**opt)
same_seeds(opt.seed)
try:
    os.makedirs(f"{opt.rst_dir}/{opt.no}", 0o777)
except:
    pass

dataset = CTDataset(config=opt, isTrain=False)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

model = Model(opt)
model.load(load_mode)
imgs = model.sample_image(dataset, sample_mode=sample_mode)
for idx, img in imgs.items():
    save_image(img, f"{opt.rst_dir}/{opt.no}/{idx}.png")
