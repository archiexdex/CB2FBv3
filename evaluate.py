import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import json
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
if os.path.exists(f"{opt.rst_dir}/{opt.no}"):
    shutil.rmtree(f"{opt.rst_dir}/{opt.no}")
os.makedirs(f"{opt.rst_dir}/{opt.no}", 0o777)

dataset = CTDataset(config=opt, isTrain=False)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

model = Model(opt)
model.load(load_mode)
imgs = model.sample_image(dataset, sample_mode=sample_mode)

for idx, img in imgs.items():
    real_cb, real_fb, fake_fb, mask, grid = img
    # Evaluate images 
    origin_qa = get_qa(real_cb, real_fb, mask)
    output_qa = get_qa(fake_fb, real_fb, mask)
    # Write analysis to csv
    save_csv(f"origin_{idx}", origin_qa, f"{opt.rst_dir}/{opt.no}")
    save_csv(f"output_{idx}", output_qa, f"{opt.rst_dir}/{opt.no}")
    # Save image results
    save_image(grid, f"{opt.rst_dir}/{opt.no}/{idx}.png")
