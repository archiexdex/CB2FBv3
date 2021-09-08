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
device = get_device()

if os.path.exists(f"{opt.rst_dir}/{opt.no}"):
    shutil.rmtree(f"{opt.rst_dir}/{opt.no}")
os.makedirs(f"{opt.rst_dir}/{opt.no}", 0o777)

dataset = CTDataset(config=opt, isTrain=False)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

model = Model(opt)
model.load(load_mode)
imgs = model.sample_image(dataset, sample_mode=sample_mode)
refs = [
    torch.from_numpy(np.load("./data/v1_test_data/npy/125.npy")) .unsqueeze(0).to(device),
    torch.from_numpy(np.load("./data/v1_test_data/npy/547.npy")) .unsqueeze(0).to(device),
    torch.from_numpy(np.load("./data/v1_test_data/npy/1093.npy")).unsqueeze(0).to(device),
]

for idx, (key, img) in enumerate(imgs.items()):
    real_cb, real_fb, fake_fb, mask, grid = img
    ref_fb = refs[idx]
    # Save image results
    save_image(real_cb, f"{opt.rst_dir}/{opt.no}/{key}_cb.png")
    save_image(real_fb, f"{opt.rst_dir}/{opt.no}/{key}_fb.png")
    save_image(fake_fb, f"{opt.rst_dir}/{opt.no}/{key}_fake.png")
    save_image(ref_fb,  f"{opt.rst_dir}/{opt.no}/{key}_ref.png")
    #save_image(grid, f"{opt.rst_dir}/{opt.no}/{idx}.png")
    # Evaluate images 
    origin_qa = get_qa(real_cb, real_fb, mask)
    output_qa = get_qa(fake_fb, real_fb, mask)
    refer_qa  = get_qa(ref_fb,  real_fb, mask)
    # Write analysis to csv
    save_csv(f"origin_{key}", origin_qa, f"{opt.rst_dir}/{opt.no}")
    save_csv(f"output_{key}", output_qa, f"{opt.rst_dir}/{opt.no}")
    save_csv(f"refer_{key}",  refer_qa,  f"{opt.rst_dir}/{opt.no}")
