import os, shutil
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from model import *
from dataset import *
from utils import *

opt = parse()
same_seeds(opt.seed)

if opt.no is not None:
    no = opt.no
else:
    print("n is not a number")
    exit()

try:
    cfg_dir = os.makedirs(os.path.join(opt.cfg_dir), 0o777)
except:
    pass
cpt_dir = os.path.join(opt.cpt_dir, str(no))
log_dir = os.path.join(opt.log_dir, str(no))
if (not opt.yes) and os.path.exists(cpt_dir):
    res = input(f"no: {no} exists, are you sure continue training? It will override all files.[y:N]")
    res = res.lower()
    if res not in ["y", "yes"]:
        print("Stop training")
        exit()
    print("Override all files.")
if os.path.exists(cpt_dir):    
    shutil.rmtree(cpt_dir)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(cpt_dir, 0o777)
os.makedirs(log_dir, 0o777)
save_config(opt)

train_dataset = CTDataset(config=opt, isTrain=True)
valid_dataset = CTDataset(config=opt, isTrain=False)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=os.cpu_count(), drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

model = Model(opt)

total_epoch = opt.total_epoch
patience = opt.patience
best_msssim = 0
best_ssim = 0
best_psnr = 0
earlystop_counter = 0
msssim_curve = {"train": [], "valid": []}
ssim_curve   = {"train": [], "valid": []}
psnr_curve   = {"train": [], "valid": []}
st_time = datetime.now()

for epoch in range(total_epoch):
    print(f"Epoch: {epoch}!!")
    # Train model
    train_loss = model.train(train_dataloader, epoch)
    valid_loss = model.test(valid_dataloader)
    model.scheduler_step(epoch)
    # Record loss
    msssim_curve["train"].append(train_loss[0]); ssim_curve["train"].append(train_loss[1]); psnr_curve["train"].append(train_loss[2])
    msssim_curve["valid"].append(valid_loss[0]); ssim_curve["valid"].append(valid_loss[1]); psnr_curve["valid"].append(valid_loss[2])
    # Update best loss
    if (best_msssim < valid_loss[0] or best_ssim < valid_loss[1]) or best_psnr < valid_loss[2]:
        best_msssim, best_ssim, best_psnr = valid_loss
        model.save(cpt_dir)
        earlystop_counter = 0
        print(f"\33[91m>>>> Best model saved epoch: {epoch}!!<<\33[0m")
    # if best_loss doesn't improve for patience times, terminate training
    else:
        model.save(cpt_dir, "freq")
        print(f">> Freq model saved epoch: {epoch}!!")
        earlystop_counter += 1
        if not opt.no_early_stop and earlystop_counter >= patience:
            print("Early stop!!!")
            break
# Record
with open(f"{log_dir}/ssim_curve.json", "w") as fp:
    json.dump(ssim_curve, fp)
with open(f"{log_dir}/psnr_curve.json", "w") as fp:
    json.dump(psnr_curve, fp)
print(f"Finish training no: {no}, cost time: {datetime.now()-st_time}")
