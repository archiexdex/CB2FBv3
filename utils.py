import os, shutil
import torch
import numpy as np
import random
import argparse
import json
import csv
from IQA_pytorch import SSIM, FSIM, DISTS, GMSD, MS_SSIM, LPIPSvgg, CW_SSIM, VIF

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse(args=None):
    parser = argparse.ArgumentParser()
    from arguments import add_arguments
    parser = add_arguments(parser)
    if args is not None:
        return parser.parse_args(args=args)
    else:
        return parser.parse_args()

def save_config(config):
    _config = vars(config)
    with open(os.path.join(config.cfg_dir, f"{config.no}.json"), "w") as fp:
        json.dump(_config, fp, ensure_ascii=False, indent=4)

# Get gray image
def get_pixels_hu(info):
    img = info.pixel_array
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # img[img < -2000] = 0
    # Convert to Hounsfield units (HU)
    for idx, item in enumerate(info.PerFrameFunctionalGroupsSequence):
        intercept = item.RealWorldValueMappingSequence[0].RealWorldValueIntercept
        slope = item.RealWorldValueMappingSequence[0].RealWorldValueSlope
        img[idx] = slope * img[idx].astype(np.float32) + intercept
    return np.array(img, dtype='float32')

def get_psnr(a, b):
    dif = a - b
    mse = dif ** 2
    psnr = 10 * torch.log10(1 / mse.mean())
    return psnr

def get_qa(cb, fb, mk):
    device = get_device()
    trans = lambda x: x.unsqueeze(0)
    get_ssim = SSIM().to(device)
    get_fsim = FSIM().to(device)
    get_gmsd = GMSD(1).to(device)
    get_vif = VIF(channels=1).to(device)
    get_msssim = MS_SSIM(channels=1).to(device)

    cb = trans(cb)
    fb = trans(fb)
    mk = trans(mk)
    
    cb[mk<250] = 0
    fb[mk<250] = 0
    
    ssim   = get_ssim  (cb, fb, False)[0].item()
    psnr   = get_psnr  (cb, fb).item()
    # TODO: 
    fsim   = 0 #get_fsim  (cb, fb, False)[0].item()
    gmsd   = get_gmsd  (cb, fb, False)[0].item()
    msssim = get_msssim(cb, fb, False)[0].item()
    vif    = get_vif   (cb, fb, False)[0].item()

    return ssim, psnr, fsim, gmsd, msssim, vif

def save_csv(name, qas, save_path):
    fieldnames = ["name", "ssim", "psnr", "fsim", "gmsd", "msssim", "vif"]
    if not os.path.exists(f"{save_path}/qa.csv"):
        with open(f"{save_path}/qa.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(f"{save_path}/qa.csv", "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
                fieldnames[0]: name,
                fieldnames[1]: f"{qas[0]:.4f}",
                fieldnames[2]: f"{qas[1]:.4f}",
                fieldnames[3]: f"{qas[2]:.4f}",
                fieldnames[4]: f"{qas[3]:.4f}",
                fieldnames[5]: f"{qas[4]:.4f}",
                fieldnames[6]: f"{qas[5]:.4f}",
            })
