import os, shutil
import torch
import numpy as np
import random
import argparse
import json

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
