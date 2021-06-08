import os
import random
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torchvision import transforms as tf
from scipy.ndimage import rotate

class CTDataset(Dataset):
    """ dataset."""
    def __init__(self, config, isTrain):
        """
        Args:
        """
        root_dir = config.train_data_path if isTrain else config.test_data_path
        self.isTrain = isTrain
        self.crop = config.crop
        if self.crop:
            self.crop_size = config.crop_size
        
        self.input_data_path = os.path.join(root_dir, "cb")
        self.label_data_path = os.path.join(root_dir, "fb")
        self.mask_data_path  = os.path.join(root_dir, "mask")
            
        self.sz = len([f for f in os.listdir(os.path.join(root_dir, "cb")) if os.path.isfile(os.path.join(root_dir, "cb", f))])
        
    def transform(self, x, y, cm, qafb=None):
        if self.isTrain:
            if self.crop:
                # Random crop
                h, w = x.shape[:2]
                th, tw = (self.crop_size, self.crop_size)
                i = random.randint(0, (h-th))
                j = random.randint(0, (w-tw))
                x = x[i:i+th, j:j+tw]
                y = y[i:i+th, j:j+tw]
                cm = cm[i:i+th, j:j+tw]

            # Random horizontal flipping
            if random.random() > 0.5:
                x = x[-1::-1]
                y = y[-1::-1]
                cm = cm[-1::-1]
            
            # Random vertical flipping
            if random.random() > 0.5:
                x = x[:,-1::-1]
                y = y[:,-1::-1]
                cm = cm[:,-1::-1]
            
            # Random Rotation
            angle = random.uniform(-40, 40)
            x = rotate(x, angle, reshape=False)
            y = rotate(y, angle, reshape=False)
            cm = rotate(cm, angle, reshape=False)

        # Transform to tensor
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)))
        y = torch.from_numpy(np.transpose(y, (2, 0, 1)))
        cm = torch.from_numpy(np.transpose(cm, (2, 0, 1)))
        
        return x, y, cm

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        cb = np.load(os.path.join(self.input_data_path, "{}.npy".format(idx))).astype(np.float32)
        fb = np.load(os.path.join(self.label_data_path, "{}.npy".format(idx))).astype(np.float32)
        mask = np.load(os.path.join(self.mask_data_path, "{}.npy".format(idx)))
        
        cb = np.expand_dims(cb, axis=-1)
        fb = np.expand_dims(fb, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        
        return self.transform(cb, fb, mask)
