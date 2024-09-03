import numpy as np
import torch
from torch.utils.data import Dataset


class ConditionalGANDataset(Dataset):
    def __init__(self, data_path_G, data_path_R):
        data_G = np.load(data_path_G)
        data_R = np.load(data_path_R)
        
        self.images_G = torch.from_numpy(data_G['arr_0']).float()
        self.images_R = torch.from_numpy(data_R['arr_0']).float()
        
        # Normalize images to [-1, 1]
        self.images_G = (self.images_G / 127.5) - 1
        self.images_R = (self.images_R / 127.5) - 1
        
        # Ensure both datasets have the same number of images
        min_length = min(len(self.images_G), len(self.images_R))
        self.images_G = self.images_G[:min_length]
        self.images_R = self.images_R[:min_length]

        print(f"Dataset loaded:")
        print(f"Number of image pairs: {min_length}")
        print(f"Image G dimensions: {self.images_G.shape}")
        print(f"Image R dimensions: {self.images_R.shape}")

    def __len__(self):
        return len(self.images_G)

    def __getitem__(self, idx):
        return self.images_G[idx], self.images_R[idx]