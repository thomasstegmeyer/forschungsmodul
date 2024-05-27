import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CrackDatasetTrain(Dataset):
    def __init__(self,transform = None):
        self.damages = os.listdir("../dmg-train/dmg-train")
        self.measures = os.listdir("../dmg-train/measures")
        self.transform = transform


    def __len__(self):
        return len(self.damages)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #damage = torch.tensor(np.float32(np.loadtxt("../dmg-train/dmg-train/"+self.damages[idx])))
        #measures = torch.tensor(np.float32(np.loadtxt("../dmg-train/measures/"+self.measures[idx])))
        damage = np.float32(np.loadtxt("../dmg-train/dmg-train/"+self.damages[idx]))
        measures = np.float32(np.loadtxt("../dmg-train/measures/"+self.measures[idx]))
        sample = {'damage': damage,'measures': measures}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
data = CrackDatasetTrain()

print(data[3])


