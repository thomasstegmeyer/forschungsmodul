import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CrackDatasetTest(Dataset):
    def __init__(self,transform = None):
        self.damages = os.listdir("../dmg-test/dmg-test")
        self.measures = os.listdir("../dmg-test/measures")
        self.transform = transform

        print(self.damages[0:3])
        print(self.measures[0:3])

    def __len__(self):
        return len(self.damages)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        damage = np.loadtxt("../dmg-test/dmg-test/"+self.damages[idx])
        measures = np.loadtxt("../dmg-test/measures/"+self.measures[idx])
        sample = {'damage': damage,'measures': measures}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
data = CrackDatasetTest()

print(data[3])


