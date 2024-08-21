import os
import torch
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
from torchvision.datasets import ImageFolder


class CrackDatasetTrain(Dataset):
    def __init__(self,transform = None):

        self.damages = os.listdir("../mat-dist-train/mat-dist-train")
        #self.measures = os.listdir("../dmg-train/measures_normalized")
        self.measures = os.listdir("../dmg-train/measures")
        self.transform = transform


    def __len__(self):
        return len(self.damages)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        damage = torch.tensor(np.array([np.float32(np.loadtxt("../mat-dist-train/mat-dist-train/"+self.damages[idx]))]))
        measures = torch.tensor(np.float32(np.loadtxt("../dmg-train/measures/"+self.measures[idx]))[0])
        #measures = torch.tensor(np.float32(np.loadtxt("../dmg-train/measures_normalized/"+self.measures[idx])))


        #damage = np.float32(np.loadtxt("../mat-dist-train/mat-dist-train/"+self.damages[idx]))
        #measures = np.float32(np.loadtxt("../dmg-train/measures/"+self.measures[idx]))
       # sample = {'damage': damage,'measures': measures}
        #sample = {damage: measures}

        #if self.transform:
        #    sample = self.transform(sample)

        #return sample
        return damage, measures
        
#data = CrackDatasetTrain()

#print(data[3])

#damage,label = data[3]
#print(damage)
#print()
#print(label)
#
#
#dataloader = DataLoader(data, batch_size=32, shuffle=True)
#
#for images, labels in dataloader:
#    print("hi")
#    print(images.shape)
#    print("there")
#    print(labels.shape)
#    break



