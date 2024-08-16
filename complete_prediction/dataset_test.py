import os
import torch
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils


class CrackDatasetTest(Dataset):
    def __init__(self,transform = None):
        self.damages = os.listdir("../mat-dist-test/mat-dist-test")
        self.measures = os.listdir("../dmg-test/measures")
        self.transform = transform


    def __len__(self):
        return len(self.damages)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        damage = torch.tensor(np.array([np.float32(np.loadtxt("../mat-dist-test/mat-dist-test/"+self.damages[idx]))]))
        measures = torch.tensor(np.float32(np.loadtxt("../dmg-test/measures/"+self.measures[idx])))


        #if self.transform:
        #    sample = self.transform(sample)

        #return sample
        return damage, measures
        
#data = CrackDatasetTest()
#
#print(data[3])
#damage,label = data[3]
#print(damage)
#print(label)
#
#
#dataloader = DataLoader(data, batch_size=32, shuffle=True)
#
#for images, labels in dataloader:
#    print(images)
#    print(labels)
#    break
#
#
#print(images.shape)



