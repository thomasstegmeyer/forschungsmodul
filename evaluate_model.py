from dataset_train import CrackDatasetTrain
from forschungsmodul_data.forschungsmodul.simple_convolutional_network import Net
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import bottleneck as bn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np


model = torch.load("final.pkl")
model.eval()

data = CrackDatasetTrain()

input, label = data[0]

output = model(input)

criterion = nn.MSELoss()

loss = criterion(output,label)

print("loss: ", loss)
print("output:")
print(output)
print("label: ")
print(label)