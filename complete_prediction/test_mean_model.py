from dataset_test import CrackDatasetTest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import bottleneck as bn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

means = np.loadtxt("means.txt")

def net(inputs):
    return means



data = CrackDatasetTest()
testloader = DataLoader(data, batch_size=1, shuffle=True)

relative_errors = []

for i,data in tqdm(enumerate(testloader)):

    inputs,labels = data
    inputs = inputs
    labels = labels

    outputs = net(inputs)

    relative_error = torch.abs(labels-outputs)/torch.abs(labels)

    relative_errors.append(relative_error)


stacked = torch.stack(relative_errors)

mean_values = np.array(torch.mean(stacked, dim = 0).detach())

np.savetxt("simple_conv_test_data_error.txt",mean_values)
print(mean_values)