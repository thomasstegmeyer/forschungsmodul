import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

from .train_networks_volume import train_networks

from multiprocessing import Process

import time
time.sleep(14400)

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept single-channel input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Copy the weights from the pretrained model for RGB channels to the single channel
pretrained_weights = model.conv1.weight.data
model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

# Modify the final fully connected layer to output 9 regression parameters
num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 9)
model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )

processes = []
#processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_1e-3/resNet18_lr_1e-3_gamma_0.95_",0.001,0.95))
#processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_5e-3/resNet18_lr_5e-3_gamma_0.95_",0.005,0.95))
processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_1e-4/resNet18_lr_5e-4_gamma_0.95_",0.0005,0.95))

for p in processes:
    p.start()

for p in processes:
    p.join()



# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept single-channel input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Copy the weights from the pretrained model for RGB channels to the single channel
pretrained_weights = model.conv1.weight.data
model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

# Modify the final fully connected layer to output 9 regression parameters
num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 9)
model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )

processes = []
#processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_1e-3/resNet18_lr_1e-3_gamma_0.95_",0.001,0.95))
processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_5e-3/resNet18_lr_5e-3_gamma_0.95_",0.005,0.95))
#processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_1e-4/resNet18_lr_5e-4_gamma_0.95_",0.0005,0.95))

for p in processes:
    p.start()

for p in processes:
    p.join()



# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept single-channel input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Copy the weights from the pretrained model for RGB channels to the single channel
pretrained_weights = model.conv1.weight.data
model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

# Modify the final fully connected layer to output 9 regression parameters
num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 9)
model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )

processes = []
processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_1e-3/resNet18_lr_1e-3_gamma_0.95_",0.001,0.95))
#processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_5e-3/resNet18_lr_5e-3_gamma_0.95_",0.005,0.95))
#processes.append(train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_1e-4/resNet18_lr_5e-4_gamma_0.95_",0.0005,0.95))

for p in processes:
    p.start()

for p in processes:
    p.join()