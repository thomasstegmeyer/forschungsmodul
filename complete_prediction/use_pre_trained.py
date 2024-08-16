import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

from train_networks import train_networks

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
        nn.Linear(num_ftrs, 9)
    )


train_networks(model,"test_pretrained/exponential_scheduler_huber_dropout_3e-3/resNet18_lr_3e-3_gamma_0.98_",0.003,0.98)


