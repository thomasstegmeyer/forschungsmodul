import torch
import torch.nn as nn

from pretrainedmodels import se_resnext50_32x4d

from train_networks import train_networks

from torch.utils.data import DataLoader
from dataset_test import CrackDatasetTest
from tqdm import tqdm
import numpy as np

# Load the pretrained SEResNeXt model
#model = se_resnext50_32x4d(pretrained='imagenet')

import timm
model = timm.create_model('seresnext26d_32x4d', pretrained=True)

# The first convolutional layer is part of a Sequential block (model.conv1[0])
first_conv_layer = model.conv1[0]

# Create a new convolutional layer with 1 input channel and the same number of output channels
new_conv1 = nn.Conv2d(
    in_channels=1, 
    out_channels=first_conv_layer.out_channels, 
    kernel_size=first_conv_layer.kernel_size, 
    stride=first_conv_layer.stride, 
    padding=first_conv_layer.padding, 
    bias=first_conv_layer.bias is not None
)

# Copy the weights from the old convolutional layer to the new one, averaging across the RGB channels
with torch.no_grad():
    new_conv1.weight = nn.Parameter(first_conv_layer.weight.sum(dim=1, keepdim=True))

# Replace the original convolutional layer with the new one
model.conv1[0] = new_conv1


# Modify the last fully connected layer to output 9 continuous values
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)


#train_networks(model,"test_seresnext/exponential_scheduler_1e-2_0_9/seresnext_lr_1e-3_gamma_0.9_",0.01,0.9,epochs = 20)

model = torch.load("test_seresnext/exponential_scheduler_1e-2_0_9/seresnext_lr_1e-3_gamma_0.9_14_min_validation_loss.pkl").cpu()

#testing_routine
model.eval()

testdata = CrackDatasetTest()
testloader = DataLoader(testdata, batch_size=20, shuffle=True)

relative_errors = []

print(len(testloader))

for i,data in tqdm(enumerate(testloader)):

    inputs,labels = data
    outputs = model(inputs)

    relative_error = torch.abs(labels-outputs)/torch.abs(labels)
    #print(relative_error)

    relative_errors.extend(relative_error)
    


stacked = torch.stack(relative_errors)

mean_values = np.array(torch.mean(stacked, dim = 0).detach())

print(mean_values)
