from dataset_train import CrackDatasetTrain
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import bottleneck as bn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from train_networks import train_networks

from multiprocessing import Process

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16,6, 5)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc1 = nn.Linear(96, 90)
        self.fc2 = nn.Linear(90,90) 
        self.fc3 = nn.Linear(90, 84)
        self.fc4 = nn.Linear(84, 9)

        bias = np.loadtxt("means.txt")

        with torch.no_grad():
            self.fc4.bias.copy_(torch.tensor(bias))

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        #print("input shape", input.shape)
        conv = self.conv1(input)
        #print("shape conv", conv.shape)
        c1 = F.leaky_relu(conv)

        #print("shape c1", c1.shape)
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        #print("shape s2", s2.shape)
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.leaky_relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)

        c4 = F.leaky_relu(self.conv3(s4))
        s5 = F.max_pool2d(c4,2)
        #print("shape s4", s4.shape)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s5 = torch.flatten(s5,1)
        #print("shape s4 flatten", s4.shape)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.leaky_relu(self.fc1(s5))
        #print("shape f5", f5.shape)
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.leaky_relu(self.fc2(f5))
        #print("shape f6", f6.shape)
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        f7 = F.leaky_relu(self.fc3(f6))
        #print("shape output", output.shape)
        output = self.fc4(f7)
        return output



processes = []

processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e1simple_conv_biased_more_kernels_lr_1e-1/extended_dense_lr_1e-1",1e-1,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e1simple_conv_biased_more_kernels_lr_5e-1/extended_dense_lr_1e-1",5e-1,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e2simple_conv_biased_more_kernels_lr_1e-2/extended_dense_lr_1e-1",1e-2,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e2simple_conv_biased_more_kernels_lr_5e-2/extended_dense_lr_1e-1",5e-2,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e3simple_conv_biased_more_kernels_lr_1e-3/extended_dense_lr_1e-1",1e-3,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e3simple_conv_biased_more_kernels_lr_5e-3/extended_dense_lr_1e-1",5e-3,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e4simple_conv_biased_more_kernels_lr_1e-4/extended_dense_lr_1e-1",1e-4,0.98)))
processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/simple_conv_biased_more_kernels/e4simple_conv_biased_more_kernels_lr_5e-4/extended_dense_lr_1e-1",5e-4,0.98)))

for p in processes:
    p.start()



for p in processes:
    p.join()



