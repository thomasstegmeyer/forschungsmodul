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
        self.conv1 = nn.Conv2d(1, 6, 10)
        self.conv2 = nn.Conv2d(6, 16, 10)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc1 = nn.Linear(1296, 1200) 
        self.fc2 = nn.Linear(1200,1000)
        self.fc3 = nn.Linear(1000,800)
        self.fc4 = nn.Linear(800,120)
        self.fc5 = nn.Linear(120, 84)
        self.fc6 = nn.Linear(84, 9)

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
        #print("shape c3", c3.shape)
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        #print("shape s4", s4.shape)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4,1)
        #print("shape s4 flatten", s4.shape)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.leaky_relu(self.fc1(s4))
        #print("shape f5", f5.shape)
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.leaky_relu(self.fc2(f5))
        #print("shape f6", f6.shape)
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        f7 = F.leaky_relu(self.fc3(f6))
        f8 = F.leaky_relu(self.fc4(f7))
        f9 = F.leaky_relu(self.fc5(f8))
        output = self.fc6(f9)
        #print("shape output", output.shape)
        return output



p11 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e1_lr_1e-1/lr_1e-1_",1e-1,0.99))
p15 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e1_lr_5e-1/lr_5e-1_",5e-1,0.99))   

p41 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e4_lr_1e-4/lr_1e-4_",1e-4,0.99))    
p45 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e4_lr_5e-4/lr_5e-4_",5e-4,0.99))    

p21 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e2_lr_1e-2/lr_1e-2_",1e-2,0.99))  
p25 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e2_lr_5e-2/lr_5e-2_",5e-2,0.99))   

p31 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_1e-3/lr_1e-3_",1e-3,0.99))
p32 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_2e-3/lr_2e-3_",2e-3,0.99))   
p33 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_3e-3/lr_3e-3_",3e-3,0.99))   
p34 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_4e-3/lr_4e-3_",4e-3,0.99))   
p35 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_5e-3/lr_5e-3_",5e-3,0.99))   
p36 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_6e-3/lr_6e-3_",6e-3,0.99))   
p37 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_7e-3/lr_7e-3_",7e-3,0.99))   
p38 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_8e-3/lr_8e-3_",8e-3,0.99))   
p39 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/run_oneCycle_params_max_lr_2e-3_epochs_200/e3_lr_9e-3/lr_9e-3_",9e-3,0.99))   




p11.start()
p15.start()

p21.start()
p25.start()

p31.start()
p32.start()
p33.start()
p34.start()
p35.start()
p36.start()
p37.start()
p38.start()
p39.start()


p41.start()
p45.start()


p11.join()
p15.join()
p21.join()
p25.join()
p31.join()
p32.join()
p33.join()
p34.join()
p35.join()
p36.join()
p37.join()
p38.join()
p39.join()
p41.join()
p45.join()