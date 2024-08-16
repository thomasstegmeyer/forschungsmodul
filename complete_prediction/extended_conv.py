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

from train_networks import train_networks

from multiprocessing import Process

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, dilation = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc1 = nn.Linear(2304, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

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
        output = self.fc3(f6)
        #print("shape output", output.shape)
        return output




net = torch.load("test_all_unnormalized/run_gamma_095_dilation_2_more_eps/e4simple_conv_lkrelu_lr_9e-4/simple_conv_lkrelu_lr_9e-4_gamma_95_144_min_validation_loss.pkl").cuda()

net.eval()


data = CrackDatasetTest()
testloader = DataLoader(data, batch_size=1, shuffle=True)

relative_errors = []

for i,data in tqdm(enumerate(testloader)):

    inputs,labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()

    outputs = net(inputs)

    relative_error = torch.abs(labels-outputs)/torch.abs(labels)

    relative_errors.append(relative_error)


stacked = torch.stack(relative_errors)

mean_values = np.transpose(np.array(torch.mean(stacked, dim = 0).cpu().detach()))

np.savetxt("mean_relative_error_test.txt",mean_values)
print(mean_values)






#
#p01 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_1e-1/simple_conv_lkrelu_lr_1e-1_gamma_95_",1e-1,0.95))
#p02 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_2e-1/simple_conv_lkrelu_lr_2e-1_gamma_95_",2e-1,0.95))    
#p03 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_3e-1/simple_conv_lkrelu_lr_3e-1_gamma_95_",3e-1,0.95))    
#p04 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_4e-1/simple_conv_lkrelu_lr_4e-1_gamma_95_",4e-1,0.95))    
#p05 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_5e-1/simple_conv_lkrelu_lr_5e-1_gamma_95_",5e-1,0.95))   
#p06 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_6e-1/simple_conv_lkrelu_lr_6e-1_gamma_95_",6e-1,0.95))   
#p07 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_7e-1/simple_conv_lkrelu_lr_7e-1_gamma_95_",7e-1,0.95))   
#p08 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_8e-1/simple_conv_lkrelu_lr_8e-1_gamma_95_",8e-1,0.95))   
#p09 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e1simple_conv_lkrelu_lr_9e-1/simple_conv_lkrelu_lr_9e-1_gamma_95_",9e-1,0.95))   
#
#p11 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_1e-2/simple_conv_lkrelu_lr_1e-2_gamma_95_",1e-2,0.95))
#p12 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_2e-2/simple_conv_lkrelu_lr_2e-2_gamma_95_",2e-2,0.95))    
#p13 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_3e-2/simple_conv_lkrelu_lr_3e-2_gamma_95_",3e-2,0.95))    
#p14 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_4e-2/simple_conv_lkrelu_lr_4e-2_gamma_95_",4e-2,0.95))    
#p15 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_5e-2/simple_conv_lkrelu_lr_5e-2_gamma_95_",5e-2,0.95))   
#p16 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_6e-2/simple_conv_lkrelu_lr_6e-2_gamma_95_",6e-2,0.95))   
#p17 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_7e-2/simple_conv_lkrelu_lr_7e-2_gamma_95_",7e-2,0.95))   
#p18 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_8e-2/simple_conv_lkrelu_lr_8e-2_gamma_95_",8e-2,0.95))   
#p19 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e2simple_conv_lkrelu_lr_9e-2/simple_conv_lkrelu_lr_9e-2_gamma_95_",9e-2,0.95))   
#
#
#p21 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_1e-3/simple_conv_lkrelu_lr_1e-3_gamma_95_",1e-3,0.95))
#p22 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_2e-3/simple_conv_lkrelu_lr_2e-3_gamma_95_",2e-3,0.95))    
#p23 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_3e-3/simple_conv_lkrelu_lr_3e-3_gamma_95_",3e-3,0.95))    
#p24 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_4e-3/simple_conv_lkrelu_lr_4e-3_gamma_95_",4e-3,0.95))    
#p25 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_5e-3/simple_conv_lkrelu_lr_5e-3_gamma_95_",5e-3,0.95))   
#p26 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_6e-3/simple_conv_lkrelu_lr_6e-3_gamma_95_",6e-3,0.95))   
#p27 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_7e-3/simple_conv_lkrelu_lr_7e-3_gamma_95_",7e-3,0.95))   
#p28 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_8e-3/simple_conv_lkrelu_lr_8e-3_gamma_95_",8e-3,0.95))   
#p29 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e3simple_conv_lkrelu_lr_9e-3/simple_conv_lkrelu_lr_9e-3_gamma_95_",9e-3,0.95))    
#
#p31 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_1e-4/simple_conv_lkrelu_lr_1e-4_gamma_95_",1e-4,0.95))
#p32 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_2e-4/simple_conv_lkrelu_lr_2e-4_gamma_95_",2e-4,0.95))    
#p33 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_3e-4/simple_conv_lkrelu_lr_3e-4_gamma_95_",3e-4,0.95))    
#p34 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_4e-4/simple_conv_lkrelu_lr_4e-4_gamma_95_",4e-4,0.95))    
#p35 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_5e-4/simple_conv_lkrelu_lr_5e-4_gamma_95_",5e-4,0.95))   
#p36 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_6e-4/simple_conv_lkrelu_lr_6e-4_gamma_95_",6e-4,0.95))   
#p37 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_7e-4/simple_conv_lkrelu_lr_7e-4_gamma_95_",7e-4,0.95))   
#p38 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_8e-4/simple_conv_lkrelu_lr_8e-4_gamma_95_",8e-4,0.95))   
#p39 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/e4simple_conv_lkrelu_lr_9e-4/simple_conv_lkrelu_lr_9e-4_gamma_95_",9e-4,0.95))    
#
#
#p01.start()
#p02.start()
#p03.start()
#p04.start()
#p05.start()
#p06.start()
#p07.start()
#p08.start()
#p09.start()
#
#p11.start()
#p12.start()
#p13.start()
#p14.start()
#p15.start()
#p16.start()
#p17.start()
#p18.start()
#p19.start()
#
#p21.start()
#p22.start()
#p23.start()
#p24.start()
#p25.start()
#p26.start()
#p27.start()
#p28.start()
#p29.start()
#
#p31.start()
#p32.start()
#p33.start()
#p34.start()
#p35.start()
#p36.start()
#p37.start()
#p38.start()
#p39.start()
#
#p01.join()
#p02.join()
#p03.join()
#p04.join()
#p05.join()
#p06.join()
#p07.join()
#p08.join()
#p09.join()
#
#p11.join()
#p12.join()
#p13.join()
#p14.join()
#p15.join()
#p16.join()
#p17.join()
#p18.join()
#p19.join()
#
#p21.join()
#p22.join()
#p23.join()
#p24.join()
#p25.join()
#p26.join()
#p27.join()
#p28.join()
#p29.join()
#
#p31.join()
#p32.join()
#p33.join()
#p34.join()
#p35.join()
#p36.join()
#p37.join()
#p38.join()
#p39.join()
#
#
#