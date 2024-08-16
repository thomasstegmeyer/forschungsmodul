from dataset_train import CrackDatasetTrain
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

        self.parallel1 = nn.Linear(4096,2000)
        self.parallel2 = nn.Linear(2000,100)

        self.merge = nn.Linear(184,84)

        
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):

        #convolutional block
        conv = self.conv1(input)
        c1 = F.leaky_relu(conv)
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.leaky_relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4,1)
        f5 = F.leaky_relu(self.fc1(s4))
        f6 = F.leaky_relu(self.fc2(f5))

        #linear block
        r1 = torch.flatten(input,1)
        r1 = self.dropout(r1)

        r2 = F.leaky_relu(self.parallel1(r1))
        r3 = F.leaky_relu(self.parallel2(r2))

        merged_data = F.leaky_relu(self.merge(torch.cat((f6,r3),1)))


        output = self.fc3(merged_data)
        return output




net = torch.load("test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_9e-4/extended_conv_parallel_oneCycle_lr_9e-4_gamma_95_17_min_validation_loss.pkl").cuda()

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

mean_values = np.array(torch.mean(stacked, dim = 0).cpu().detach())

np.savetxt("mean_relative_error_extended_conv_parallel_test_9e4.txt",mean_values)
print(mean_values)







#p01 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_1e-1/extended_conv_parallel_oneCycle_lr_1e-1_gamma_95_",1e-1,0.95))
#p02 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_2e-1/extended_conv_parallel_oneCycle_lr_2e-1_gamma_95_",2e-1,0.95))    
#p03 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_3e-1/extended_conv_parallel_oneCycle_lr_3e-1_gamma_95_",3e-1,0.95))    
#p04 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_4e-1/extended_conv_parallel_oneCycle_lr_4e-1_gamma_95_",4e-1,0.95))    
#p05 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_5e-1/extended_conv_parallel_oneCycle_lr_5e-1_gamma_95_",5e-1,0.95))   
#p06 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_6e-1/extended_conv_parallel_oneCycle_lr_6e-1_gamma_95_",6e-1,0.95))   
#p07 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_7e-1/extended_conv_parallel_oneCycle_lr_7e-1_gamma_95_",7e-1,0.95))   
#p08 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_8e-1/extended_conv_parallel_oneCycle_lr_8e-1_gamma_95_",8e-1,0.95))   
#p09 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e1extended_conv_parallel_oneCycle_9e-1/extended_conv_parallel_oneCycle_lr_9e-1_gamma_95_",9e-1,0.95))   

#p11 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_1e-2/extended_conv_parallel_oneCycle_lr_1e-2_gamma_95_",1e-2,0.95))
##p12 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_2e-2/extended_conv_parallel_oneCycle_lr_2e-2_gamma_95_",2e-2,0.95))    
#p13 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_3e-2/extended_conv_parallel_oneCycle_lr_3e-2_gamma_95_",3e-2,0.95))    
##p14 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_4e-2/extended_conv_parallel_oneCycle_lr_4e-2_gamma_95_",4e-2,0.95))    
#p15 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_5e-2/extended_conv_parallel_oneCycle_lr_5e-2_gamma_95_",5e-2,0.95))   
##p16 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_6e-2/extended_conv_parallel_oneCycle_lr_6e-2_gamma_95_",6e-2,0.95))   
#p17 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_7e-2/extended_conv_parallel_oneCycle_lr_7e-2_gamma_95_",7e-2,0.95))   
##p18 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_8e-2/extended_conv_parallel_oneCycle_lr_8e-2_gamma_95_",8e-2,0.95))   
#p19 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e2extended_conv_parallel_oneCycle_9e-2/extended_conv_parallel_oneCycle_lr_9e-2_gamma_95_",9e-2,0.95))
#
#p21 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_1e-3/extended_conv_parallel_oneCycle_lr_1e-3_gamma_95_",1e-3,0.95))
##p22 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_2e-3/extended_conv_parallel_oneCycle_lr_2e-3_gamma_95_",2e-3,0.95))    
#p23 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_3e-3/extended_conv_parallel_oneCycle_lr_3e-3_gamma_95_",3e-3,0.95))    
##p24 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_4e-3/extended_conv_parallel_oneCycle_lr_4e-3_gamma_95_",4e-3,0.95))    
#p25 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_5e-3/extended_conv_parallel_oneCycle_lr_5e-3_gamma_95_",5e-3,0.95))   
##p26 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_6e-3/extended_conv_parallel_oneCycle_lr_6e-3_gamma_95_",6e-3,0.95))   
#p27 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_7e-3/extended_conv_parallel_oneCycle_lr_7e-3_gamma_95_",7e-3,0.95))   
##p28 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_8e-3/extended_conv_parallel_oneCycle_lr_8e-3_gamma_95_",8e-3,0.95))   
#p29 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e3extended_conv_parallel_oneCycle_9e-3/extended_conv_parallel_oneCycle_lr_9e-3_gamma_95_",9e-3,0.95))
#   
#p31 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_1e-4/extended_conv_parallel_oneCycle_lr_1e-4_gamma_95_",1e-4,0.95))
##p32 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_2e-4/extended_conv_parallel_oneCycle_lr_2e-4_gamma_95_",2e-4,0.95))    
#p33 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_3e-4/extended_conv_parallel_oneCycle_lr_3e-4_gamma_95_",3e-4,0.95))    
##p34 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_4e-4/extended_conv_parallel_oneCycle_lr_4e-4_gamma_95_",4e-4,0.95))    
#p35 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_5e-4/extended_conv_parallel_oneCycle_lr_5e-4_gamma_95_",5e-4,0.95))   
##p36 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_6e-4/extended_conv_parallel_oneCycle_lr_6e-4_gamma_95_",6e-4,0.95))   
#p37 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_7e-4/extended_conv_parallel_oneCycle_lr_7e-4_gamma_95_",7e-4,0.95))   
##p38 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_8e-4/extended_conv_parallel_oneCycle_lr_8e-4_gamma_95_",8e-4,0.95))   
#p39 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_conv_parallel/e4extended_conv_parallel_oneCycle_9e-4/extended_conv_parallel_oneCycle_lr_9e-4_gamma_95_",9e-4,0.95))    
#
#
##p01.start()
##p02.start()
##p03.start()
##p04.start()
##p05.start()
##p06.start()
##p07.start()
##p08.start()
##p09.start()
#
#p11.start()
##p12.start()
#p13.start()
##p14.start()
#p15.start()
##p16.start()
#p17.start()
##p18.start()
#p19.start()
#
#p21.start()
##p22.start()
#p23.start()
##p24.start()
#p25.start()
##p26.start()
#p27.start()
##p28.start()
#p29.start()
#
#p31.start()
##p32.start()
#p33.start()
##p34.start()
#p35.start()
##p36.start()
#p37.start()
##p38.start()
#p39.start()
#
##p01.join()
##p02.join()
##p03.join()
##p04.join()
##p05.join()
##p06.join()
##p07.join()
##p08.join()
##p09.join()
#
#p11.join()
##p12.join()
#p13.join()
##p14.join()
#p15.join()
##p16.join()
#p17.join()
##p18.join()
#p19.join()
#
#p21.join()
##p22.join()
#p23.join()
##p24.join()
#p25.join()
##p26.join()
#p27.join()
##p28.join()
#p29.join()
#
#p31.join()
##p32.join()
#p33.join()
##p34.join()
#p35.join()
##p36.join()
#p37.join()
##p38.join()
#p39.join()
#
#
#x = torch.randn(1,1,64,64,requires_grad = True)
#torch.onnx.export(Net(),x,"test_all_unnormalized/extended_conv_parallel/extended_conv_parallel_architecture.onnx",input_names = ["damages"],output_names = ["measures"])