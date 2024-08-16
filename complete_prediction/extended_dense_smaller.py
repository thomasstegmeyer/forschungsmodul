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

        self.fc1 = nn.Linear(4096, 8000) 
        self.fc2 = nn.Linear(8000, 4000)
        self.fc3 = nn.Linear(4000,2000)
        self.fc4 = nn.Linear(2000,1000)
        self.fc5 = nn.Linear(1000,500)
        self.fc6 = nn.Linear(500,400)
        self.fc7 = nn.Linear(400,300)
        self.fc8 = nn.Linear(300,200)
        self.fc9 = nn.Linear(200,100)
        self.fc10 = nn.Linear(100, 9)

        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        #print("input shape", input.shape)
        r1 = torch.flatten(input,1)
        r1 = self.dropout(r1)

        r2 = F.relu(self.fc1(r1))
        r3 = F.relu(self.fc2(r2))
        r4 = F.relu(self.fc3(r3))
        r5 = F.relu(self.fc4(r4))
        r6 = F.relu(self.fc5(r5))
        r7 = F.relu(self.fc6(r6))
        r8 = F.relu(self.fc7(r7))
        r9 = F.relu(self.fc8(r8))
        r10 = F.relu(self.fc9(r9))
        output = F.relu(self.fc10(r10))

        return output


p01 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_1e-1/extended_dense_smaller_lr_1e-1_",1e-1,0.95))
p02 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_2e-1/extended_dense_smaller_lr_2e-1_",2e-1,0.95))    
p03 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_3e-1/extended_dense_smaller_lr_3e-1_",3e-1,0.95))    
p04 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_4e-1/extended_dense_smaller_lr_4e-1_",4e-1,0.95))    
p05 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_5e-1/extended_dense_smaller_lr_5e-1_",5e-1,0.95))   
p06 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_6e-1/extended_dense_smaller_lr_6e-1_",6e-1,0.95))   
p07 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_7e-1/extended_dense_smaller_lr_7e-1_",7e-1,0.95))   
p08 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_8e-1/extended_dense_smaller_lr_8e-1_",8e-1,0.95))   
p09 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e1extended_dense_smaller_lr_9e-1/extended_dense_smaller_lr_9e-1_",9e-1,0.95))   
p11 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_1e-2/extended_dense_smaller_lr_1e-2_",1e-2,0.95))
p12 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_2e-2/extended_dense_smaller_lr_2e-2_",2e-2,0.95))    
p13 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_3e-2/extended_dense_smaller_lr_3e-2_",3e-2,0.95))    
p14 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_4e-2/extended_dense_smaller_lr_4e-2_",4e-2,0.95))    
p15 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_5e-2/extended_dense_smaller_lr_5e-2_",5e-2,0.95))   
p16 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_6e-2/extended_dense_smaller_lr_6e-2_",6e-2,0.95))   
p17 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_7e-2/extended_dense_smaller_lr_7e-2_",7e-2,0.95))   
p18 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_8e-2/extended_dense_smaller_lr_8e-2_",8e-2,0.95))   
p19 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e2extended_dense_smaller_lr_9e-2/extended_dense_smaller_lr_9e-2_",9e-2,0.95))   
p21 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_1e-3/extended_dense_smaller_lr_1e-3_",1e-3,0.95))
p22 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_2e-3/extended_dense_smaller_lr_2e-3_",2e-3,0.95))    
p23 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_3e-3/extended_dense_smaller_lr_3e-3_",3e-3,0.95))    
p24 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_4e-3/extended_dense_smaller_lr_4e-3_",4e-3,0.95))    
p25 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_5e-3/extended_dense_smaller_lr_5e-3_",5e-3,0.95))   
p26 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_6e-3/extended_dense_smaller_lr_6e-3_",6e-3,0.95))   
p27 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_7e-3/extended_dense_smaller_lr_7e-3_",7e-3,0.95))   
p28 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_8e-3/extended_dense_smaller_lr_8e-3_",8e-3,0.95))   
p29 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e3extended_dense_smaller_lr_9e-3/extended_dense_smaller_lr_9e-3_",9e-3,0.95))    
p31 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_1e-4/extended_dense_smaller_lr_1e-4_",1e-4,0.95))
p32 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_2e-4/extended_dense_smaller_lr_2e-4_",2e-4,0.95))    
p33 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_3e-4/extended_dense_smaller_lr_3e-4_",3e-4,0.95))    
p34 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_4e-4/extended_dense_smaller_lr_4e-4_",4e-4,0.95))    
p35 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_5e-4/extended_dense_smaller_lr_5e-4_",5e-4,0.95))   
p36 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_6e-4/extended_dense_smaller_lr_6e-4_",6e-4,0.95))   
p37 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_7e-4/extended_dense_smaller_lr_7e-4_",7e-4,0.95))   
p38 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_8e-4/extended_dense_smaller_lr_8e-4_",8e-4,0.95))   
p39 = Process(target = train_networks, args = (Net(),"test_all_unnormalized/extended_dense_smaller/e4extended_dense_smaller_lr_9e-4/extended_dense_smaller_lr_9e-4_",9e-4,0.95))    


p01.start()
p02.start()
p03.start()
p04.start()
p05.start()
p06.start()
p07.start()
p08.start()
p09.start()


p01.join()
p02.join()
p03.join()
p04.join()
p05.join()
p06.join()
p07.join()
p08.join()
p09.join()

p11.start()
p12.start()
p13.start()
p14.start()
p15.start()
p16.start()
p17.start()
p18.start()
p19.start()

p11.join()
p12.join()
p13.join()
p14.join()
p15.join()
p16.join()
p17.join()
p18.join()
p19.join()

p21.start()
p22.start()
p23.start()
p24.start()
p25.start()
p26.start()
p27.start()
p28.start()
p29.start()

p21.join()
p22.join()
p23.join()
p24.join()
p25.join()
p26.join()
p27.join()
p28.join()
p29.join()

p31.start()
p32.start()
p33.start()
p34.start()
p35.start()
p36.start()
p37.start()
p38.start()
p39.start()

p31.join()
p32.join()
p33.join()
p34.join()
p35.join()
p36.join()
p37.join()
p38.join()
p39.join()




x = torch.randn(1,1,64,64,requires_grad = True)
torch.onnx.export(Net(),x,"test_all_unnormalized/extended_dense_smaller/extended_dense_architecture.onnx",input_names = ["damages"],output_names = ["measures"])