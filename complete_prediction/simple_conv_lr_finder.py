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



from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar


import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc1 = nn.Linear(2704, 120) 
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


net = Net()

device = "cuda"
data = CrackDatasetTrain()
train_data, validation_data = torch.utils.data.random_split(data, [50000,10000])
trainloader = DataLoader(train_data, batch_size=256, shuffle=True)
validationloader = DataLoader(validation_data, batch_size=256, shuffle=True)

net.to(device)

optimizer = optim.Adam(net.parameters(), 1e-2)
criterion = nn.MSELoss()

trainer = create_supervised_trainer(net,optimizer,criterion,device = device)
ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})


lr_finder = FastaiLRFinder()
to_save={'model': net, 'optimizer': optimizer}
with lr_finder.attach(trainer, to_save, diverge_th=1.5) as trainer_with_lr_finder:
    trainer_with_lr_finder.run(trainloader)
    
trainer.run(trainloader, max_epochs=10)

evaluator = create_supervised_evaluator(net, metrics={"loss": Loss(nn.MSELoss())}, device=device)
evaluator.run(validationloader)

print(evaluator.state.metrics)


fig,ax = plt.subplots()

lr_finder.plot(ax = ax)
fig.savefig("learning_rate_finder.png")

print(lr_finder.lr_suggestion())


#p1 = Process(target = train_networks, args = (Net(),"simple_conv_lkrelu_lr_finder_lr_1e-3/simple_conv_lkrelu_lr_finder",1e-3,0.95))
#p2 = Process(target = train_networks, args = (Net(),"simple_conv_lkrelu_lr_finder_lr_1e-4/simple_conv_lkrelu_lr_finder",1e-4,0.95))
#p3 = Process(target = train_networks, args = (Net(),"simple_conv_lkrelu_lr_finder_lr_1e-5/simple_conv_lkrelu_lr_finder",1e-5,0.95))
#p4 = Process(target = train_networks, args = (Net(),"simple_conv_lkrelu_lr_finder_lr_1e-6/simple_conv_lkrelu_lr_finder",1e-6,0.95))
#
#
#p1.start()
#p2.start()
#p3.start()
#p4.start()
#
#
#p1.join()
#p2.join()
#p3.join()
#p4.join()
#



