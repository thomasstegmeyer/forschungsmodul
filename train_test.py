from dataset_train import CrackDatasetTrain
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

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
        self.fc1 = nn.Linear(59536, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        conv = self.conv1(input)
        #print("shape conv", conv.shape)
        c1 = F.relu(conv)
        #print("shape c1", c1.shape)
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        #print("shape s2", s2.shape)
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        #print("shape c3", c3.shape)
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        #print("shape s4", s4.shape)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        #print("shape s4", s4.shape)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        #print("shape f5", f5.shape)
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        #print("shape f6", f6.shape)
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()
#print(net)

data = CrackDatasetTrain()
#print(data[0])
#print(data[0]['damage'])

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(10000):
    out = net(torch.tensor([[data[i]["damage"]]]))
    target = torch.tensor([data[i]["measures"]])
    criterion = nn.MSELoss()

    #print(target)
    loss = criterion(out, target)

    #print(out)
    #print(loss)

    net.zero_grad()
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    plt.scatter(i,loss.detach())
    

print("done")
plt.show()