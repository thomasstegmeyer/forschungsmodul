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
        s4 = torch.flatten(s4,1)
        #print("shape s4 flatten", s4.shape)
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
        #print("shape output", output.shape)
        return output


net = Net()
#print(net)

data = CrackDatasetTrain()
#print(data[0])
#print(data[0]['damage'])

# create your optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)



#new training loop with validation
train_data, validation_data = torch.utils.data.random_split(data, [50016,9984])

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
validationloader = DataLoader(validation_data, batch_size=32, shuffle=True) # Welche Parameter machen hier Sinn?
epochs = 50

min_valid_loss = np.inf
criterion = nn.MSELoss()
length = len(trainloader)
length_val = len(validationloader)

epochs_done = []
traininglosses = []
validationlosses = []

for e in range(epochs):
    train_loss = 0.0
    net.train()

    for i,data in tqdm(enumerate(trainloader)):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        train_loss += current_loss/length
        #tqdm.write(str(current_loss))

    valid_loss = 0.0
    net.eval()
    for i,data in tqdm(enumerate(validationloader)):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        current_loss = loss.item()
        valid_loss += current_loss/length_val

    print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')

    epochs_done.append(e+1)
    traininglosses.append(train_loss)
    validationlosses.append(valid_loss)


    np.savetxt("simple_conv_adam_0001.txt",[epochs_done,traininglosses,validationlosses])

    plt.plot(epochs_done,traininglosses)
    plt.plot(epochs_done,validationlosses)
    plt.legend(["training","validation"])
    plt.xlabel("Epochs")
    plt.ylabel("MSE loss")
    plt.savefig("simple_conv_adam_0001.png")
    plt.close()

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        path = "simple_conv_adam_" + str(e+1) + "_min_validation_loss_lr0_0001.pkl"
        torch.save(net, path)



#small evaluation
#model = torch.load("net20.pkl")
#model.eval()
#
#data = CrackDatasetTrain()
#
#for i,data in tqdm(enumerate(dataloader)):
#
#    inputs, labels = data
#
#
#    output = model(inputs)
#
#    criterion = nn.MSELoss()
#
#    loss = criterion(output,labels)
#
#    print("loss: ", loss)
#    print("output:")
#    print(output[0])
#    print("label: ")
#    print(labels[0])
#    break