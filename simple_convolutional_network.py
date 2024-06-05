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
optimizer = optim.SGD(net.parameters(), lr=0.0005)

#loss_storage = []
#for x in range(5):
#    for i in range(10000):
#        input,target = data[i]
#        #print(input.shape)
#        #print(target.shape)
#        out = net(torch.tensor([input]))
#        #print(out.shape)
#        target = torch.tensor(target)
#        criterion = nn.MSELoss()
#        #break
#
#        #print(target)
#        loss = criterion(out, target)
#
#        #print(out)
#        #print(loss)
#
#        net.zero_grad()
#        loss.backward()
#
#        optimizer.zero_grad()
#        optimizer.step()
#        #plt.scatter(i,loss.detach())
#        loss_storage.append(loss.detach())
#    
#
#print("done")
#moving_avg = bn.move_mean(loss_storage,100)
#plt.plot(range(len(moving_avg)),moving_avg)
#plt.savefig("train1000.png")

#dataloader = DataLoader(data, batch_size=32, shuffle=True)


#criterion = nn.MSELoss()


#first simple training loop
#epochs = 200

#length = len(dataloader)
#for e in tqdm(range(21, epochs)):
#    avg_loss = 0.
#
#    for i,data in tqdm(enumerate(dataloader)):
#        inputs, labels = data
#        optimizer.zero_grad()
#
#        outputs = net(inputs)
#        
#
#        loss = criterion(outputs,labels)
#
#        loss.backward()
#
#        optimizer.step()
#
#        current_loss = loss.item()
#        avg_loss += current_loss/length
#    
#    print(outputs[0])
#    print(current_loss)
#    print(f'Epoch {e+1} \t\t Training Loss: {avg_loss}')
#    if e == 10:
#        torch.save(net,"net10.pkl")
#    if e == 20:
#        torch.save(net,"net20.pkl")
#    if e == 30:
#        torch.save(net,"net30.pkl")
#    if e == 40:
#        torch.save(net,"net40.pkl")
#    if e == 50:
#        torch.save(net,"net50.pkl")
#    if e == 60:
#        torch.save(net,"net60.pkl")
#    if e == 70:
#        torch.save(net,"net70.pkl")
#    if e == 80:
#        torch.save(net,"net80.pkl")
#    if e == 90:
#        torch.save(net,"net90.pkl")
#    if e == 100:
#        torch.save(net,"net100.pkl")
#    if e == 110:
#        torch.save(net,"net110.pkl")
#    if e == 120:
#        torch.save(net,"net120.pkl")
#    if e == 130:
#        torch.save(net,"net130.pkl")
#    if e == 140:
#        torch.save(net,"net140.pkl")
#    if e == 150:
#        torch.save(net,"net150.pkl")
#    if e == 160:
#        torch.save(net,"net160.pkl")
#    if e == 170:
#        torch.save(net,"net170.pkl")
#    if e == 180:
#        torch.save(net,"net180.pkl")
#    if e == 190:
#        torch.save(net,"net190.pkl")
#    if e == 200:
#        torch.save(net,"net200.pkl")
#
#torch.save(net,"final.pkl")


#new training loop with validation
train_data, validation_data = torch.utils.data.random_split(data, [50016,9984])

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
validationloader = DataLoader(validation_data, batch_size=32, shuffle=True) # Welche Parameter machen hier Sinn?

epochs = 50

min_valid_loss = 175
criterion = nn.MSELoss()
length = len(trainloader)
length_val = len(validationloader)

net = torch.load("epoch8min_validation_loss.pkl")

for e in range(10,epochs):
    train_loss = 0.0
    net.train()

    for i,data in tqdm(enumerate(trainloader)):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        train_loss += current_loss/length

    valid_loss = 0.0
    net.eval()
    for i,data in tqdm(enumerate(validationloader)):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        current_loss = loss.item()
        valid_loss += current_loss/length_val

    print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        path = "epoch" + str(e) + "min_validation_loss.pkl"
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