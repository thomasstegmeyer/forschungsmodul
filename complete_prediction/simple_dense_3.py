
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import bottleneck as bn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataset_train import CrackDatasetTrain

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(4096, 4000) 
        self.fc2 = nn.Linear(4000, 4000)
        self.fc3 = nn.Linear(4000,4000)
        self.fc4 = nn.Linear(4000,4000)
        self.fc5 = nn.Linear(4000,4000)
        self.fc6 = nn.Linear(4000,4000)
        self.fc7 = nn.Linear(4000, 9)

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
        output = F.relu(self.fc7(r7))

        return output


net = Net()
#print(net)



data = CrackDatasetTrain()
#print(data[0])
#print(data[0]['damage'])

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001)
#Learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


#new training loop with validation
train_data, validation_data = torch.utils.data.random_split(data, [50048,9952])

trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
validationloader = DataLoader(validation_data, batch_size=128, shuffle=True) # Welche Parameter machen hier Sinn?

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

    plt.plot(epochs_done,traininglosses)
    plt.plot(epochs_done,validationlosses)
    plt.legend(["training","validation"])
    plt.xlabel("Epochs")
    plt.ylabel("MSE loss")
    plt.savefig("simple_dense_3_progress.png")
    plt.close()

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        path = "simple_dense_3_epoch_" + str(e+1) + "_min_validation_loss_lr0_0001_gamma_098.pkl"
        torch.save(net, path)
    scheduler.step()