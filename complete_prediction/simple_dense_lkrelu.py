
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import bottleneck as bn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataset_test import CrackDatasetTest

def goto(linenum):
    global line
    line = linenum

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(4096, 4000) 
        self.fc2 = nn.Linear(4000, 4000)
        self.fc3 = nn.Linear(4000,4000)
        self.fc4 = nn.Linear(4000,4000)
        self.fc5 = nn.Linear(4000,400)
        self.fc6 = nn.Linear(4000,4000)
        self.fc7 = nn.Linear(4000,4000)
        self.fc8 = nn.Linear(4000,4000)
        self.fc9 = nn.Linear(4000,4000)
        self.fc10 = nn.Linear(4000, 9)

        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        #print("input shape", input.shape)
        r1 = torch.flatten(input,1)
        r1 = self.dropout(r1)

        r2 = F.leaky_relu(self.fc1(r1))
        r3 = F.leaky_relu(self.fc2(r2))
        r4 = F.leaky_relu(self.fc3(r3))
        r5 = F.leaky_relu(self.fc4(r4))
        r6 = F.leaky_relu(self.fc5(r5))
        r7 = F.leaky_relu(self.fc6(r6))
        r8 = F.leaky_relu(self.fc7(r7))
        r9 = F.leaky_relu(self.fc8(r8))
        r10 = F.leaky_relu(self.fc9(r9))
        output = F.leaky_relu(self.fc10(r10))

        return output


model = Net()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
#x = torch.randn(1,1,64,64,requires_grad = True)
#torch.onnx.export(Net(),x,"simple_dense_architecture.onnx",input_names = ["damages"],output_names = ["measures"])


#net = torch.load("simple_dense_lkrelu/simple_dense_lkrelu_epoch_22_min_validation_loss_lr0_000001_gamma_098.pkl").cuda()
#
#net.eval()
#
#
#data = CrackDatasetTest()
#testloader = DataLoader(data, batch_size=1, shuffle=True)
#
#relative_errors = []
#
#for i,data in tqdm(enumerate(testloader)):
#
#    inputs,labels = data
#    inputs = inputs.cuda()
#    labels = labels.cuda()
#
#    outputs = net(inputs)
#
#    relative_error = torch.abs(labels-outputs)/torch.abs(labels)
#
#    relative_errors.append(relative_error)
#
#
#stacked = torch.stack(relative_errors)
#
#mean_values = np.transpose(np.array(torch.mean(stacked, dim = 0).cpu().detach()))
#
#np.savetxt("mean_relative_error_simple_dense_lkrelu_test.txt",mean_values)
#print(mean_values)
#
#
#
#
#net = Net()
##print(net)
#
#
#
#data = CrackDatasetTrain()
##print(data[0])
##print(data[0]['damage'])
#
## create your optimizer
#optimizer = optim.SGD(net.parameters(), lr=0.00001)
##Learning rate scheduler
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
#
#
##new training loop with validation
#train_data, validation_data = torch.utils.data.random_split(data, [50048,9952])
#
#trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
#validationloader = DataLoader(validation_data, batch_size=32, shuffle=True) # Welche Parameter machen hier Sinn?
#
#epochs = 200
#
#min_valid_loss = np.inf
#criterion = nn.MSELoss()
#length = len(trainloader)
#length_val = len(validationloader)
#
#epochs_done = []
#traininglosses = []
#validationlosses = []
#
#range_epochs = range(epochs)
#e = 0
#
#for e in range_epochs:
#    train_loss = 0.0
#    net.train()
#
#    for i,data in tqdm(enumerate(trainloader)):
#        inputs, labels = data
#        optimizer.zero_grad()
#        outputs = net(inputs)
#        #print(outputs)
#        loss = criterion(outputs,labels)
#        current_loss = loss.item()
#        loss.backward()
#        optimizer.step()
#        train_loss += current_loss/length
#        #tqdm.write(str(current_loss))
#
#    valid_loss = 0.0
#    net.eval()
#    for i,data in tqdm(enumerate(validationloader)):
#        inputs, labels = data
#        outputs = net(inputs)
#        loss = criterion(outputs,labels)
#        current_loss = loss.item()
#        valid_loss += current_loss/length_val
#
#    print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
#    if np.isnan(train_loss):
#        print("starting again")
#        goto(60)
#
#    epochs_done.append(e+1)
#    traininglosses.append(train_loss)
#    validationlosses.append(valid_loss)
#
#    np.savetxt("lkrelu_progress.txt",[epochs_done,traininglosses,validationlosses])
#
#    plt.plot(epochs_done,traininglosses)
#    plt.plot(epochs_done,validationlosses)
#    plt.legend(["training","validation"])
#    plt.xlabel("Epochs")
#    plt.ylabel("MSE loss")
#    plt.savefig("simple_dense_lkrelu_progress.png")
#    plt.close()
#
#    if min_valid_loss > valid_loss:
#        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
#        min_valid_loss = valid_loss
#        # Saving State Dict
#        path = "simple_dense_lkrelu_epoch_" + str(e+1) + "_min_validation_loss_lr0_000001_gamma_098.pkl"
#        torch.save(net, path)
#    scheduler.step()