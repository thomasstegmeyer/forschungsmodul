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

import os

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

def train_networks(network,name,lr, gamma, epochs = 200, batchsize = 20, optimizer = "ADAM",weight_decay = 0):


    try:
        parent_directory = os.path.dirname(name)

        if parent_directory:
            os.makedirs(parent_directory,exist_ok = True)
    except Exception as e:
        # Catch all exceptions and print the error message
        print(f"An error occurred: {e}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(str(device) + " learning rate: " + str(lr))

    #initialize_weights(network)

    net = network.cuda()

    data = CrackDatasetTrain()

    if optimizer == "ADAM":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr = 2e-3,total_steps = 200)
    
    #scheduler2 = optim.lr_scheduler.LambdaLR(optimizer,labmda x:x, last_epoch = -1)
    #scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.5,10)
    #scheduler = ChainedScheduler([scheduler1,scheduler2])

    testdata = CrackDatasetTest()
    train_data, validation_data = torch.utils.data.random_split(data, [50000,10000])

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    validationloader = DataLoader(validation_data, batch_size=batchsize, shuffle=True)

    min_valid_loss = np.inf
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    length = len(trainloader)
    length_val = len(validationloader)

    epochs_done = []
    traininglosses = []
    validationlosses = []

    for e in range(epochs):

        train_loss = 0.0
        net.train()
        
        #training
        for i,data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            #print(outputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            train_loss += current_loss/length
            #tqdm.write(str(current_loss))

        #validation
        valid_loss = 0.0
        net.eval()
        for i,data in enumerate(validationloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            current_loss = loss.item()
            valid_loss += current_loss/length_val

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')


        epochs_done.append(e+1)
        traininglosses.append(train_loss)
        validationlosses.append(valid_loss)

        np.savetxt(name + "_progress.txt",[epochs_done,traininglosses,validationlosses])


        plt.plot(epochs_done,traininglosses)
        plt.plot(epochs_done,validationlosses)
        plt.legend(["training","validation"])
        plt.xlabel("Epochs")
        plt.ylabel("MSE loss")
        plt.savefig(name + "_progress.png")
        plt.close()

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            path = name + str(e+1) + "_min_validation_loss.pkl"
            torch.save(net, path)

        
        if min_valid_loss*10 < valid_loss:
            break

        scheduler.step()

        #if e%10 == 0:
        #    net.eval()
#
#
        #    
        #    testloader = DataLoader(testdata, batch_size=1, shuffle=True)
#
        #    relative_errors = []
#
        #    for i,data in tqdm(enumerate(testloader)):
        #    
        #        inputs,labels = data
        #        inputs = inputs.cuda()
        #        labels = labels.cuda()
#
        #        outputs = net(inputs)
#
        #        relative_error = torch.abs(labels-outputs)/torch.abs(labels)
#
        #        relative_errors.append(relative_error)
        #        
#
#
        #    stacked = torch.stack(relative_errors)
#
        #    mean_values = np.array(torch.mean(stacked, dim = 0).cpu().detach())
#
        #    print(mean_values)
