import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

from .dataset_train_single import CrackDatasetTrain
from .dataset_test_single import CrackDatasetTest




def train_for_single_measure(measure, path, lr, gamma = 0.95, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0001, epochs = 100, dropout = True):

    #------------testing subroutine---------------
    def evaluate_performance(net,testdata):
        net.eval()
        testloader = DataLoader(testdata, batch_size=1, shuffle=True)
        relative_errors = []
        for i,data in enumerate(testloader):
        
            inputs,labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            relative_error = torch.abs(labels-outputs)/torch.abs(labels)
            relative_errors.append(relative_error)

        stacked = torch.stack(relative_errors)
        mean_values = np.array(torch.mean(stacked, dim = 0).cpu().detach())
        return mean_values

    #---------------Data------------------------------------------------------------------
    testdata = CrackDatasetTest(measure)
    testloader = DataLoader(testdata, batch_size=1, shuffle=True)

    data = CrackDatasetTrain(measure)

    #full dataset
    train_data, validation_data = torch.utils.data.random_split(data, [50000,10000])
    #small dataset
    #train_data, validation_data, rest = torch.utils.data.random_split(data, [100,10,59890])
    #medium dataset
    #train_data, validation_data, rest = torch.utils.data.random_split(data, [1000,100,58900])

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    validationloader = DataLoader(validation_data, batch_size=batchsize, shuffle=True)


    #---------------Save at---------------------------------------------------------------
    top_folder = path
    lr_string = "{:.2e}".format(lr).replace('.',':')
    gamma_string = "{:.2e}".format(gamma).replace('.',':')
    weight_decay_string = "{:.1e}".format(weight_decay).replace('.',':')
    subfolder = f"measure_{measure}_lr_{lr_string}_gamma_{gamma_string}_{optimizer}_weight_decay_{weight_decay_string}/"

    try:
        os.makedirs(top_folder+subfolder,exist_ok = True)
    except Exception as e:
        print(f"An error occurred: {e}")


    #----------------Model-----------------------------------------------------------------
    # Load pre-trained ResNet model
    net = models.resnet18(pretrained=True)

    # Modify the first convolutional layer to accept single-channel input
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Copy the weights from the pretrained model for RGB channels to the single channel
    pretrained_weights = net.conv1.weight.data
    net.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

    # Modify the final fully connected layer to output 9 regression parameters
    num_ftrs = net.fc.in_features

    if dropout:
        net.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 1)
            )
    else:
        net.fc = nn.Linear(num_ftrs, 1)

    #---------------Optimizer and Scheduler-----------------------------------------------
    if optimizer == "ADAM":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


    #---------------Training and Validation-----------------------------------------------
    min_valid_loss = np.inf
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    length = len(trainloader)
    length_val = len(validationloader)

    epochs_done = []
    traininglosses = []
    validationlosses = []

    net.cuda()

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
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            train_loss += current_loss/length


        #validation
        valid_loss = 0.0
        net.eval()
        for i,data in enumerate(validationloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            current_loss = loss.item()
            valid_loss += current_loss/length_val


        print(f'Epoch {e+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
        
        epochs_done.append(e+1)
        traininglosses.append(train_loss)
        validationlosses.append(valid_loss)

        np.savetxt(top_folder+subfolder + "progress.txt",[epochs_done,traininglosses,validationlosses])
        


        plt.plot(epochs_done,traininglosses)
        plt.plot(epochs_done,validationlosses)
        plt.legend(["training","validation"])
        plt.xlabel("Epochs")
        plt.ylabel("MSE loss")
        plt.savefig(top_folder+ subfolder + "progress.png")
        plt.close()

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), top_folder+subfolder+"best_network.pkl")

            performance = evaluate_performance(net,testdata)
            np.savetxt(top_folder+subfolder+"best_epoch_performance.txt",[e,float(performance)])

        
        #if min_valid_loss*10 < valid_loss:
        #    break

        scheduler.step()
