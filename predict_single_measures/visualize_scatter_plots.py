import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dataset_train_single import CrackDatasetTrain
from .dataset_test_single import CrackDatasetTest









def load_visualize_save(measure,lr):

    loadingpath = "predict_single_measures/results/measure_"+str(measure)+"_lr_"+lr+"_gamma_9:50e-01_ADAM_weight_decay_1:0e-04/best_network.pkl"
    savingpath = "predict_single_measures/labels_vs_predictions_measure_"+str(measure)+"_lr_"+lr+"_gamma_9:50e-01_ADAM_weight_decay_1:0e-04.png"
    #----------------Model-----------------------------------------------------------------
    # Load pre-trained ResNet model
    net = models.resnet18(pretrained=True)

    ## Modify the first convolutional layer to accept single-channel input
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #
    ## Copy the weights from the pretrained model for RGB channels to the single channel
    pretrained_weights = net.conv1.weight.data
    net.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
    #
    ## Modify the final fully connected layer to output 9 regression parameters
    num_ftrs = net.fc.in_features
    ##model.fc = nn.Linear(num_ftrs, 9)
    net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )


    net.load_state_dict(torch.load(loadingpath))
    #net = torch.load("predict_single_measures/results/measure_0_lr_1:00e-03_gamma_9:50e-01_ADAM_weight_decay_1:0e-04/best_network.pkl")


    data = CrackDatasetTrain(measure)
    dataloader = DataLoader(data, batch_size=100, shuffle=True)

    labels = []
    predictions = []

    net.eval()

    net = net.cuda()

    # Iterate once through the training data
    with torch.no_grad():  # Disable gradient calculation
        for data in tqdm(dataloader):
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets
            outputs = net(inputs)
            predicted = outputs.clone().detach().cpu()
            #print(predicted)
            
            labels.extend(targets.numpy())
            predictions.extend(predicted.numpy())


    # Convert lists to arrays
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(labels, predictions, color='red', alpha=0.1, marker='o')

    # Add titles and labels
    plt.title("True Labels vs Predictions")
    plt.xlabel('Label')
    plt.ylabel('Prediction')
    plt.grid(True)

    # Save the plot
    plt.savefig(savingpath)

    plt.close()

for measure in range(9):
    for lr in ["1:00e-03","5:00e-03","1:00e-04","5:00e-04","1:00e-05","5:00e-05"]:
        load_visualize_save(measure,lr)
