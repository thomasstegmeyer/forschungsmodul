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
from torchview import draw_graph

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

        bias = np.loadtxt("means.txt")

        self.dropout = nn.Dropout(0.1)

        with torch.no_grad():
            self.fc3.bias.copy_(torch.tensor(bias))

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        #print("input shape", input.shape)
        conv = self.conv1(input)
        #print("shape conv", conv.shape)
        c1 = F.silu(conv)
        #print("shape c1", c1.shape)
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        #print("shape s2", s2.shape)
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.silu(self.conv2(s2))
        #print("shape c3", c3.shape)
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        #print("shape s4", s4.shape)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4,1)
        s4 = self.dropout(s4)
        #print("shape s4 flatten", s4.shape)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.elu(self.fc1(s4))
        #print("shape f5", f5.shape)
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.elu(self.fc2(f5))
        #print("shape f6", f6.shape)
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        #print("shape output", output.shape)
        return output

##visualization
#model = Net()
#dummy_input = torch.zeros([1, 1, 64, 64])
#
## Generate a graph using torchviz
#output = model(dummy_input)
#dot = make_dot(output, params=dict(list(model.named_parameters())))
#
#def trim_graph(dot):
#    for node in dot.body:
#        # Remove unnecessary attributes (e.g., input, gradients)
#        node_fields = node.split()
#        if len(node_fields) > 1:
#            node_name = node_fields[0]
#            if "Tensor" in node or "weight" in node or "bias" in node:
#                dot.body.remove(node)
#            else:
#                node_idx = dot.body.index(node)
#                node_label = node_fields[-1].replace('label="', '').replace('"]', '')
#                dot.body[node_idx] = f'{node_name} [label="{node_label}"]\n'
#    return dot
#
#slim_dot = trim_graph(dot)
#
#slim_dot.render("model_architecture_slim", format="pdf")

#using torchview
#fig,ax = plt.subplots()
#
#model_graph = draw_graph(Net(), input_size=(1,1,64,64), expand_nested=True)
#ax.imshow(model_graph.visual_graph)
#plt.savefig("model_architecture_torchview.png", bbox_inches='tight')
#x = torch.randn(1,1,64,64,requires_grad = True)
#torch.onnx.export(Net(),x,"model_architecture.onnx",input_names = ["damages"],output_names = ["measures"])



processes = []

processes.append(Process(target = train_networks, args = (Net(),"test_all_unnormalized/other_activation/elu_sigmoid_ReduceOP/simple_conv_elu_sigmoid_lr_5e-3_gamma_0.95_",5e-3,0.95,200,256,"ADAM",0.001)))

for p in processes:
    p.start()
    p.join()