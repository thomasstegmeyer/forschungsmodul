from dataset_train import CrackDatasetTrain
import torch.nn as nn
import torch

# With square kernels and equal stride
m = nn.Conv2d(1, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
#m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
#m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(10, 1, 5, 5)
print(input)
#print(m(input))

data = CrackDatasetTrain()
print(torch.tensor([[data[0]["damage"]]]))
out = m(torch.tensor([[data[0]["damage"]]]))