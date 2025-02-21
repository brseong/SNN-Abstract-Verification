import torch as th
import torch.nn as nn
from .spikingjelly.spikingjelly.activation_based import neuron

class MNISTNet(th.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten1 = nn.Flatten()
        self.Linear1 = nn.Linear(28 * 28, 512, bias = False)
        self.sn1 = neuron.IFNode()
        self.Linear2 = nn.Linear(512, 10, bias = False)

    def forward(self, x):
        x = self.flatten1(x)
        x = self.Linear1(x)
        x = self.sn1(x)
        x = self.Linear2(x)
        return x

