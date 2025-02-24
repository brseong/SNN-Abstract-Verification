import torch as th
import torch.nn as nn
from .spikingjelly.spikingjelly.activation_based import neuron


class MNISTNet(th.nn.Module):
    def __init__(
        self,
        in_features: int = 28 * 28,
        hidden_features: int = 512,
        out_features: int = 10,
    ):
        super(MNISTNet, self).__init__()
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
        self.sn1 = neuron.IFNode()
        self.linear2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: th.Tensor):
        """Forward pass of the network

        Args:
            x (th.Tensor): Input tensor, shape ``(batch_size, 1, 28, 28)``

        Returns:
            th.Tensor: Output tensor, shape ``(batch_size, 10)``
        """
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.sn1(x)
        x = self.linear2(x)
        return x
