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
        # prev_pot = self.sn1.v
        x = self.sn1(x)
        # print(prev_pot - self.sn1.v)
        # assert (prev_pot - self.sn1.v < self.sn1.v_threshold).all(), (
        #     prev_pot - self.sn1.v < self.sn1.v_threshold
        # )
        x = self.linear2(x)
        return x


class ContinuousMNISTNet(th.nn.Module):
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

    def forward(self, x: list[list[float]]):
        """Forward pass of the network

        Args:
            x (th.Tensor): Input spike times, shape ``(batch_size, neuron, num_spikes)``

        Returns:
            th.Tensor: Output tensor, shape ``(batch_size, 10)``
        """
        x_ = th.zeros(len(x), 28 * 28)
        for i in range(len(x)):
            for j in range(28 * 28):
                x_[i, j] = len(x[i][j])
        x = x_
        x = self.linear1(x)
        x = th.floor(x)
        x = self.linear2(x)
        x = th.floor(x)
        return x
