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
        self.linear1 = nn.Linear(
            in_features, hidden_features, bias=False
        )  # (hidden_features, in_features)
        self.sn1 = neuron.IFNode(v_reset=1.0)
        self.linear2 = nn.Linear(
            hidden_features, out_features, bias=False
        )  # (out_features, hidden_features)
        self.sn2 = neuron.IFNode(v_reset=1.0)

    def forward_single(self, x: th.Tensor):
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.sn1(x)
        x = self.linear2(x)
        x = self.sn2(x)
        return x

    def forward(self, x: th.Tensor, is_sequence: bool = False):
        """Forward pass of the network

        Args:
            x (th.Tensor): Input tensor.
                shape ``(batch_size, num_steps, *features)`` if ``is_sequence``, ``(batch_size, *features)`` otherwise.
            is_sequence (bool): Whether the input is a sequence.

        Returns:
            th.Tensor: Output tensor, shape ``(batch_size, 10)``
        """
        if is_sequence:
            ret = self.forward_single(x[:, 0])
            for t in range(1, x.size(1)):
                ret.add_(self.forward_single(x[:, t]))
            return ret / x.size(1)
        else:
            return self.forward_single(x)


class AbsMNISTNet(MNISTNet):
    def forward(self, x: th.Tensor):
        """Forward pass of the network. Input is the number of spikes.

        Args:
            x (th.Tensor): Input tensor, shape ``(batch_size, 1, 28, 28)``

        Returns:
            th.Tensor: Output tensor, shape ``(batch_size, 10)``
        """
        x = self.flatten1(x)
        x = self.linear1(x)
        x = x.floor()
        x = self.linear2(x)
        # x = x.floor()
        return x
