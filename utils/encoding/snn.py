import dis
import torch as th
from torch.distributions import Exponential
from ..spikingjelly.spikingjelly.activation_based.encoding import PoissonEncoder


def encode_input(
    data: th.Tensor,
    num_steps: int,
    encoder: PoissonEncoder = PoissonEncoder(),
    as_counts: bool = False,
) -> th.Tensor:
    """Encode the input data into spikes with Poisson process.

    Args:
        data (th.Tensor): Input data, shape ``(batch_size, 1, 28, 28)``
        num_steps (int): Number of steps in the simulation.
        encoder (PoissonEncoder, optional): Encoder object. Defaults to ``PoissonEncoder()``.

    Returns:
        th.Tensor: Encoded spikes, shape (batch_size, num_steps, 1, 28, 28) if ``as_counts`` is False,
        and (batch_size, 1, 28, 28) if ``as_counts`` is True.
    """
    batch_size, *features = data.shape
    encoded = th.zeros(batch_size, num_steps, *features, device=data.device)
    for t in range(num_steps):
        encoded[:, t] = encoder(data)
    if as_counts:
        encoded = encoded.sum(dim=1)
    return encoded


if __name__ == "__main__":
    from torchvision import transforms
    from torchvision.datasets import MNIST
    import pdb

    MNIST_train = MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    x = encode_input(MNIST_train[0][0], 50, as_counts=True)
    pdb.set_trace()
