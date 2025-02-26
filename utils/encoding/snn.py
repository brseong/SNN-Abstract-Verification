import dis
import torch as th
from torch.distributions import Exponential
from ..spikingjelly.spikingjelly.activation_based.encoding import PoissonEncoder


def encode_input(
    data: th.Tensor, num_steps: int, encoder: PoissonEncoder = PoissonEncoder()
) -> th.Tensor:
    """Encode the input data into spikes with Poisson process.

    Args:
        data (th.Tensor): Input data, shape ``(batch_size, 1, 28, 28)``
        num_steps (int): Number of steps in the simulation.
        encoder (PoissonEncoder, optional): Encoder object. Defaults to ``PoissonEncoder()``.

    Returns:
        th.Tensor: Encoded spikes, shape (batch_size, num_steps, 1, 28, 28)
    """
    batch_size, *features = data.shape
    encoded = th.zeros(batch_size, num_steps, *features, device=data.device)
    for t in range(num_steps):
        encoded[:, t] = encoder(data)
    return encoded


def encode_input_continuous(
    data: th.Tensor,
    rate: float = 1e-2,
    dist_type: type[th.distributions.Distribution] | None = None,
    eps: float = 1e-6,
) -> list[list[list[float]]]:
    """Encode the input data into spikes with continuous time.

    Args:
        data (th.Tensor): Input data, shape ``(batch_size, 1, 28, 28)``
        rate (float, optional): Spike rate. Defaults to ``1e-2``, 10Hz during 1000ms.
        dist_type (type[th.distributions.Distribution] | None, optional): Distribution type. Defaults to ``torch.distributions.Exponential``.
        eps (float, optional): Min spike rate. Defaults to ``1e-6``.

    Returns:
        list[list[list[float]]]: Encoded spikes, shape ``(batch_size, 784, num_spikes)``.
        The number of spikes is not fixed for each neuron.
    """
    if dist_type is None:
        dist_type = Exponential
    data = data.flatten(start_dim=1).clamp(min=eps)  # shape (batch_size, 784)
    batch_size, features = data.shape

    encoded = list[list[list[float]]]()
    for batch_index in range(batch_size):
        encoded.append(list[list[float]]())
        for neuron in range(features):
            encoded[batch_index].append(list[float]())
            time_remaining = 1000.0  # ms
            dist = dist_type(rate=rate * data[batch_index, neuron])  # type: ignore
            while time_remaining > 0:
                dt = dist.sample().item()
                time_remaining -= dt
                if time_remaining < 0:
                    break
                encoded[batch_index][neuron].append(time_remaining)
            encoded[batch_index][neuron].reverse()

    return encoded


if __name__ == "__main__":
    from torchvision import transforms
    from torchvision.datasets import MNIST
    import pdb

    MNIST_train = MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    x = encode_input_continuous(MNIST_train[0][0])
    pdb.set_trace()
