import pdb
import torch as th
import torch.nn.functional as F

from z3.z3 import Int, Real, Solver, And, Implies, sat
from utils.encoding.encoding import generate_snn, allocate_input
from utils.model import MNISTNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from utils.spikingjelly.spikingjelly.activation_based import functional
from utils.spikingjelly.spikingjelly.activation_based.encoding import PoissonEncoder

num_epochs = 50
batch_size = 1
num_workers = 4
learning_rate = 1e-2
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

th.random.manual_seed(42)  # type: ignore
th.use_deterministic_algorithms(True)


def encode_input(
    data: th.Tensor, num_steps: int, encoder: PoissonEncoder = PoissonEncoder()
) -> th.Tensor:
    """Encode the input data into spikes with Poisson process.

    Args:
        data (th.Tensor): Input data, shape (batch_size, 1, 28, 28)
        num_steps (int): Number of steps in the simulation
        encoder (PoissonEncoder, optional): Encoder object. Defaults to PoissonEncoder().

    Returns:
        th.Tensor: Encoded spikes, shape (num_steps, batch_size, 1, 28, 28)
    """
    encoded = th.zeros(num_steps, *data.shape)
    for t in range(num_steps):
        encoded[t] = encoder(data)
    return encoded


if __name__ == "__main__":
    model = MNISTNet()
    model.load_state_dict(th.load("saved/model.pt"), strict=True)  # type: ignore

    MNIST_train = MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader[tuple[th.Tensor, th.Tensor]](
        MNIST_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    pdb.set_trace()

    x = Real("x")
    y = Int("y")
    z = Int("z")

    s = Solver()
    s.add(x == 0.5)
    s.add(floor(x, y))

    if s.check() == sat:
        print(s.model())
    else:
        print("unsat")
