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
from utils.types import Z3Data

num_epochs = 50
batch_size = 1
num_workers = 4
learning_rate = 1e-2
n_steps = 20
load_sexpr = False
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
        th.Tensor: Encoded spikes, shape (batch_size, num_steps, 1, 28, 28)
    """
    batch_size, *features = data.shape
    encoded = th.zeros(batch_size, num_steps, *features)
    for t in range(num_steps):
        encoded[:, t] = encoder(data)
    return encoded


def net_inference(model: th.nn.Module, data: th.Tensor, num_steps: int) -> th.Tensor:
    """Perform inference on the network.

    Args:
        model (th.nn.Module): Network model
        data (th.Tensor): Input data, shape (batch_size, num_steps, 1, 28, 28)
        num_steps (int): Number of steps in the simulation

    Returns:
        th.Tensor: Output tensor, shape (batch_size, 10)
    """
    model.eval()
    y_hat = th.tensor(0)
    for t in range(num_steps):
        y_hat += model(data[:, t].flatten(start_dim=1))
    return y_hat.argmax(dim=1)


if __name__ == "__main__":
    model = MNISTNet().to(device)
    model.load_state_dict(th.load("saved/model.pt"), strict=True)  # type: ignore

    MNIST_train = MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader[tuple[th.Tensor, th.Tensor]](
        MNIST_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    data, target = next(
        iter(train_loader)
    )  # data.shape = (batch_size, 1, 28, 28), target.shape = (batch_size,)
    data, target = data.to(device), target.to(device)
    data = encode_input(
        data, num_steps=n_steps
    )  # data.shape = (batch_size, num_steps, 1, 28, 28)

    s = Solver()
    weight_list = [model.linear1.weight, model.linear2.weight]
    z3data = Z3Data(
        n_steps=n_steps,
        n_features=[28 * 28, 512, 10],
        n_spikes={},
        weight={},
    )
    save_path = None if not load_sexpr else "saved/sexpr/snn_z3.txt"
    generate_snn(s, weight_list=weight_list, data=z3data, save_path=save_path)
    allocate_input(s, data=z3data, _input=data[0].flatten(start_dim=1))

    print("Start solving")
    if s.check() == sat:
        print(
            f"Solver prediction: {s.model()['prediction']}, Native prediction: {net_inference(model, data[0], n_steps)}"
        )
    else:
        print("unsat")
