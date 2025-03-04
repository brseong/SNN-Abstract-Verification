import pdb
import torch as th
import torch.nn.functional as F
import wandb

from z3.z3 import Int, Real, Solver, And, Implies, sat, set_param
from utils.encoding.z3 import generate_snn, allocate_input
from utils.encoding.snn import encode_input
from utils.model import AbsMNISTNet, MNISTNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from utils.spikingjelly.spikingjelly.activation_based import functional
from utils.spikingjelly.spikingjelly.activation_based.encoding import PoissonEncoder
from utils.types import Z3Data

batch_size = 1
num_workers = 4
learning_rate = 1e-2
n_steps = 20
save_sexpr = True
load_sexpr = True
hidden_size = 512
model_suffix = f"{hidden_size}"
sexpr_suffix = f"{hidden_size}"
cfg = {
    "batch_size": batch_size,
    "num_workers": num_workers,
    "learning_rate": learning_rate,
    "n_steps": n_steps,
    "save_sexpr": save_sexpr,
    "load_sexpr": load_sexpr,
    "hidden_size": hidden_size,
}

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

th.random.manual_seed(42)  # type: ignore
th.cuda.manual_seed(42)
th.use_deterministic_algorithms(True)


def net_inference_single(model: th.nn.Module, data: th.Tensor) -> th.Tensor:
    """Perform inference on the network.

    Args:
        model (th.nn.Module): Network model
        data (th.Tensor): Input data, shape ``(T, 1, 28, 28)``

    Returns:
        th.Tensor: Output tensor, shape ``(n_maxima,)``
    """
    model.eval()
    y_hat = th.tensor(0.0)
    for t in range(data.shape[0]):
        y_hat = y_hat + model(data[t].unsqueeze(0)).squeeze(0)
    return (y_hat == y_hat.max()).nonzero(as_tuple=True)[0]


@th.no_grad()
def run_verification(model: MNISTNet, data: th.Tensor, target: th.Tensor):
    """Run verification on the network.

    Args:
        model (MNISTNet): MNISTNet model
        data (th.Tensor): Input data, shape ``(batch_size, 1, 28, 28)``
        target (th.Tensor): Target labels, shape ``(batch_size,)``
    """
    data, target = data.to(device), target.to(device)
    data = encode_input(
        data, num_steps=n_steps, as_counts=False
    )  # data.shape = (batch_size, n_steps, 1, 28, 28)

    th_pred = net_inference_single(model, data[0])

    s = Solver()
    weight_list = [model.linear1.weight, model.linear2.weight]
    z3data = Z3Data(
        n_steps=n_steps,
        n_features=[28 * 28, hidden_size, 10],
        n_spikes={},
        weight={},
    )
    save_path = f"saved/sexpr/snn_z3_{sexpr_suffix}.txt" if save_sexpr else None
    generate_snn(
        s,
        weight_list=weight_list,
        data=z3data,
        save_path=save_path,
        load_sexpr=load_sexpr,
    )
    allocate_input(s, data=z3data, _input=data[0].flatten(start_dim=1))

    print("Start solving")
    set_param(verbose=2)
    # set_param("parallel.enable", True)
    while s.check() == sat:
        print(
            f"Solver prediction: {s.model()[Int('prediction')]}, Native prediction: {th_pred}"
        )
        s.add(Int("prediction") != s.model()[Int("prediction")])


if __name__ == "__main__":
    wandb.init(project="snn-abs-verification", config=cfg)

    model = MNISTNet(hidden_features=hidden_size).to(device)
    model.load_state_dict(th.load(f"saved/model_{model_suffix}.pt"), strict=True)  # type: ignore

    MNIST_train = MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader[tuple[th.Tensor, th.Tensor]](
        MNIST_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    i = 0
    for data, target in train_loader:
        run_verification(
            model=model,
            data=data,
            target=target,
        )
        i += 1
        if i == 10:
            break
