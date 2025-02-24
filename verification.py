import torch as th
import torch.nn.functional as F

from z3.z3 import Int, Real, Solver, And, Implies, sat
from utils.encoding.encoding import generate_snn
from utils.model import MNISTNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from utils.spikingjelly.spikingjelly.activation_based import functional, encoding

if __name__ == "__main__":
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
