from functools import partial
import pdb
import torch as th
import torch.nn.functional as F

from utils.model import AbsMNISTNet, MNISTNet
from utils.encoding.snn import encode_input
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from utils.spikingjelly.spikingjelly.activation_based import functional, encoding


def train(
    net: MNISTNet,
    dataset: Dataset[tuple[th.Tensor, th.Tensor]],
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    T: int = 20,
    save: bool = False,
    abs_test: bool = False,
) -> None:
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    optim = th.optim.Adam(net.parameters(), lr=learning_rate)
    encoder = encoding.PoissonEncoder()
    for epoch in range(num_epochs):
        total_loss_train = 0
        total_acc_train = 0
        net.train()
        for data, target in tqdm(iter(data_loader)):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            target_onehot = F.one_hot(target, 10).float()
            y_hat = th.tensor(0.0)
            for _ in range(T):
                encode = encoder(data)
                y_hat = y_hat + net(encode)
            y_hat = y_hat / T
            loss = F.mse_loss(y_hat, target_onehot)
            loss.backward()
            optim.step()
            total_loss_train += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_train += (pred_target == target).sum()
            functional.reset_net(net)
        loss_train = total_loss_train / len(dataset)
        acc_train = (total_acc_train / len(dataset)) * 100
        print(f"{epoch + 1} epoch loss : {loss_train}, train accuracy : {acc_train}")
        if save:
            th.save(net.state_dict(), "./saved/model.pt")

        test_model = net
        if abs_test:
            test_model = AbsMNISTNet(hidden_features=hidden_features).to(device)
            test_model.load_state_dict(th.load("./saved/model.pt"))

            train_loss, train_acc = test(net=test_model, dataset=MNIST_train, T=T)
            print(
                f"abstraction train loss = {train_loss}, abstraction train accuracy = {train_acc}"
            )

        test_loss, test_acc = test(net=test_model, dataset=MNIST_test, T=T)
        print(
            f"{'abstraction ' if abs_test else ''}test loss = {test_loss}, {'abstraction ' if abs_test else ''}test accuracy = {test_acc}"
        )


@th.no_grad()
def test(net: MNISTNet, dataset: Dataset[tuple[th.Tensor, th.Tensor]], T: int = 20):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    net.eval()
    total_loss_test = 0
    total_acc_test = 0
    for data, target in tqdm(iter(data_loader)):
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target, 10).float()
        y_hat = th.tensor(0.0)
        if isinstance(net, AbsMNISTNet):
            y_hat = net(encode_input(data, num_steps=T, as_counts=True))
        else:
            encode = encode_input(data, num_steps=T)
            for t in range(T):
                y_hat = y_hat + net(encode[:, t])
        y_hat = y_hat / T
        loss = F.mse_loss(y_hat, target_onehot)
        total_loss_test += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc_test += (pred_target == target).sum()
        functional.reset_net(net)
    loss_test = total_loss_test / len(dataset)
    acc_test = (total_acc_test / len(dataset)) * 100
    return loss_test, acc_test


if __name__ == "__main__":
    th.manual_seed(0)
    th.cuda.manual_seed(0)
    th.use_deterministic_algorithms(True)

    num_epochs = 50
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-2
    num_steps = 20
    hidden_features = 32
    save = True

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    net = MNISTNet(hidden_features=hidden_features).to(device)

    MNIST_train = MNIST(
        root="./data", download=True, train=True, transform=transforms.ToTensor()
    )
    MNIST_test = MNIST(
        root="./data", download=True, train=False, transform=transforms.ToTensor()
    )

    train(
        net=net,
        dataset=MNIST_train,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        T=num_steps,
        save=save,
        abs_test=False,
    )
