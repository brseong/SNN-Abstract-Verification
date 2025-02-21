from typing import Callable
from utils.model import MNISTNet
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.functional as F
from tqdm.auto import tqdm
from utils.spikingjelly.spikingjelly.activation_based import functional


def train(data_loader : DataLoader,
          loss_fn,
          num_epochs : int = 10,
          batch_size : int = 32, 
          learning_rate : float = 1e-3,
          ) -> None : 
    Optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        total_Loss_train = 0
        total_acc_train = 0
        net.train()
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data = data.to(device)
            target = target.to(device)
            Optimizer.zero_grad()
            y_hat = net(data)
            Loss = loss_fn(y_hat, target)
            Loss.backward()
            Optimizer.step()
            total_Loss_train += Loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_train += (pred_target == target).sum()
            functional.reset_net(net)
        Loss_train = total_Loss_train / (60000 / batch_size)
        acc_train = (total_acc_train / 60000) * 100 
        print(f'{epoch + 1} epoch\'s of Loss : {Loss_train}, accuracy rate : {acc_train}')
        
def test(data_loader : DataLoader, 
         loss_fn,) : 
    net.eval()
    total_Loss_test = 0
    total_acc_test = 0
    with th.no_grad():
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data = data.to(device)
            target = target.to(device)
            y_hat = net(data)
            Loss = loss_fn(y_hat, target)
            total_Loss_test += Loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_test += (pred_target == target).sum()
            functional.reset_net(net)
        Loss_test = total_Loss_test / (10000/32)
        acc_test = (total_acc_test / 10000) * 100
    return Loss_test, acc_test
    
    
if __name__ == "__main__":
    num_epochs = 50
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-2
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    net = MNISTNet().to(device)
    
    MNIST_train = MNIST(root = './data', download = True, train = True, transform = transforms.ToTensor())
    MNIST_test = MNIST(root = './data', download = True, train = False, transform = transforms.ToTensor())

    train_loader = DataLoader(
        MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
    )
    test_loader = DataLoader(
        MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
    )
    Loss_function= nn.CrossEntropyLoss().to(device)
    train(num_epochs = num_epochs,
          batch_size = batch_size,
          learning_rate = learning_rate,
          data_loader= train_loader,
          loss_fn = Loss_function
          )
    test_loss, test_acc = test(
        data_loader=test_loader,
        loss_fn = Loss_function
        )
    print(f'test loss = {test_loss}, and test accuracy = {test_acc}')
    th.save(net.state_dict(), './saved/model.pt')
        
