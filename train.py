from utils.model import MNISTNet
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.spikingjelly.spikingjelly.activation_based import functional, encoding


def train(data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
          num_epochs : int = 10,
          batch_size : int = 32, 
          learning_rate : float = 1e-3,
          T : int = 20
          ) -> None :
    optim = th.optim.Adam(net.parameters(), lr = learning_rate)
    encoder = encoding.PoissonEncoder()
    for epoch in range(num_epochs):
        total_loss_train = 0
        total_acc_train = 0
        net.train()
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            target_onehot = F.one_hot(target, 10).float()
            y_hat = 0.0
            for _ in range(T):
                encode = encoder(data)
                y_hat += net(encode)
            y_hat = y_hat / T 
            loss = F.mse_loss(y_hat, target_onehot)
            loss.backward()
            optim.step()
            total_loss_train += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_train += (pred_target == target).sum()
            functional.reset_net(net)
        Loss_train = total_loss_train / (60000 / batch_size)
        acc_train = (total_acc_train / 60000) * 100 
        print(f'{epoch + 1} epoch\'s of Loss : {Loss_train}, accuracy rate : {acc_train}')
        
def test(data_loader : DataLoader[tuple[th.Tensor, th.Tensor]], 
         T : int = 20) :
    encoder = encoding.PoissonEncoder() 
    net.eval()
    total_Loss_test = 0
    total_acc_test = 0
    with th.no_grad():
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data, target = data.to(device), target.to(device)
            target_onehot = F.one_hot(target, 10).float()
            y_hat = 0.0
            for _ in range(T):
                encode = encoder(data)
                y_hat += net(encode)
            y_hat = y_hat / T
            Loss = F.mse_loss(y_hat, target_onehot)
            total_Loss_test += Loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_test += (pred_target == target).sum()
            functional.reset_net(net)
        Loss_test = total_Loss_test / (10000/32)
        acc_test = (total_acc_test / 10000) * 100
    return Loss_test, acc_test
    
    
if __name__ == "__main__":
    th.manual_seed(0)
    th.cuda.manual_seed(0)
    
    num_epochs = 50
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-2
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    net = MNISTNet().to(device)
    
    MNIST_train = MNIST(root = './data', download = True, train = True, transform = transforms.ToTensor())
    MNIST_test = MNIST(root = './data', download = True, train = False, transform = transforms.ToTensor())

    train_loader = DataLoader[tuple[th.Tensor, th.Tensor]](
        MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
    )
    test_loader = DataLoader[tuple[th.Tensor, th.Tensor]](
        MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
    )
    train(num_epochs = num_epochs,
          batch_size = batch_size,
          learning_rate = learning_rate,
          data_loader= train_loader,
          T = 20
          )
    test_loss, test_acc = test(
        data_loader=test_loader,
        T = 20
        )
    print(f'test loss = {test_loss}, and test accuracy = {test_acc}')
    th.save(net.state_dict(), './saved/model.pt')
        
