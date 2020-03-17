from train import train
from data import get_dataloader, get_transforms
from torchvision.datasets import CIFAR10
from network import NetworkBuilder
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

if __name__ == "__main__":
    ## Global args will be moved to argparse ##
    train_batch_size = 256
    train_dataloader_num_workers = 2
    train_epochs = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-3
    ## End of Global args ##
    
    train_dataset = CIFAR10(root='datasets', train=True, transform=get_transforms(), download=True)
    train_dataloader = get_dataloader(dataset=train_dataset,
                                        bs=train_batch_size, 
                                        num_workers=train_dataloader_num_workers)
    num_classes = len(train_dataset.classes)
    optimizer = optim.Adam
    loss_fn = CrossEntropyLoss()
    network = NetworkBuilder(num_classes=num_classes, arch='vgg', optimizer=optimizer, loss_fn=loss_fn)
    network.train_network(train_epochs=train_epochs, device=device, 
                            dataloader=train_dataloader, lr=learning_rate)
