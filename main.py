from data import get_dataloader, get_transforms
from torchvision.datasets import CIFAR10
from network import NetworkBuilder
from network import ARCH_NAMES

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import os


def make_folders():
    if not os.path.isdir('models'):
        os.makedirs('models')
    if not os.path.isdir('softmax_outputs'):
        os.makedirs('softmax_outputs')


if __name__ == "__main__":
    ## Global args will be moved to argparse ##
    train_batch_size = 256
    train_dataloader_num_workers = 2
    test_batch_size = 256
    test_dataloader_num_workers = 2
    train_epochs = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-3
    phase = 'train'
    ## End of Global args ##
    make_folders()

    print("Device: {}".format(torch.cuda.get_device_name()))

    train_dataset = CIFAR10(root='datasets', train=True,
                            transform=get_transforms(resize_shape=64), download=True)
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      bs=train_batch_size,
                                      num_workers=train_dataloader_num_workers)

    test_dataset = CIFAR10(root='datasets', train=False,
                           transform=get_transforms(resize_shape=64), download=True)
    test_dataloader = get_dataloader(dataset=test_dataset,
                                     bs=test_batch_size,
                                     num_workers=test_dataloader_num_workers)
    num_classes = len(train_dataset.classes)
    optimizer = optim.Adam
    loss_fn = CrossEntropyLoss()

    for ARCH in ARCH_NAMES:
        print("Training:: {}".format(ARCH))
        network = NetworkBuilder(num_classes=num_classes, arch=ARCH, optimizer=optimizer, loss_fn=loss_fn)
        network.train_network(train_epochs=train_epochs, device=device,
                              dataloader=train_dataloader, lr=learning_rate)
        model_save_path = 'models/cifar_{}.pth'.format(ARCH)
        network.save_network_params(save_path=model_save_path)
        softmax_save_path = 'softmax_outputs/outputs_softmax_{}.pt'.format(ARCH)
        network.test_network(device=device, dataloader=test_dataloader,
                             model_path=model_save_path,
                             save_softmax_outputs=True, softmax_save_path=softmax_save_path)
