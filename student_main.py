import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
from student_network import StudentNetwork
from data import get_dataloader, get_transforms

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

    num_heads = 3
    softmax_files = ['outputs_softmax_resnet', 'outputs_softmax_resnext', 'outputs_softmax_shufflenet']
    ## End of Global args ##

    print("Device: {}".format(torch.cuda.get_device_name()))

    train_dataset = CIFAR10(root='datasets', train=False,
                            transform=get_transforms(resize_shape=64), download=True)

    arr_softmax_outputs = []
    for sf in softmax_files:
        arr_softmax_outputs.append(torch.load(os.path.join('softmax_outputs', '{}.pt'.format(sf)),
                                              map_location=torch.device('cpu')))
    combined_output = torch.stack(tuple(arr_softmax_outputs), dim=1)
    train_dataset.targets = combined_output

    train_dataloader = get_dataloader(dataset=train_dataset,
                                      bs=train_batch_size,
                                      num_workers=train_dataloader_num_workers)

    num_classes = len(train_dataset.classes)
    optimizers = [optim.Adam for _ in range(num_heads)]
    loss_fns = [CrossEntropyLoss() for _ in range(num_heads)]

    print("Training")
    network = StudentNetwork(num_classes=num_classes, num_heads=num_heads, optimizers=optimizers, loss_fns=loss_fns)
    network.train_network(train_epochs=train_epochs, device=device,
                          dataloader=train_dataloader, lr=learning_rate)
    # model_save_path = 'models/cifar_{}.pth'.format(ARCH)
    # network.save_network_params(save_path=model_save_path)
    # softmax_save_path = 'softmax_outputs/outputs_softmax_{}.pt'.format(ARCH)
    # network.test_network(device=device, dataloader=test_dataloader,
    #                      model_path=model_save_path,
    #                      save_softmax_outputs=True, softmax_save_path=softmax_save_path)
