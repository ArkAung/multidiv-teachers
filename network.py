import torch
import torchvision.models as models
from train import train
from test import test

ARCH_NAMES = ['vgg', 'resnet', 'shufflenet', 'squeeze', 'resnext', 'mnasnet']

class NetworkBuilder:
    def __init__(self, num_classes, arch, optimizer, loss_fn):
        self.arch_dict = {
                            'vgg': models.vgg11,
                            'resnet': models.resnet18,
                            'shufflenet': models.shufflenet_v2_x1_0,
                            'squeeze': models.squeezenet1_0,
                            'resnext': models.resnext50_32x4d,
                            'mnasnet': models.mnasnet1_0
                        }

        self.num_classes = num_classes
        self.arch = arch
        self.model = self.build_network() 
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def build_network(self):
        assert self.arch in ARCH_NAMES
        network = self.arch_dict[self.arch](num_classes=self.num_classes)
        return network

    def train_network(self, train_epochs, device, dataloader, lr):
        self.model = self.model.to(device)
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        train(epochs=train_epochs, device=device, dataloader=dataloader,
                net=self.model, optimizer=optimizer, criterion=self.loss_fn)

    def test_network(self, device, dataloader, model_path=None, save_softmax_outputs=False, softmax_save_path=None):
        test(device=device, dataloader=dataloader, net=self.model, model_path=model_path, 
             save_softmax_outputs=save_softmax_outputs, softmax_save_path=softmax_save_path)

    def save_network_params(self, save_path):
        torch.save(self.model.state_dict(), f=save_path)
