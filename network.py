from torchvision.models import alexnet
from torchvision.models import vgg11
from torchvision.models import resnet18
from torchvision.models import shufflenet_v2_x0_5
from torchvision.models import inception_v3
from train import train
from test import test

ARCH_NAMES = ['alexnet', 'vgg', 'resnet', 'shufflenet', 'inception']

class NetworkBuilder:
    def __init__(self, num_classes, arch, optimizer, loss_fn):
        self.arch_dict = {'alexnet': alexnet,
                            'vgg': vgg11,
                            'resnet': resnet18,
                            'shufflenet': shufflenet_v2_x0_5,
                            'inception': inception_v3}
        self.num_classes = num_classes
        self.arch = arch
        self.model = self.build_network() 
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def build_network(self):
        assert self.arch in ARCH_NAMES
        network = self.arch_dict[self.arch](num_classes=self.num_classes)
        network = network.to(self.device)
        return network

    def train_network(self, train_epochs, device, dataloader, lr):
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        train(epochs=train_epochs, device=device, dataloader=dataloader,
                net=self.model, optimizer=optimizer, criterion=self.loss_fn)

    def test_network(self, device, dataloader):
        test(device=device, dataloader=dataloader)

    def save_network_params(self, save_path):
        torch.save(self.model.state_dict(), f=save_path)
