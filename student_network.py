from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import student_train


class StudentNetwork:
    def __init__(self, num_classes, num_heads, optimizers, loss_fns):
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.model = MultiHeadNetwork(num_heads=num_heads, num_classes=num_classes)
        self.optimizers = optimizers
        self.loss_fns = loss_fns

    def train_network(self, train_epochs, device, dataloader, lr):
        student_train.train(epochs=train_epochs, device=device, dataloader=dataloader,
                            net=self.model, optimizers=self.optimizers, criterions=self.loss_fns)


class MultiHeadNetwork(ResNet):
    def __init__(self, num_heads, num_classes):
        super(MultiHeadNetwork, self).__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=256)
        # This num_classes is just for the output nodes of base network

        self.num_classes = num_classes
        # This is the real num_classes for student network final output nodes

        head_names = ['head_{}'.format(i) for i in range(num_heads)]
        self.dict_head_nodes = {}
        for head in head_names:
            self.dict_head_nodes[head] = nn.Linear(in_features=256, out_features=self.num_classes)

    def forward(self, x):
        x = self._forward_impl(x)
        output_heads = []
        for head_key in self.dict_head_nodes.keys():
            output_heads.append(self.dict_head_nodes[head_key](x))
        return output_heads

    
    

