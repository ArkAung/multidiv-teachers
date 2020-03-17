from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_dataloader(dataset, bs, num_workers):
    return DataLoader(dataset,batch_size=bs,num_workers=num_workers)

def get_transforms():
    return transforms.Compose([transforms.ToTensor()])