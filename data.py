from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_dataloader(dataset, bs, num_workers):
    return DataLoader(dataset,batch_size=bs,num_workers=num_workers)

def get_transforms(resize_shape=None):
    transforms_pipeline = []
    if resize_shape is not None:
        transforms_pipeline.append(transforms.Resize(resize_shape))
    transforms_pipeline.append(transforms.ToTensor())
    return transforms.Compose(transforms_pipeline)
