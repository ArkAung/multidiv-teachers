from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch


def test(device, dataloader, net, model_path=None, save_softmax_outputs=False, softmax_save_path=None):
    total_datapoints = len(dataloader.dataset.data)
    total_batches = total_datapoints // dataloader.batch_size
    net = net.to(device)
    arr_softmax_outputs = []
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    total = 0
    correct = 0
    with torch.no_grad():
        with tqdm(total=total_batches) as pbar:
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                softmax_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                arr_softmax_outputs.append(softmax_outputs)
                pbar.update(1)

        print('Test accuracy on {} data points: {}%'.format(total_datapoints, 100 * correct / total))
    if save_softmax_outputs:
        assert softmax_save_path is not None
        torch.save(torch.cat(arr_softmax_outputs), f=softmax_save_path)
