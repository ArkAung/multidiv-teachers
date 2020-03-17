from tqdm import tqdm
import torch

def test(device, dataloader, net, model_path=None):
    total_datapoints = len(dataloader.dataset.data)
    total_batches = total_datapoints//dataloader.batch_size
    net = net.to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        with tqdm(total=total) as pbar:
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)

        print('Accuracy on {} data points: {}%'.format(total_datapoints, 100 * correct/total))