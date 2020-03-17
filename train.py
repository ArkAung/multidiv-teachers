"""
    Modular train function which takes in
"""
from tqdm import tqdm

def train(epochs, device, dataloader, net, optimizer, criterion):
    total_batches = len(dataloader.dataset.data)//dataloader.batch_size
    
    for e in range(1, epochs+1): # For each epoch
        running_loss = 0
        num_datapoints = 0
        correct = 0
        with tqdm(total=total_batches) as pbar:
            for batch_idx, data in enumerate(dataloader):  # For each batch
                images, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_datapoints += len(labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            mean_loss = running_loss/num_datapoints
            pbar.update(1)
        print("Epoch: {} -- Average Loss: {} -- Acc: {}%".format(e, mean_loss, 100*correct/num_datapoints))