"""
    Modular train function for multiple-heads
    
    TODO: Combine this with train.py
"""
from tqdm import tqdm
import torch


def train(epochs, device, dataloader, net, optimizers, criterions):
    total_batches = len(dataloader.dataset.data) // dataloader.batch_size

    for e in range(1, epochs + 1):  # For each epoch
        running_loss = 0
        num_datapoints = 0
        correct = 0
        with tqdm(total=total_batches) as pbar:
            for batch_idx, data in enumerate(dataloader):  # For each batch
                images, labels = data[0].to(device), data[1].to(device)

                for opt in optimizers:
                    opt.zero_grad()

                outputs = net(images)

                arr_loss = []
                for output, lbl, crit in zip(outputs, labels, criterions):
                    loss = crit(output, lbl)
                    arr_loss.append(loss)
                sum_loss = sum(arr_loss)
                sum_loss.backward()

                for opt in optimizers:
                    opt.step()
                
                running_loss += sum_loss()
                num_datapoints += images.size[0]
                pbar.update(1)

        mean_loss = running_loss / num_datapoints
        print("Epoch: {} -- Average Loss: {}".format(e, mean_loss))
