# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab1

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

import my_utils

class Trainer:
    def __init__(self, dataset, model, device, log_freq, convnet = False):
        self.dataset = dataset        
        self.model = model
        self.device = device
        self.log_freq = log_freq
        self.convnet = convnet
        self.global_step = 0

    def train(self, train_loader, criterion, optimizer, epoch, scheduler = None):
        self.model.train()
        _, input_size, _ = my_utils.set_shapes(self.dataset)
        train_loss = 0
        batches_ct = 0
        correct, total = 0, 0
        losses, accuracies = [], []

        for batch, (images, labels) in tqdm(enumerate(train_loader), desc = f'Training epoch {epoch + 1}'):
            optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            if not self.convnet:
                outputs = self.model(images.reshape(-1, input_size))
            else:
                outputs = self.model(images)

            loss = criterion(outputs, labels)
            train_loss += loss.item()
            batches_ct += 1
            self.global_step += len(images)            

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculating training accuracy
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            batch_accuracy = 100 * correct / total

            losses.append(loss.item())
            accuracies.append(batch_accuracy)
            mean_loss, mean_accuracy = np.mean(losses[-self.log_freq:]), np.mean(accuracies[-self.log_freq:])

            if ((batch + 1) % self.log_freq) == 0:
                print(f'Epoch {epoch + 1} | Train Loss = {mean_loss:.4f}; Train Accuracy = {mean_accuracy:.2f}%')
                wandb.log({"Training/Training Loss": loss.item(),
                            "Training/Training Accuracy": batch_accuracy,
                            "Training/Training Epochs": epoch + 1}, step = self.global_step)
            
        if scheduler:
            scheduler.step()

        return mean_loss, mean_accuracy


    @torch.no_grad()
    def evaluate(self, test_loader, epoch):
        self.model.eval()
        _, input_size, _ = my_utils.set_shapes(self.dataset)
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc = f'Testing epoch {epoch + 1}'):
                images, labels = images.to(self.device), labels.to(self.device)
                if not self.convnet:
                    outputs = self.model(images.reshape(-1, input_size))
                else:
                   outputs = self.model(images)

                test_loss += F.cross_entropy(outputs, labels, reduction = 'sum')
                _, pred = torch.max(outputs.data, 1)
                correct += (pred == labels).sum().item()

            test_loss /= len(test_loader.dataset)
            test_accuracy = 100. * correct / len(test_loader.dataset)

        return test_loss, test_accuracy
