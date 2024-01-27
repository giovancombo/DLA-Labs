import torch
from tqdm import tqdm
import numpy as np
import wandb

import aux_functions as aux

# Training Loop
def train(model, train_loader, val_loader, criterion, optimizer, device, config):

    # Telling W&B to watch gradients and the model parameters
    wandb.watch(model, criterion, log = "all", log_freq = config.log_interval)
    example_ct = 0

    print("\nStarting training...")
    for epoch in tqdm(range(config.epochs), desc = "Training Epochs", ncols = 100):
        model.train()
        losses, accuracies = [], []
        for batch, (images, labels) in enumerate(train_loader):
            loss, accuracy = train_batch(model, images, labels, criterion, optimizer, device, config)

            example_ct += len(images)
            losses.append(loss.item())
            accuracies.append(accuracy)
            mean_loss, mean_accuracy = np.mean(losses[-config.log_interval:]), np.mean(accuracies[-config.log_interval:])

            if ((batch + 1) % config.log_interval) == 0:
                aux.log_train(epoch, loss, accuracy, mean_loss, mean_accuracy, example_ct, config)

        # Validation at the end of the epoch
        val_loss, val_accuracy = test(model, val_loader, device, config)

        # Logging losses and accuracies at the end of the epoch
        aux.log_validation(epoch, mean_loss, val_loss, mean_accuracy, val_accuracy, example_ct)
    
    print("Training completed!")


# Function for training a single batch
def train_batch(model, images, labels, criterion, optimizer, device, config):
    if not config.convnet:
        images = images.reshape(-1, config.input_size).to(device)
    else:
        images = images.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()

    # Calculating training accuracy
    correct, total = 0, 0
    _, pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum().item()
        
    accuracy = 100 * correct / total

    return loss, accuracy


# Evaluation Loop (for Validation and Test)
@torch.no_grad()
def test(model, test_loader, device, config):
    test_loss = 0
    correct, total = 0, 0
    model.eval()
    for images, labels in test_loader:
        if not config.convnet:
            images = images.reshape(-1, config.input_size).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        test_loss += F.cross_entropy(outputs, labels, reduction = 'sum')
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / total

    return test_loss, test_accuracy