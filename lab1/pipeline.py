import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import wandb

import models
import utils
import main23

# Loads the dataset
def load(config):

    assert config.dataset in ["MNIST", "CIFAR10"], "Dataset not supported!"

    # Transformations applied to the dataset
    if config.dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    elif config.dataset == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_data = getattr(torchvision.datasets, config.dataset)("./data", train = True, download = True, transform = transform)
    test_set = getattr(torchvision.datasets, config.dataset)("./data", train = False, download = True, transform = transform)
    
    # Splitting the Training Set into Training and Validation Sets
    train_set, val_set = random_split(train_data, [len(train_data) - config.val_size, config.val_size])

    train_loader = DataLoader(train_set, config.batch_size, shuffle = True, num_workers = config.num_workers)
    val_loader = DataLoader(val_set, config.batch_size, num_workers = config.num_workers)
    test_loader = DataLoader(test_set, config.batch_size, num_workers = config.num_workers)

    print(f"Dataset {config.dataset} loaded with {len(train_set)} Train samples, {len(val_set)} Validation samples, {len(test_set)} Test samples.\n")

    return train_loader, val_loader, test_loader
    

# Instantiates model, loss function and optimizer
def build_model(device, config: dict):

    input_shape, input_size, classes = utils.set_shapes(config.dataset)

    # Instantiating the model
    if config.convnet:
        if config.fullycnn:
            pass
        elif config.residual:
            if config.resnet:
                m = models.ResNet(config.resnet_name, input_shape, config.resnet_hidden_size, classes, config.activation, config.use_bn, config.dropout)
            else:
                m = models.ResidualCNN(input_shape, config.CNN_hidden_size, classes, config.depth, config.activation, config.use_bn)
        else:
            m = models.CNN(input_shape, config.CNN_hidden_size, classes, config.depth, config.kernel_size, config.stride, config.padding, config.activation, config.dropout, config.pool, config.pool_size, config.use_bn)
    elif config.residual:
        m = models.ResidualMLP(input_size, config.MLP_hidden_size, classes, config.activation, config.dropout)
    else:
        m = models.MLP(input_size, config.MLP_hidden_size, classes, config.activation, config.dropout)
    
    model = m.to(device)

    # Defining the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()

    assert config.optimizer in ["Adam", "SGD", "RMSprop"], "Optimizer not supported!"

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = config.learning_rate,
                                     weight_decay = config.weight_decay)
    elif config.optimizer == "SGD" or config.optimizer == "RMSprop":
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters(),
                                                           lr = config.learning_rate,
                                                           momentum = config.momentum,
                                                           weight_decay = config.weight_decay)

    print(f"Model instantiated: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    print(model)
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(optimizer)
    print(f"Loss: {criterion}")
    print(f"Device: {device}")

    return model, criterion, optimizer


# Training Loop
def train(model, train_loader, val_loader, criterion, optimizer, device, config):
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
                log_train(epoch, loss, accuracy, mean_loss, mean_accuracy, example_ct, config)

        # Validation at the end of the epoch
        val_loss, val_accuracy = test(model, val_loader, device, config)

        # Logging losses and accuracies at the end of the epoch
        log_validation(epoch, mean_loss, val_loss, mean_accuracy, val_accuracy, example_ct)

        if config.fullycnn:
            main23.cam_test(model, val_loader, epoch)
    
    print("Training completed!")


# Training of a single batch
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


# Periodically logs Loss and Accuracy from Training and Evaluation phases.
# Periodically logs training data
def log_train(epoch, loss, accuracy, mean_loss, mean_acc, example_ct, config):
    print(f'Epoch {epoch + 1}/{config.epochs} | Train Loss = {mean_loss:.4f}; Train Accuracy = {mean_acc:.2f}%')
    wandb.log({"Training/Training Loss": loss,
                "Training/Training Accuracy": accuracy,
                "Training/Training Epochs": epoch + 1}, step = example_ct)

# Logs validation data at the end of an epoch
def log_validation(epoch, mean_loss, val_loss, mean_accuracy, val_accuracy, example_ct):
    print(f'\nEnd of epoch {epoch + 1} | Validation Loss: {val_loss:.4f}; Validation Accuracy: {val_accuracy}%\n')

    wandb.log({"Train Loss": mean_loss, 
               "Validation Loss": val_loss,
               "Epoch": epoch + 1,
               "Train Accuracy": mean_accuracy,
               "Validation Accuracy": val_accuracy}, step = example_ct)