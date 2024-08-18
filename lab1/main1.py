# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab1

# Code for EXERCISES 1.1, 1.2 and 2.1

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import time
import yaml
import wandb

from models import MLP, CNN, ResidualMLP, ResidualCNN, ResNet
from trainer import Trainer
import my_utils


if __name__ == '__main__':        
    # Loading the configuration file
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations applied to the dataset
    assert params['dataset'] in ["MNIST", "CIFAR10"], "Dataset not supported!"
    if params['dataset'] == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    elif params['dataset'] == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_data = getattr(torchvision.datasets, params['dataset'])("./data", train = True, download = True, transform = transform)
    test_set = getattr(torchvision.datasets, params['dataset'])("./data", train = False, download = True, transform = transform)
    
    # Splitting the Training Set into Training and Validation Sets
    val_size = 10_000
    train_set, val_set = random_split(train_data, [len(train_data) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size = params['batch_size'], shuffle = True, num_workers = params['num_workers'])
    val_loader = DataLoader(val_set, batch_size = params['batch_size'], num_workers = params['num_workers'])
    test_loader = DataLoader(test_set, batch_size = params['batch_size'], num_workers = params['num_workers'])

    print(f"Dataset {params['dataset']} loaded with {len(train_set)} Train samples, {len(val_set)} Validation samples, {len(test_set)} Test samples.\n")


    # Instantiating the model
    input_shape, input_size, classes = my_utils.set_shapes(params['dataset'])

    if params['convnet']:
        if params['residual']:
            if params['resnet']:
                model = ResNet(params['resnet_name'], input_shape, params['resnet_hidden_size'], classes, params['dropout']).to(device)
            else:
                model = ResidualCNN(input_shape, params['CNN_hidden_size'], classes, params['depth'], params['dropout']).to(device)
        else:
            model = CNN(input_shape, params['CNN_hidden_size'], classes, params['depth'], params['dropout']).to(device)
    elif params['residual']:
        model = ResidualMLP(params['depth'], input_size, params['MLP_hidden_size'], classes, params['dropout']).to(device)
    else:
        model = MLP(input_size, params['MLP_hidden_size'], classes, params['dropout']).to(device)
        

    # Defining the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = params['learning_rate'], weight_decay = params['weight_decay'])
    
    if params['scheduler']:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = params['gamma'])
    else:
        scheduler = None

    print(f"Model instantiated: {model.__class__.__name__}")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    print(f"Optimizer: {optimizer.__class__.__name__}; Loss: {criterion}; Device: {device}")


    # Training
    wandb.login()
    with wandb.init(project = 'DLA_Lab1_CNN', config = params, name = f"{model.__class__.__name__}-{params['dataset']}_depth{params['depth']}_bs{params['batch_size']}_lr{params['learning_rate']}_wd{params['weight_decay']}_dr{params['dropout']}"):
        config = wandb.config

        wandb.watch(model, criterion, log = "all", log_freq = params['log_freq'])
        trainer = Trainer(train_data, model, device, params['log_freq'], params['convnet'])

        # Directory for saving checkpoints
        checkpoint_dir = f"checkpoints/{params['dataset']}/{model.__class__.__name__}_depth{params['depth']}_ep{params['epochs']}_bs{params['batch_size']}_lr{params['learning_rate']}_wd{params['weight_decay']}_dr{params['dropout']}_" + str(time.time())[:8]
        os.makedirs(checkpoint_dir, exist_ok = True)

        # Epochs loop
        best_val_acc = 0
        for epoch in range(params['epochs']):
            train_loss, train_acc = trainer.train(train_loader, criterion, optimizer, epoch, scheduler)
            val_loss, val_acc = trainer.evaluate(val_loader)
     
            print(f'\nEnd of epoch {epoch + 1} | Validation Loss: {val_loss:.4f}; Validation Accuracy: {val_acc:.2f}%\n')
            wandb.log({"Train Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Epoch": epoch + 1,
                    "Train Accuracy": train_acc,
                    "Validation Accuracy": val_acc}, step = trainer.global_step)

            # Saving checkpoints
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                my_utils.save_checkpoint(epoch, model, optimizer, best_val_acc, checkpoint_dir, is_best = True)
                print("New best model saved!")
            
            my_utils.save_checkpoint(epoch, model, optimizer, best_val_acc, checkpoint_dir)
            print(f"Checkpoint saved for epoch {epoch+1}")

        # Testing
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f'\nEnd of Testing | Test Loss: {test_loss:.4f}; Test Accuracy: {test_acc:.2f}%')
        wandb.log({"Test Loss": test_loss, "Test Accuracy": test_acc})
