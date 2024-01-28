import os
import time
import torch
import wandb

# Automatically sets correct shapes for model inputs basing on the dataset chosen
def set_shapes(dataset):
    if dataset == "MNIST":
        input_shape = [1, 28, 28]
        input_size = 28*28*1
        classes = 10
    elif dataset == "CIFAR10":
        input_shape = [3, 32, 32]
        input_size = 32*32*3
        classes = 10
    
    return input_shape, input_size, classes


# Saves the model
def save_model(model, config):
    folder = str(f"models/{config['dataset']}/" + model.__class__.__name__)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model, folder + model_path(config, model.__class__.__name__) + ".pt")
    print("Model saved!")


# Assigns a unique name to models for saving
def model_path(config, folder):
    if folder == "MLP":
        path = "/mlp-" + "depth" + str(len(config.hidden_size)) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    if folder == "ResidualMLP":
        path = "/residualmlp-" + "depth" + str(len(config.hidden_size)) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    elif folder == "CNN":
        path = "/cnn-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    elif folder == "ResidualCNN":
        path = "/residualcnn-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    elif folder == "ResNet":
        path = "/resnet" + str(config.resnet_name) + "-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    return path
