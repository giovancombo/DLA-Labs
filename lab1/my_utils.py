# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab1

import os
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
def save_model(model, dataset, config, model_name):
    folder = f"models/{dataset}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model, folder + model_path(config, model_name) + "-" + str(time.time()) + "model.pt")
    print("Model saved!")


# Assigns a unique name to models for saving
def model_path(config, model_name):
    if model_name == "MLP":
        path = "/mlp-" + "depth" + str(len(config.hidden_size)) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    if model_name == "ResidualMLP":
        path = "/residualmlp-" + "depth" + str(len(config.hidden_size)) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    elif model_name == "CNN":
        path = "/cnn-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    elif model_name == "ResidualCNN":
        path = "/residualcnn-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    elif model_name == "ResNet":
        path = "/resnet" + str(config.resnet_name) + "-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size) + "-dr" + str(config.dropout)
    return path


# Plots some images from the dataset
def plot_images(model, images, labels, epoch, config):
    fig, axs = plt.subplots(1, 5, figsize=(15, 4))

    for i, image_path in enumerate(images):
        img = mpimg.imread(image_path)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(labels[i], fontsize=20)

    plt.tight_layout()
    plt.figtext(0.5, 0.05, "Epoch: " + str(epoch), ha = 'center', fontsize = 20)

    folder = f"images/{config['dataset']}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + f"/{model.__class__.__name__}-epoch{epoch}.png")
    plt.close()