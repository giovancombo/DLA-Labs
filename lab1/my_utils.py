# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab1

import os
import torch
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


# Saves the model checkpoint
def save_checkpoint(epoch, model, optimizer, best_accuracy, directory, is_best = False):
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'best_accuracy': best_accuracy
                  }
    if is_best:
        torch.save(checkpoint, os.path.join(directory, f'bestmodel.pth'))
    torch.save(checkpoint, os.path.join(directory, f'checkpoint_ep{epoch+1}.pth'))
