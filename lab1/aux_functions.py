import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def model_path(config, folder):
    if folder == "MLP":
        path = "/mlp-" + "depth" + str(len(config.hidden_size)) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size)
    if folder == "ResidualMLP":
        path = "/residualmlp-" + "depth" + str(len(config.hidden_size)) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size)
    elif folder == "CNN":
        path = "/cnn-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size)
    elif folder == "ResidualCNN":
        path = "/residualcnn-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size)
    elif folder == "ResNet":
        path = "/resnet" + str(config.resnet_name) + "-depth" + str(config.depth) + "-ep" + str(config.epochs) + "-lr" + str(config.learning_rate) + "-bs" + str(config.batch_size)
    return path


# Functions for simplifying the choice of dataset and architecture to train (not so needed, just for fun)
# SWEEP FOR MLP ON MNIST DATASET
def mlp_mnist(config, mlp_config):
    config.update({"dataset": "MNIST", "architecture": "MLP",
                   'input_shape': (1, 28, 28), 'input_size': 28*28*1, 'classes': 10})
    config = config | mlp_config
    return config

# SWEEP FOR MLP ON CIFAR10 DATASET
def mlp_cifar(config, mlp_config):
    config.update({"dataset": "CIFAR10", "architecture": "MLP",
                   'input_shape': (3, 32, 32), 'input_size': 32*32*3, 'classes': 10})
    config = config | mlp_config
    return config

# SWEEP FOR CONVNETS ON MNIST DATASET
def cnn_mnist(config, convnet_config):
    config.update({"dataset": "MNIST", "architecture": "CNN",
                   'input_shape': (1, 28, 28), 'input_size': 28*28*1, 'classes': 10})
    config = config | convnet_config
    return config

# SWEEP FOR CONVNETS ON CIFAR10 DATASET
def cnn_cifar(config, convnet_config):
    config.update({"dataset": "CIFAR10", "architecture": "CNN",
                   'input_shape': (3, 32, 32), 'input_size': 32*32*3, 'classes': 10})
    config = config | convnet_config
    return config

# SWEEP FOR RESIDUAL CNN ON MNIST DATASET
def res_cnn_mnist(config, convnet_config):
    config.update({"dataset": "MNIST", "architecture": "ResidualCNN",
                   'input_shape': (1, 28, 28), 'input_size': 28*28*1, 'classes': 10})
    config = config | convnet_config
    return config

# SWEEP FOR RESIDUAL CNN ON CIFAR10 DATASET
def res_cnn_cifar(config, convnet_config):
    config.update({"dataset": "CIFAR10", "architecture": "ResidualCNN",
                   'input_shape': (3, 32, 32), 'input_size': 32*32*3, 'classes': 10})
    config = config | convnet_config
    return config

# SWEEP FOR RESNETS ON MNIST DATASET
def resnet_mnist(config, resnet_config):
    config.update({"dataset": "MNIST", "architecture": "ResNet",
                   'input_shape': (1, 28, 28), 'input_size': 28*28*1, 'classes': 10})
    config.pop()
    return config

# SWEEP FOR RESNETS ON CIFAR10 DATASET
def resnet_cifar(config, resnet_config):
    config.update({"dataset": "CIFAR10", "architecture": "ResNet",
                   'input_shape': (3, 32, 32), 'input_size': 32*32*3, 'classes': 10})
    config = config | resnet_config
    return config