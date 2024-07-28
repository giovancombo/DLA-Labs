# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, classes, dropout):
        super(MLP, self).__init__()

        mlp = [nn.Linear(input_size, hidden_size[0]),
               nn.ReLU()]
        for i in range(len(hidden_size) - 1):
            mlp.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size[-1], classes))

    def forward(self, x):
        out = self.mlp(x)
        out = self.head(out)
        return out
    

# Multi-Layer Perceptron with residual connections
class ResidualMLP(nn.Module):
    def __init__(self, depth, input_size, hidden_size, classes, dropout):
        super(ResidualMLP, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.mlp = nn.ModuleList([nn.Linear(hidden_size[0], hidden_size[0]) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size[0], classes))

    def forward(self, x):
        x = F.relu(self.l1(x))
        for layer in self.mlp:
            x = F.relu(layer(x) + x)
        out = self.head(x)
        return out


# Convolutional Neural Network: Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
    
# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, input_shape, hidden_size, classes, depth, dropout, kernel_size = 3, stride = 1, padding = 1, pool_size = 2):
        super(CNN, self).__init__()

        convlayers = [ConvBlock(input_shape[0], hidden_size[0], kernel_size, stride, padding),
                      nn.ReLU()]
        
        for i in range(len(hidden_size)):
            for _ in range(depth - 1):
                convlayers.append(ConvBlock(hidden_size[i], hidden_size[i], kernel_size, stride, padding))
                convlayers.append(nn.ReLU())
            convlayers.append(nn.MaxPool2d(pool_size))
            if len(hidden_size) > i + 1:
                convlayers.append(ConvBlock(hidden_size[i], hidden_size[i+1], kernel_size, stride, padding))
                convlayers.append(nn.ReLU())
        self.features = nn.Sequential(*convlayers)

        final_dim = np.square(np.floor(input_shape[1] / np.power(pool_size, len(hidden_size)))).astype(int)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size[-1] * final_dim, classes))

    def forward(self, x):
        cnn_out = self.features(x)
        out = torch.flatten(cnn_out, 1)
        out = self.head(out)
        return out
    


# Defining the architecture of a Residual Neural Network
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, projection = None):
        super(ResidualBlock, self).__init__()

        res_channels = in_channels // 4
        self.conv1 = ConvBlock(in_channels, res_channels, kernel_size, stride, 1)
        self.conv2 = ConvBlock(res_channels, out_channels, kernel_size, stride, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.projection = projection

    def forward(self, x):
        if self.projection is not None:
            x = self.projection(x)         
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        h = F.relu(out)     
        return h
    

# Residual Convolutional Neural Network
class ResidualCNN(nn.Module):
    def __init__(self, input_shape, hidden_size, classes, depth):
        super(ResidualCNN, self).__init__()

        rescnn = nn.ModuleList([ConvBlock(input_shape[0], hidden_size[0]),
                                nn.ReLU()])
        for _ in range(depth):
            rescnn.append(ResidualBlock(hidden_size[0], hidden_size[0]))
        self.features = nn.Sequential(*rescnn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size[0], classes)

    def forward(self, x):
        out = self.features(x)
        x = self.avgpool(out)
        gapout = x.view(x.size(0), -1)
        x = self.fc(gapout)
        return x    #, out, gapout
    

    
# Implementing an actual ResNet
class ResNet(nn.Module):
    def __init__(self, name, input_shape, hidden_size, classes, dropout):
        super(ResNet, self).__init__()

        resnet_versions = {
            9: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        no_blocks = resnet_versions[name]
        self.in_channels = hidden_size[0]
        
        self.conv1 = ConvBlock(input_shape[0], hidden_size[0])
        self.act = nn.ReLU()

        self.layer1 = self._make_layer(ResidualBlock, no_blocks[0], hidden_size[0], 1)
        self.layer2 = self._make_layer(ResidualBlock, no_blocks[1], hidden_size[1], 2)
        self.layer3 = self._make_layer(ResidualBlock, no_blocks[2], hidden_size[2], 2) 
        self.layer4 = self._make_layer(ResidualBlock, no_blocks[3], hidden_size[3], 2)  

        self.features = nn.Sequential(self.conv1, self.act, self.layer1, self.layer2, self.layer3, self.layer4)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))                  # Average Pool 1x1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size[-1], classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _make_layer(self, ResidualBlock, no_blocks, out_channels, stride):
        projection = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            projection = nn.Sequential(ConvBlock(self.in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),        # 1x1 Convolution with stride = 2 (better alternative to MaxPooling for matching dimensions)
                                       self.act)
            layers.append(projection)
        self.in_channels = out_channels
        layers.append(ResidualBlock(self.in_channels, out_channels))   
        for _ in range(no_blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)
