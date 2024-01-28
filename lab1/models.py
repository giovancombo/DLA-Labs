import torch
import torch.nn as nn
import numpy as np

# Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, classes, activation, dropout):
        super(MLP, self).__init__()

        mlp = [nn.Linear(input_size, hidden_size[0]),
               getattr(nn, activation)()]
        for i in range(len(hidden_size) - 1):
            mlp.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            mlp.append(getattr(nn, activation)())
        self.mlp = nn.Sequential(*mlp)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size[-1], classes))

    def forward(self, x):
        out = self.mlp(x)
        out = self.fc(out)
        return out
    

# Multi-Layer Perceptron with residual connections
class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size, classes, activation, dropout):
        super(MLP, self).__init__()

        self.mlp = nn.ModuleList[nn.Linear(input_size, hidden_size[0]),
               getattr(nn, activation)()]
        for _ in range(len(hidden_size) - 1):
            self.mlp.append(nn.Linear(hidden_size[0], hidden_size[0]))
            self.mlp.append(getattr(nn, activation)())
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size[0], classes))

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x) + x
        out = self.fc(x)
        return out


# Convolutional Neural Network: Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))
    
# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, input_shape, hidden_size, classes, depth, kernel_size, stride, padding, activation, dropout, pool, pool_size, use_bn):
        super(CNN, self).__init__()

        self.act = getattr(nn, activation)()
        convlayers = [ConvBlock(input_shape[0], hidden_size[0], kernel_size, stride, padding, use_bn),
                      self.act]
        
        for i in range(len(hidden_size)):
            for _ in range(depth - 1):
                convlayers.append(ConvBlock(hidden_size[i], hidden_size[i], kernel_size, stride, padding, use_bn))
                convlayers.append(self.act)
            convlayers.append(nn.MaxPool2d(pool_size) if pool else nn.Identity())
            if len(hidden_size) > i + 1:
                convlayers.append(ConvBlock(hidden_size[i], hidden_size[i+1], kernel_size, stride, padding, use_bn))
                convlayers.append(self.act)
        self.convlayers = nn.Sequential(*convlayers)

        final_dim = np.square(input_shape[1]) if not pool else np.square(np.floor(input_shape[1] / np.power(pool_size, len(hidden_size)))).astype(int)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size[-1] * final_dim, classes))

    def forward(self, x):
        out = self.convlayers(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    



# Defining the architecture of a Residual Neural Network
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, use_bn, kernel_size = 3, stride = 1, projection = None):
        super(ResidualBlock, self).__init__()

        res_channels = in_channels
        self.conv1 = ConvBlock(in_channels, res_channels, kernel_size, stride, 1, True)
        self.conv2 = ConvBlock(res_channels, out_channels, kernel_size, stride, 1, True)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = getattr(nn, activation)()
        self.projection = projection

    def forward(self, x):
        if self.projection is not None:
            x = self.projection(x)         
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += x
        h = self.act(out)     
        return h
    

class ResidualCNN(nn.Module):
    def __init__(self, input_shape, hidden_size, classes, no_blocks, activation, use_bn):
        super(ResidualCNN, self).__init__()

        rescnn = nn.ModuleList([ConvBlock(input_shape[0], hidden_size[0], 3, 1, 1, True),
                                getattr(nn, activation)()])
        for _ in range(no_blocks - 1):
            rescnn.append(ResidualBlock(hidden_size[0], hidden_size[0], activation, use_bn))
        self.rescnn = nn.Sequential(*rescnn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size[0], classes)

    def forward(self, x):
        x = self.avgpool(self.rescnn(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

    

# Implementing an actual ResNet
class ResNet(nn.Module):
    def __init__(self, name, input_shape, hidden_size, classes, activation, use_bn, dropout):
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
        
        self.conv1 = ConvBlock(input_shape[0], hidden_size[0], 3, 1, 1, True)
        self.act = getattr(nn, activation)()

        self.layer1 = self._make_layer(ResidualBlock, no_blocks[0], hidden_size[0], 1, activation, use_bn)
        self.layer2 = self._make_layer(ResidualBlock, no_blocks[1], hidden_size[1], 2, activation, use_bn)
        self.layer3 = self._make_layer(ResidualBlock, no_blocks[2], hidden_size[2], 2, activation, use_bn) 
        self.layer4 = self._make_layer(ResidualBlock, no_blocks[3], hidden_size[3], 2, activation, use_bn)  
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))          # Average Pool 1x1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size[-1], classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _make_layer(self, ResidualBlock, no_blocks, out_channels, stride, activation):
        projection = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            projection = nn.Sequential(ConvBlock(self.in_channels, out_channels, 1, stride, 0, True),        # 1x1 Convolution with stride = 2 (better alternative to MaxPooling for matching dimensions)
                                       self.act)
            layers.append(projection)
        self.in_channels = out_channels
        layers.append(ResidualBlock(self.in_channels, out_channels, activation))   
        for _ in range(no_blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels, activation))
        return nn.Sequential(*layers)
    


# class FullyCNN(nn.Module):
#     def __init__(self, input_shape, hidden_size, classes, depth, kernel_size, stride, padding, activation, dropout, pool, pool_size, use_bn):
#         super(FullyCNN, self).__init__()

#         self.act = getattr(nn, activation)()
#         convlayers = [ConvBlock(input_shape[0], hidden_size[0], kernel_size, stride, padding, use_bn),
#                       self.act]
        
#         for i in range(len(hidden_size)):
#             for _ in range(depth - 1):
#                 convlayers.append(ConvBlock(hidden_size[i], hidden_size[i], kernel_size, stride, padding, use_bn))
#                 convlayers.append(self.act)
#             convlayers.append(nn.MaxPool2d(pool_size) if pool else nn.Identity())
#             if len(hidden_size) > i + 1:
#                 convlayers.append(ConvBlock(hidden_size[i], hidden_size[i+1], kernel_size, stride, padding, use_bn))
#                 convlayers.append(self.act)
#         self.convlayers = nn.Sequential(*convlayers)

#         # final_dim = np.square(input_shape[1]) if not pool else np.square(np.floor(input_shape[1] / np.power(pool_size, len(hidden_size)))).astype(int)
#         # self.fc = nn.Sequential(
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_size[-1] * final_dim, classes))

#         # Kernel, Stride and Padding values of ConvTranspose were found through calculations in order to have 28 as output size
#         self.head = nn.Sequential(ConvBlock(hidden_size[-1], classes, 1, 1, 0, use_bn),
#                                     self.act,
#                                     nn.ConvTranspose2d(classes, classes, 5, 4, 1))

#     def forward(self, x):
#         out = self.convlayers(x)
#         out = self.head(out)
#         return out