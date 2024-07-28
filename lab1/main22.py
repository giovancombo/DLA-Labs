# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab1

# Code for EXERCISE 2.2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.datasets as datasets
import yaml

from trainer import Trainer

classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Creo il nuovo modello e lo setto in modalità valutazione
    transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_data = datasets.CIFAR10("./data", train = True, download = True, transform = transform)
    test_set = datasets.CIFAR10("./data", train = False, download = True, transform = transform)

    # Splitting the Training Set into Training and Validation Sets
    val_size = 10_000
    train_set, val_set = random_split(train_data, [len(train_data) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size = params['batch_size'], shuffle = True, num_workers = params['num_workers'])
    val_loader = DataLoader(val_set, batch_size = params['batch_size'], num_workers = params['num_workers'])
    test_loader = DataLoader(test_set, batch_size = params['batch_size'], num_workers = params['num_workers'])

    print(f"Dataset {params['dataset']} loaded with {len(train_set)} Train samples, {len(val_set)} Validation samples, {len(test_set)} Test samples.\n")


    # Carico il modello da modificare
    net = torch.load("models/CIFAR10/residualcnn-depth5-ep10-lr0.0001-bs256-dr0.2-1708516763.2621446model.pt")
    # Rimuovo il layer fc e l'Average Pooling
    mod_net = [val for key, val in net._modules.items() if 'rescnn' in key][0]

    # Creo il layer da aggiungere in coda al modello
    class Net(nn.Module):
        def __init__(self, input_shape, classes):
            super(Net, self).__init__()
            
            # Calcoliamo le dimensioni dell'output del modello precedente
            dummy_input = torch.zeros(1, *input_shape).to(device)
            with torch.no_grad():
                output = mod_net(dummy_input)
            _, channels, _, _ = output.shape
            
            self.final = nn.Conv2d(channels, classes, kernel_size = 1)

        def forward(self, x):
            # Adattiamo il reshape in base alle dimensioni calcolate
            x = self.final(x)
            return F.softmax(x, dim = 1)
        
    model = nn.Sequential(mod_net, Net((3,32,32), classes)).to(device)
    model.eval()

    optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])
    criterion = nn.CrossEntropyLoss()


    trainer = Trainer('CIFAR10', model, device, params['log_freq'], params['convnet'])

    for epoch in range(params['epochs']):
        train_loss, train_acc, ct = trainer.train(train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = trainer.evaluate(val_loader, epoch)
