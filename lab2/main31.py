# Imports and dependencies
import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

# Generalizing code for handling different datasets and models
datasets = ['ag_news', 'dair-ai/emotion']
dataset_name = datasets[1]
pretrained_model = 'distilbert-base-uncased'

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


if __name__ == '__main__':
    # 1) Loading the selected dataset using the load_dataset function from Hugging Face datasets
    data = load_dataset(dataset_name)
    print(f"Dataset \'{dataset_name}\' loaded!")

    # 2) Instantiate the Tokenizer to tokenize the raw data
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)

    # 3) Instantiate the pre-trained Model
    model = DistilBertModel.from_pretrained(pretrained_model).to(device)

    print(f"\n{tokenizer.__class__.__name__} and {model.__class__.__name__} instantiated!")


    # Features Extraction from the dataset (to be done only the first time: following times we can load the saved features)
    extract_features = False         # If True, extracts the features from the dataset and saves them again
    batch_size = 64

    # Saving configuration
    save_classification = False
    folder = f'3_1_text_classification/{dataset_name}'

    if extract_features:
        
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        # Extracting features from the Train dataset
        for i in tqdm(range(0, len(data['train']), batch_size), desc = 'Extracting Train Features'):
            
            encoded_inputs = tokenizer(data['train']['text'][i : i + batch_size], padding = True, truncation = True, return_tensors = 'pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            encoded_features = outputs.last_hidden_state.mean(dim = 1).detach().cpu().numpy()

            train_features.append(encoded_features)
            train_labels.append(data['train']['label'][i : i + batch_size])

        # Extracting features from the Test dataset
        for i in tqdm(range(0, len(data['test']), batch_size), desc = 'Extracting Test Features'):
            
            encoded_inputs = tokenizer(data['test']['text'][i : i + batch_size], padding = True, truncation = True, return_tensors = 'pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            encoded_features = outputs.last_hidden_state.mean(dim = 1).detach().cpu().numpy()
            
            test_features.append(encoded_features)
            test_labels.append(data['test']['label'][i : i + batch_size])

        # Concatenating the lists of features and labels into a single NumPy array
        # train_features is now a (16000, 768) array: 768 is the embedding dimension
        train_features = np.concatenate(train_features, axis = 0)
        train_labels = np.concatenate(train_labels, axis = 0)

        test_features = np.concatenate(test_features, axis = 0)
        test_labels = np.concatenate(test_labels, axis = 0)

        if save_classification:
            print(f"\nSaving features and labels in \'{folder}\'...")
            if not os.path.exists(folder):
                os.makedirs(folder)

            np.save(f'{folder}/train_features.npy', train_features)
            np.save(f'{folder}/test_features.npy', test_features)
            np.save(f'{folder}/train_labels.npy', train_labels)
            np.save(f'{folder}/test_labels.npy', test_labels)

            print('Features and Labels saved!')
    else:
        print(f"\nLoading features and labels from \'{folder}\'...")

        train_features = np.load(f'{folder}/train_features.npy')
        test_features = np.load(f'{folder}/test_features.npy')
        train_labels = np.load(f'{folder}/train_labels.npy')
        test_labels = np.load(f'{folder}/test_labels.npy')

    
    # Hyperparameters for the shallow classifier (MLP) instantiated
    hidden_size = 512
    epochs = 20
    learning_rate = 1e-3

    # Weighted Loss for handling class imbalance
    weighted = False

    class MLP(nn.Module):
        def __init__(self, hidden_size):
            super(MLP, self).__init__()

            self.fc1 = nn.Linear(768, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 6)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        

    # Instantiating Model, Optimizer and Loss
    shallowcls = MLP(hidden_size).to(device)
    optimizer = torch.optim.Adam(shallowcls.parameters(), lr = learning_rate)

    if weighted:
        weights = torch.tensor([3.43, 2.98, 12.27, 7.41, 8.26, 27.97]).to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight = weights)

    # Training Loop
    print("Starting Training...")
    for epoch in range(epochs):
        shallowcls.train()
        for i in range(0, len(train_features), batch_size):
            X = torch.tensor(train_features[i : i + batch_size], dtype = torch.float).to(device)
            Y = torch.tensor(train_labels[i : i + batch_size], dtype = torch.long).to(device)

            outputs = shallowcls(X)
            loss = criterion(outputs, Y)

            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

        # Time for Testing
        total, correct = 0, 0
        shallowcls.eval()
        for i in range(0, len(test_features), batch_size):
            Xval = torch.tensor(test_features[i : i + batch_size], dtype = torch.float).to(device)
            Yval = torch.tensor(test_labels[i : i + batch_size], dtype = torch.long).to(device)

            outputs = shallowcls(Xval)   
            _, pred = torch.max(outputs.data, 1)
            total += Yval.size(0)
            correct += (pred == Yval).sum().item()

        test_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}:\tTraining Loss = {loss.item():.4f}   Test Accuracy = {test_accuracy:.2f}%")

    print("\nTraining completed!")


    print("Training a Logistic Regression model...")
    logreg = LogisticRegression(max_iter = 1500).fit(train_features, train_labels)

    print("Model trained! Making predictions on new data...\n")
    pred = logreg.predict(test_features)

    accuracy = accuracy_score(test_labels, pred)*100
    f1 = f1_score(test_labels, pred, average = 'weighted')
    precision = precision_score(test_labels, pred, average = 'weighted')
    recall = recall_score(test_labels, pred, average = 'weighted')

    print(f"Prediction completed! Test Accuracy = {accuracy:.2f}%\nF1 Score = {f1:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}")
