# Imports and dependencies
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import feature_extractor
from shallowclf import *

# Generalizing code for handling different datasets and models
datasets = ['ag_news', 'dair-ai/emotion']
dataset_name = datasets[1]
pretrained_model = 'distilbert-base-uncased'

# Features Extraction from the dataset (to be done only the first time: following times we can load the saved features)
extract_features = False         # If True, extracts the features from the dataset and saves them again
batch_size = 64

# Saving configuration
save_classification = False
folder = f'31_textclassification/{dataset_name}'

# Hyperparameters for the shallow classifier (MLP) instantiated
hidden_size = 512
epochs = 20
learning_rate = 1e-3

# Weighted Loss for handling class imbalance
mlp = True
weighted = False

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


if __name__ == '__main__':
    # 1) Loading the selected dataset using the load_dataset function from Hugging Face datasets
    data = load_dataset(dataset_name)
    print(f"Dataset \'{dataset_name}\' loaded!")

    # 2) Instantiating the Tokenizer to tokenize the raw data, and the pretrained model
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
    model = DistilBertModel.from_pretrained(pretrained_model).to(device)

    # Loading features and labels from the folder
    train_features, test_features, train_labels, test_labels = feature_extractor(data, model, tokenizer,
                                                                                 batch_size,
                                                                                 folder,
                                                                                 extract = extract_features,
                                                                                 save = save_classification)

    if mlp:
        shallowcls = MLP(hidden_size).to(device)
        optimizer = torch.optim.Adam(shallowcls.parameters(), lr = learning_rate)

        if weighted:
            weights = torch.tensor([3.43, 2.98, 12.27, 7.41, 8.26, 27.97]).to(device)
        else:
            weights = None
        criterion = nn.CrossEntropyLoss(weight = weights)

        print("Starting Training the MLP...")
        training_mlp(shallowcls, optimizer, criterion, epochs, batch_size, train_features, train_labels, test_features, test_labels, device)
    else:
        print("Training a Logistic Regression model...")
        logreg = LogisticRegression(max_iter = 1500).fit(train_features, train_labels)

        print("Model trained! Making predictions on new data...\n")
        pred = logreg.predict(test_features)

        accuracy = accuracy_score(test_labels, pred)*100
        f1 = f1_score(test_labels, pred, average = 'weighted')
        precision = precision_score(test_labels, pred, average = 'weighted')
        recall = recall_score(test_labels, pred, average = 'weighted')

        print(f"Prediction completed! Test Accuracy = {accuracy:.2f}%\nF1 Score = {f1:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}")
