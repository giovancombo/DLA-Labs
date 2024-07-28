# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

# Code for EXERCISE 3.1

import torch
import torch.nn as nn
from torch.optim import Adam
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools

from my_utils import feature_extractor
from mlp import MLP, training_mlp


# Generalizing code for handling different datasets and models
datasets = ['ag_news', 'dair-ai/emotion']
name = datasets[1]
labels = ["World", "Sports", "Business", "Science/Tech"] if name == datasets[0] else ["sadness", "joy", "love", "anger", "fear", "surprise"]
classes = len(labels)
pretrained_model = 'distilbert-base-uncased'

# Features Extraction from the dataset
extract_features = False                # If True, extracts the features from the dataset and saves them again
batch_size = 64

mlp = False
weighted = False                        # Weighted Loss for handling class imbalance

# HPs for training
hidden_size = 512
epochs = 20
learning_rate = 1e-3
logreg_max_iter = 1000

# Saving configuration
save_classification = False
folder = f'31_textclassification/{name}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # 1) Loading the selected dataset using the load_dataset function from HuggingFace datasets
    data = load_dataset(name)
    print(f"Dataset \'{name}\' loaded!")

    # 2) Instantiating the Tokenizer and the pretrained model
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
    model = DistilBertModel.from_pretrained(pretrained_model).to(device)

    # Loading features and labels from the folder
    train_features, test_features, train_labels, test_labels = feature_extractor(data, model, tokenizer, batch_size, folder, extract = extract_features, save = save_classification)

    # Training with MLP or Logistic Regression
    if mlp:
        shallow_clf = MLP(hidden_size, classes).to(device)
        optimizer = Adam(shallow_clf.parameters(), lr = learning_rate)
        weights = torch.tensor([3.43, 2.98, 12.27, 7.41, 8.26, 27.97]).to(device) if weighted else None
        criterion = nn.CrossEntropyLoss(weight = weights)
        print("Training an MLP...")
        training_mlp(shallow_clf, optimizer, criterion, epochs, batch_size, train_features, train_labels, test_features, test_labels, device)
    else:
        print("Training a Logistic Regression model...")
        logreg = LogisticRegression(max_iter = logreg_max_iter).fit(train_features, train_labels)

        print("Model trained! Making predictions on new data...\n")
        pred = logreg.predict(test_features)

        # Metrics for Logistic Regression
        accuracy = accuracy_score(test_labels, pred) * 100
        f1 = f1_score(test_labels, pred, average = 'weighted')
        precision = precision_score(test_labels, pred, average = 'weighted')
        recall = recall_score(test_labels, pred, average = 'weighted')
        print(f"Prediction completed! Test Accuracy = {accuracy:.2f}%\nF1 Score = {f1:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}")

        # Plotting TSNE and Confusion Matrix
        tsne_train = TSNE(n_components = 2, random_state = 1492).fit_transform(train_features[:1000])
        plt.figure(figsize = (10, 10))
        plt.scatter(tsne_train[:, 0], tsne_train[:, 1], c = train_labels[:1000], label = labels, cmap = "viridis")
        plt.title(f"TSNE plot of train features for {name} Classification")
        plt.colorbar()
        plt.legend()
        plt.savefig(f'images/{name[:7]}/{name[:7]}-tsne_train.png')
        plt.show()

        tsne_val = TSNE(n_components = 2, random_state = 1492).fit_transform(test_features[:1000])
        plt.figure(figsize = (10, 10))
        plt.scatter(tsne_val[:, 0], tsne_val[:, 1], c = test_labels[:1000], label = labels, cmap = "magma")
        plt.title(f"TSNE plot of test features for {name} Classification")
        plt.colorbar()
        plt.legend()
        plt.savefig(f'images/{name[:7]}/{name[:7]}-tsne_val.png')
        plt.show()

        # Create confusion matrix
        cm = confusion_matrix(test_labels, pred)
        # Plot confusion matrix
        plt.figure(figsize = (8, 8))
        plt.imshow(cm, interpolation = "nearest", cmap = plt.cm.Blues)
        plt.title(f"Confusion Matrix for {name} Classification")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation = 45)
        plt.yticks(tick_marks, labels)
        fmt = "d"
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment = "center",
                    color = "white" if cm[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(f'images/{name[:7]}/{name[:7]}-confusion_matrix.png')
        plt.show()
